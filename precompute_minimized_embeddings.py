"""
precompute_minimized_embeddings.py

各 t_w に対して:
  1. /hss01/A.hattori/all_graphs/ の元JSONを読み込む
  2. timestamp <= t0 + t_w*60 のエッジのみ残す（部分グラフ）
  3. 孤立ノードを除去し、ノードを 0..N-1 へ再ラベル
  4. degree_value を部分グラフ上で再計算（フルグラフ由来の値を上書き）
  5. RWW -> Word2Vec で degree_embedding を再計算（未来情報リーク防止）
  6. rww_all_graphs_minimized/{t_w}min/ に保存

自己ループのみになるグラフの扱い:
  t_w が小さいと「時間幅内に観測されたエッジが自己ループだけ」になるグラフがある
  （実測: t_w=0 で 15件、t_w=60 で 2件）。自己ループは密度指標の定義から外すため
  除去するが、そこでグラフごと捨てると t_w ごとに入力グラフ数が変わり
  （293件〜308件）、時間幅間で精度を比較できなくなる。しかも脱落は campaign 側に
  偏る（t_w=0 では 15件中 12件）ため、小さい t_w の精度が過大評価される。
  そこで本スクリプトは、そうしたグラフを捨てずに
    「自己ループの端点ノードだけを残した 0 エッジグラフ」
  として保存する（degree_value=0、degree_embedding=ゼロベクトル）。
  「この時間幅ではまだ他者との相互作用が観測されていない」を素直に表現でき、
  全 t_w で入力グラフ数が 308 件に揃う。
  なお 0 エッジグラフに対しては RWW / Word2Vec を呼ばない。語彙サイズ1で
  gensim(hs=1) の Huffman木バグを踏んでハングするため
  （詳細: code_report_and_plan/[Report]precompute_word2vec_hang_final_report.md）。

並列化:
  multiprocessing.Pool を使い t_w 単位でプロセス並列化する。
  gensim Word2Vec との fork 相性問題を避けるため start method は spawn を使用。
  t_w が大きいほどメモリ使用量が増えるため、ティア別に Pool サイズを段階的に絞る。
    ティア1 small  (t_w=  1〜 20): Pool(10)
    ティア2 medium (t_w= 21〜120): Pool( 5)
    ティア3 large  (t_w=180〜900): Pool( 3)
    ティア4 xlarge (t_w=960〜1440): Pool( 5)
  ティアは順番に実行するためメモリのピークが重複しない。

実行例（本番）:
  python3 precompute_minimized_embeddings.py

実行例（テスト：各ティアから1値ずつ確認）:
  python3 precompute_minimized_embeddings.py --test_tw 5,40,300
"""

import sys
import os
import json
import gc
import glob
import argparse
import logging
import multiprocessing as mp
from datetime import datetime

import networkx as nx
from networkx.readwrite import json_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kcore_rww import get_degree, get_rww, get_embedding

# train_twitter_snapshot.py と同じ除外リスト
EXCEPTIONS = [
    'graph_labels',
    'Gomis_noncampaign_fulldata',
    '#Hıdırellez_noncampaign_fulldata',
    '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata',
    'Haluk_noncampaign_fulldata',
    '#ErdenTimurSezonu_noncampaign_fulldata',
    'Gustavo_noncampaign_fulldata',
]

# kcore_rww.get_embedding の Word2Vec vector_size と一致させること。
# 0 エッジグラフのゼロ埋め込みの次元数に使う。
EMBEDDING_DIM = 128

# pick -> (密度値の属性名, 埋め込みの属性名)。kcore_rww.py の命名に合わせる。
ATTR_KEYS = {
    'degree': ('degree_value', 'degree_embedding'),
    'kcore': ('kcore_value', 'structural_embedding'),
    'ktruss': ('ktruss_value', 'truss_embedding'),
}


def set_degenerate_attributes(graph, pick):
    """
    エッジ0本の退化グラフに、密度値0とゼロ埋め込みを直接付与する。

    RWW / Word2Vec は呼ばない。ウォークが「長さ1のウォーク×ノード数」しか
    生成されず語彙が退化するうえ、ノード1個のケースでは gensim(hs=1) の
    Huffman木バグでプロセスがハングするため。
    """
    value_key, embedding_key = ATTR_KEYS[pick]
    for n in graph.nodes():
        graph.nodes[n][value_key] = 0.0
        graph.nodes[n][embedding_key] = [0.0] * EMBEDDING_DIM
    return graph


def setup_logger(log_path, also_stdout=True):
    """
    ロガーをセットアップして返す。
    workers は also_stdout=False で呼ぶ（複数プロセスの stdout 混在を防ぐ）。
    spawn モードでは各ワーカーがこの関数を呼ぶたびにクリーンな状態から始まる。
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    handlers = [logging.FileHandler(log_path, encoding='utf-8')]
    if also_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def load_and_filter(file_path, t_w):
    """
    元JSONを読み込み、t_w 分以内のエッジだけ残した NetworkX DiGraph を返す。

    - all_graphs/ は旧 NetworkX 形式（'links' キー）なので edges='links' で読む。
    - エッジフィルタ後に孤立ノードを除去し、ノードを 0..N-1 へ再ラベルする。
      （get_embedding が range(graph.number_of_nodes()) でアクセスするため必須）
    - 自己ループを除いた実エッジが 1 本も残らない場合は、自己ループの端点ノード
      （= t_w 内に行動があったユーザ）だけを残した 0 エッジグラフを返す。
      呼び出し側は graph.number_of_edges() == 0 で退化ケースを判定する。
    - t_w 内にエッジが 1 本もない（自己ループすらない）場合のみ None を返す。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 旧 NetworkX 形式（'links'）と新形式（'edges'）の両方に対応
    edge_key = 'links' if 'links' in data else 'edges'
    graph = nx.DiGraph(json_graph.node_link_graph(data, edges=edge_key))
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    timestamps = [d['timestamp'] for _, _, d in graph.edges(data=True)
                  if 'timestamp' in d]
    if not timestamps:
        return None

    t0 = min(timestamps)
    threshold = t0 + t_w * 60

    remove_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get('timestamp', 0) > threshold
    ]
    graph.remove_edges_from(remove_edges)

    # 自己ループは密度指標(degree/kcore/ktruss)の定義から外すため常に除去するが、
    # 端点だけは退化ケースの復元用に控えておく。
    selfloop_nodes = {u for u, _ in nx.selfloop_edges(graph)}
    graph.remove_edges_from(nx.selfloop_edges(graph))

    isolates = list(nx.isolates(graph))
    if graph.number_of_edges() > 0:
        graph.remove_nodes_from(isolates)
    else:
        # 退化ケース: 実エッジが 1 本も残らなかった。グラフごと捨てると t_w ごとに
        # 入力グラフ数が変わってしまうため、自己ループを持っていたノードだけ残した
        # 0 エッジグラフとして返す（モジュール冒頭の docstring 参照）。
        graph.remove_nodes_from([n for n in isolates if n not in selfloop_nodes])
        if graph.number_of_nodes() == 0:
            return None

    # ノードを 0..N-1 へ再ラベル（get_embedding の range() アクセスに必須）
    graph = nx.relabel_nodes(graph, {n: i for i, n in enumerate(graph.nodes())})
    return graph


def process_one_graph(file_path, file_name, t_w, out_dir, pick, comp, logger):
    """
    1グラフを処理して out_dir に保存する。
    成功時 True、スキップ/エラー時 False を返す。
    メモリ解放のため finally で graph, walks を明示削除する。
    """
    graph = None
    walks = None
    try:
        graph = load_and_filter(file_path, t_w)
        if graph is None:
            logger.warning(f"SKIP t_w={t_w}  {file_name}: no edges at all after filter")
            return False

        if graph.number_of_edges() == 0:
            # 退化グラフ（自己ループのみ）: RWW / Word2Vec は呼ばずゼロ埋め込みを付ける
            graph = set_degenerate_attributes(graph, pick)
            logger.info(
                f"DEGENERATE t_w={t_w}  {file_name}: "
                f"self-loops only -> {graph.number_of_nodes()} nodes / 0 edges, zero embedding")
        else:
            graph = get_degree(graph)
            walks = get_rww(graph, pick, comp)

            # 防御的チェック: 語彙サイズ1の walks を Word2Vec に渡すと
            # gensim(hs=1) の Huffman木バグでハングする。実エッジが残っていれば
            # ノードは2個以上なので通常は起こらないが、念のためここで止める。
            vocab = {node for walk in walks for node in walk}
            if len(vocab) <= 1:
                graph = set_degenerate_attributes(graph, pick)
                logger.warning(
                    f"DEGENERATE t_w={t_w}  {file_name}: vocabulary size {len(vocab)} "
                    f"-> skipped Word2Vec, zero embedding")
            else:
                graph = get_embedding(walks, graph, pick)

        out_path = os.path.join(out_dir, file_name + '.json')
        graph_json = nx.node_link_data(graph)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(graph_json, f, ensure_ascii=False)

        return True

    except Exception as e:
        logger.error(f"ERROR t_w={t_w}  {file_name}: {e}")
        return False

    finally:
        # gensim Word2Vec モデルや NetworkX グラフを確実に解放する
        try:
            del graph
        except Exception:
            pass
        try:
            del walks
        except Exception:
            pass
        gc.collect()


def process_tw(args):
    """
    Pool のワーカー関数。1つの t_w に対して全グラフを処理する。
    spawn モードで起動されるため、logger はここで個別にセットアップする。
    引数はタプルで受け取る（pool.map は1引数しか渡せないため）。
    """
    t_w, src_path, label_path, out_base, pick, comp, log_path = args

    # ワーカーはファイルのみに記録（stdout は複数プロセスで混在するため）
    logger = setup_logger(log_path, also_stdout=False)
    logger.info(f"=== worker START t_w={t_w}min  {datetime.now()} ===")

    with open(os.path.join(label_path, 'graph_labels.json'), 'r', encoding='utf-8') as f:
        graph_labels = json.load(f)
    files = sorted(glob.glob(os.path.join(src_path, '*_fulldata.json')))

    out_dir = os.path.join(out_base, f'{t_w}min')
    os.makedirs(out_dir, exist_ok=True)

    n_ok = n_existing = n_skip = 0
    for file_path in files:
        file_name = os.path.basename(file_path)[:-5]  # '.json' を除去

        if file_name in EXCEPTIONS:
            continue
        if file_name[:-9] not in graph_labels:
            continue

        out_path = os.path.join(out_dir, file_name + '.json')
        if os.path.exists(out_path):
            n_existing += 1
            continue  # 冪等性：既存ファイルはスキップ

        success = process_one_graph(
            file_path, file_name, t_w, out_dir, pick, comp, logger)
        if success:
            n_ok += 1
        else:
            n_skip += 1

    logger.info(
        f"=== worker DONE  t_w={t_w}min  "
        f"saved={n_ok}, already_exists={n_existing}, skipped/error={n_skip}  "
        f"{datetime.now()} ==="
    )


def make_args(tw_values, src_path, label_path, out_base, pick, comp, log_dir):
    """Pool.map に渡す引数タプルのリストを生成する。"""
    return [
        (tw, src_path, label_path, out_base, pick, comp,
         os.path.join(log_dir, f'precompute_tw{tw}.log'))
        for tw in tw_values
    ]


if __name__ == '__main__':
    # spawn モードを最初に宣言（gensim との fork 相性問題を回避）
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description='Precompute degree embeddings on t_w-restricted subgraphs'
    )
    parser.add_argument('--src_path', default='/hss01/A.hattori/all_graphs',
                        help='Source directory with original graph JSONs')
    parser.add_argument('--label_path', default='/hss01/A.hattori/all_graphs',
                        help='Directory containing graph_labels.json')
    parser.add_argument('--out_base', default='/hss01/A.hattori/rww_all_graphs_minimized',
                        help='Output base directory')
    parser.add_argument('--pick', default='degree',
                        choices=['degree', 'kcore', 'ktruss'],
                        help='Structural attribute for RWW (degree のみ使用)')
    parser.add_argument('--comp', default='mid',
                        help='RWW comparison parameter: mid / median / 0.5')
    parser.add_argument('--log_dir', default='/hss01/A.hattori/log',
                        help='Log directory (one file per t_w: precompute_tw{t_w}.log)')
    parser.add_argument('--test_tw', default=None,
                        help='テスト用：処理する t_w をカンマ区切りで指定 (e.g. "5,40,300"). '
                             '省略時は全83値をティア別 Pool で実行。')
    parser.add_argument('--test_workers', default=1, type=int,
                        help='--test_tw 指定時の Pool プロセス数（既定1=逐次）。'
                             '特定ティアを本番並列度で回す場合に使う。')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    main_logger = setup_logger(
        os.path.join(args.log_dir, 'precompute_main.log'), also_stdout=True)
    main_logger.info(f"=== precompute_minimized_embeddings.py START {datetime.now()} ===")
    main_logger.info(f"args: {vars(args)}")

    kw = dict(
        src_path=args.src_path,
        label_path=args.label_path,
        out_base=args.out_base,
        pick=args.pick,
        comp=args.comp,
        log_dir=args.log_dir,
    )

    if args.test_tw:
        # テストモード：指定 t_w のみ Pool(test_workers) で実行
        tw_values = [int(x.strip()) for x in args.test_tw.split(',')]
        n_workers = max(1, args.test_workers)
        main_logger.info(f"TEST MODE: t_w={tw_values}, Pool({n_workers})")
        with mp.Pool(processes=n_workers) as pool:
            pool.map(process_tw, make_args(tw_values, **kw))

    else:
        # 本番モード：ティア別 Pool（ティアは順番に実行してメモリピークを抑える）

        # ティア1 small：t_w=1〜20（同時10プロセス）
        main_logger.info("--- Tier1 small  t_w=1..20  Pool(10) START ---")
        with mp.Pool(processes=10) as pool:
            pool.map(process_tw, make_args(range(1, 21), **kw))
        main_logger.info("--- Tier1 small  DONE ---")

        # ティア2 medium：t_w=21〜60, 120（同時5プロセス）
        main_logger.info("--- Tier2 medium t_w=21..60,120  Pool(5) START ---")
        with mp.Pool(processes=5) as pool:
            pool.map(process_tw, make_args(list(range(21, 61)) + [120], **kw))
        main_logger.info("--- Tier2 medium DONE ---")

        # ティア3 large：t_w=180〜900（同時3プロセス）
        main_logger.info("--- Tier3 large  t_w=180..900  Pool(3) START ---")
        with mp.Pool(processes=3) as pool:
            pool.map(process_tw, make_args(range(180, 901, 60), **kw))
        main_logger.info("--- Tier3 large  DONE ---")

        # ティア4 xlarge：t_w=960〜1440（同時5プロセス）
        main_logger.info("--- Tier4 xlarge t_w=960..1440  Pool(5) START ---")
        with mp.Pool(processes=5) as pool:
            pool.map(process_tw, make_args(range(960, 1441, 60), **kw))
        main_logger.info("--- Tier4 xlarge DONE ---")

    main_logger.info(f"=== ALL DONE {datetime.now()} ===")
