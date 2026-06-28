"""
precompute_minimized_embeddings.py

各 t_w に対して:
  1. /hss01/A.hattori/all_graphs/ の元JSONを読み込む
  2. timestamp <= t0 + t_w*60 のエッジのみ残す（部分グラフ）
  3. 孤立ノードを除去し、ノードを 0..N-1 へ再ラベル
  4. degree_value を部分グラフ上で再計算（フルグラフ由来の値を上書き）
  5. RWW -> Word2Vec で degree_embedding を再計算（未来情報リーク防止）
  6. rww_all_graphs_minimized/{t_w}min/ に保存

出力フォーマットは rww_all_graphs/degree/mid/ と同じ nx.node_link_data 形式
（'edges' キー）なので、train_twitter_snapshot.py がそのまま読める。

メモリ対策:
  - 各グラフ処理後に del graph, walks + gc.collect() で明示解放
  - Slurm ティア分割により同時実行数を t_w の大きさに応じて制限

実行例（動作確認）:
  python3 precompute_minimized_embeddings.py --tw_list 5
"""

import sys
import os
import json
import gc
import glob
import argparse
import logging
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


def setup_logger(log_path):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_and_filter(file_path, t_w):
    """
    元JSONを読み込み、t_w 分以内のエッジだけ残した NetworkX DiGraph を返す。

    - all_graphs/ は旧 NetworkX 形式（'links' キー）なので edges='links' で読む。
    - エッジフィルタ後に孤立ノードを除去し、ノードを 0..N-1 へ再ラベルする。
      （get_embedding が range(graph.number_of_nodes()) でアクセスするため必須）
    - エッジが 0 本になる場合は None を返す。
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

    # 孤立ノード削除
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)

    if graph.number_of_edges() == 0:
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
            logger.warning(f"SKIP t_w={t_w}  {file_name}: 0 edges after filter")
            return False

        graph = get_degree(graph)
        walks = get_rww(graph, pick, comp)
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


def build_tw_values(tw_list_str):
    """--tw_list の文字列から t_w の整数リストを作る。未指定時は全83値。"""
    if tw_list_str:
        return [int(x.strip()) for x in tw_list_str.split(',')]
    # 1..60（1分刻み）+ 120..1440（60分刻み）= 83値
    return list(range(1, 61)) + list(range(120, 1441, 60))


def main():
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
    parser.add_argument('--tw_list', default=None,
                        help='Comma-separated t_w values (e.g. "1,2,3"). '
                             'Omit to process all 83 values.')
    parser.add_argument('--log_path', default='./log_precompute.txt',
                        help='Log file path')
    args = parser.parse_args()

    logger = setup_logger(args.log_path)
    logger.info(f"=== precompute_minimized_embeddings.py START {datetime.now()} ===")
    logger.info(f"args: {vars(args)}")

    tw_values = build_tw_values(args.tw_list)
    logger.info(f"t_w values ({len(tw_values)} total): {tw_values}")

    label_file = os.path.join(args.label_path, 'graph_labels.json')
    with open(label_file, 'r', encoding='utf-8') as f:
        graph_labels = json.load(f)

    files = sorted(glob.glob(os.path.join(args.src_path, '*_fulldata.json')))
    logger.info(f"Found {len(files)} JSON files in {args.src_path}")

    for t_w in tw_values:
        out_dir = os.path.join(args.out_base, f'{t_w}min')
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"--- t_w={t_w}min  out_dir: {out_dir} ---")

        n_ok = n_existing = n_skip = n_err = 0

        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)[:-5]  # '.json' を除去

            if file_name in EXCEPTIONS:
                continue

            # ラベルチェック: file_name = "xxx_fulldata" → [:-9] で "_fulldata" を除去
            graph_key = file_name[:-9]
            if graph_key not in graph_labels:
                continue

            print(f"\r  t_w={t_w:4d}min [{i+1:3d}/{len(files)}] {file_name[:55]}",
                  end='', flush=True)

            out_path = os.path.join(out_dir, file_name + '.json')
            if os.path.exists(out_path):
                n_existing += 1
                continue  # 冪等性：既存ファイルはスキップ

            success = process_one_graph(
                file_path, file_name, t_w, out_dir,
                args.pick, args.comp, logger
            )
            if success:
                n_ok += 1
            else:
                n_skip += 1

        print()
        logger.info(
            f"t_w={t_w}min done: saved={n_ok}, "
            f"already_exists={n_existing}, skipped/error={n_skip}"
        )

    logger.info(f"=== ALL DONE {datetime.now()} ===")


if __name__ == '__main__':
    main()
