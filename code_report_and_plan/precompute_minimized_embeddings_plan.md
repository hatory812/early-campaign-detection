# 実装計画：未来情報リーク防止のための埋め込み再計算

## 1. 問題の整理

### 現状のリーク源

`/hss01/A.hattori/rww_all_graphs/degree/mid/` に保存済みの `degree_embedding`（128次元）は、
**フルグラフ全体**のトポロジーに基づきRWWで計算されたものである。

`train_twitter_snapshot.py` でエッジを時間幅 `t_w` で絞った場合でも、
ノード特徴量にはフルグラフ由来の埋め込みが使われているため、
`t_w` 以降の未来のエッジ情報が特徴量に混入している（未来情報リーク）。

### 修正方針

各 `t_w` に対して部分グラフを構築し、**その部分グラフ上で**
`degree_value` 再計算 → RWW → Word2Vec の計算を行い、
`degree_embedding` を差し替えた JSON を保存する。

---

## 2. t_w の範囲

| 範囲 | ステップ | 値の個数 | 具体的な値 |
|---|---|---|---|
| 1〜60 分 | 1分刻み | 60 | 1, 2, 3, ..., 60 |
| 60〜1440 分 | 60分刻み | 24 | 120, 180, ..., 1440 |
| **合計（重複除去）** | | **83** | 1, 2, ..., 60, 120, 180, ..., 1440 |

※ `t_w=60` は1分刻みに含まれるため、60分刻み側は `120` から始める。

---

## 3. 計算パイプライン

### 1グラフ × 1 t_w の処理フロー

```
/hss01/A.hattori/all_graphs/{graph}_fulldata.json   ← 元データ（タイムスタンプ付き、埋め込みなし）
        │
        ▼
① エッジフィルタ
   t0 = min(edge.timestamp)  ← グラフごとに算出
   mask: edge.timestamp ≤ t0 + t_w * 60
        │
        ▼
② 部分グラフ構築（NetworkX DiGraph）
   生き残ったエッジの両端ノードのみ残す（孤立ノードは除外）
   エッジが0本 → スキップ（ログ記録）
        │
        ▼
③ degree_value 再計算（get_degree on restricted subgraph）
   無向グラフ化 → 各ノードの次数を計算 → Min-max正規化
   ※ フルグラフ由来の degree_value を上書き
        │
        ▼
④ RWW（get_rww）実行 on restricted subgraph
   walk_length=100, num_walks=1
   重みは degree_value に基づき高次数ノードへバイアス（ pick='degree', comp='mid' ）
        │
        ▼
⑤ Word2Vec（get_embedding）
   vector_size=128, epochs=100, window=5, hs=1, seed=42
   → 各ノードの degree_embedding を更新（128次元）
        │
        ▼
⑥ JSON 保存（nx.node_link_data 形式）
   /hss01/A.hattori/rww_all_graphs_minimized/{t_w}min/{graph}_fulldata.json
```

### 固定する特徴量（再計算不要）

| フィールド | 次元 | 理由 |
|---|---|---|
| `node_attr` | 772 | テキスト埋め込みなどコンテンツベース。グラフ構造に依存しない |
| `edge_attr` | 776 | エッジのテキスト特徴。内容は不変 |

---

## 4. 出力ディレクトリ構造

```
/hss01/A.hattori/rww_all_graphs_minimized/
├── 1min/
│   ├── #19Mayıs1919_noncampaign_fulldata.json   ← 1分以内エッジ＋再計算済み degree_embedding
│   ├── #1MayıstaMüjde___2023-05-01_campaign_fulldata.json
│   └── ...（約300ファイル）
├── 2min/
│   └── ...
│   ...
├── 60min/
│   └── ...
├── 120min/
│   └── ...
│   ...
└── 1440min/
    └── ...（約300ファイル）
```

合計フォルダ数：**83**、総ファイル数：**約 83 × 300 = 約24,900 ファイル**

---

## 5. 作成するスクリプト

**ファイル名：** `precompute_minimized_embeddings.py`（新規）

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--src_path` | `/hss01/A.hattori/all_graphs` | 元グラフJSON格納先 |
| `--label_path` | `/hss01/A.hattori/all_graphs` | `graph_labels.json` のあるディレクトリ |
| `--out_base` | `/hss01/A.hattori/rww_all_graphs_minimized` | 出力先ルート |
| `--pick` | `degree` | 埋め込みの種類（degree固定） |
| `--comp` | `mid` | RWWの比較パラメータ（mid固定） |
| `--log_dir` | `/hss01/A.hattori/log` | ログ出力ディレクトリ（t_wごとに個別ファイル生成） |

### スクリプト構造

```
import multiprocessing as mp  ← 追加

setup_logger()        ← 変更なし
load_and_filter()     ← 変更なし（旧形式 'links' キーに対応済み）
process_one_graph()   ← 変更なし（del + gc.collect() 済み）
build_tw_values()     ← 変更なし

process_tw()          ← 新規追加（Pool のワーカー関数）

if __name__ == '__main__':   ← spawn モード必須のガード
    main()
```

### 新規追加：`process_tw()` ワーカー関数

`Pool.map` から呼ばれるトップレベル関数。
logger はプロセス間で渡せないため内部でセットアップする。

```python
def process_tw(args):
    t_w, src_path, label_path, out_base, pick, comp, log_path = args

    logger = setup_logger(log_path)  # ワーカー内で個別にセットアップ

    with open(os.path.join(label_path, 'graph_labels.json')) as f:
        graph_labels = json.load(f)
    files = sorted(glob.glob(os.path.join(src_path, '*_fulldata.json')))

    out_dir = os.path.join(out_base, f'{t_w}min')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"--- t_w={t_w}min start ---")

    n_ok = n_existing = n_skip = 0
    for file_path in files:
        file_name = os.path.basename(file_path)[:-5]
        if file_name in EXCEPTIONS or file_name[:-9] not in graph_labels:
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

    logger.info(f"t_w={t_w}min done: saved={n_ok}, "
                f"already_exists={n_existing}, skipped/error={n_skip}")
```

### 改修：`main()` ティア別 `Pool.map`

```python
if __name__ == '__main__':
    mp.set_start_method('spawn')  # gensim との fork 相性問題を回避

    # ... argparse は現状どおり ...

    def make_args(tw_values):
        return [
            (tw, args.src_path, args.label_path, args.out_base,
             args.pick, args.comp,
             os.path.join(args.log_dir, f'precompute_tw{tw}.log'))
            for tw in tw_values
        ]

    # ティア1：small（t_w=1〜20、同時10プロセス・メモリ小）
    with mp.Pool(processes=10) as pool:
        pool.map(process_tw, make_args(range(1, 21)))

    # ティア2：medium（t_w=21〜60 + 120、同時5プロセス）
    with mp.Pool(processes=5) as pool:
        pool.map(process_tw, make_args(list(range(21, 61)) + [120]))

    # ティア3：large（t_w=180〜1440、同時3プロセス・メモリ大）
    with mp.Pool(processes=3) as pool:
        pool.map(process_tw, make_args(range(180, 1441, 60)))
```

---

## 6. 実行コスト見積もり

### 1グラフあたりの処理時間（概算）

| t_w | グラフ規模 | RWW+W2V の目安 |
|---|---|---|
| 小さい t_w（1〜10分） | ノード数十〜数百 | 数秒 |
| 中程度（30〜60分） | ノード数百〜数千 | 十数秒 |
| 大きい t_w（720〜1440分） | ノード数千〜数万 | 数十秒〜数分 |

### 総計

| ケース | 同時プロセス数 | 合計（概算） |
|---|---|---|
| 直列実行 | 1 | **約207時間**（非現実的） |
| Python multiprocessing ティア1（small） | 10 | 約15時間（t_w=1〜20） |
| Python multiprocessing ティア2（medium） | 5 | 約25時間（t_w=21〜120） |
| Python multiprocessing ティア3（large） | 3 | 約30時間（t_w=180〜1440） |
| **合計（ティア逐次）** | | **約70時間** |

※ ティアは順番に実行（tier1完了→tier2→tier3）のためメモリ使用量が重複しない。

---

## 7. 並列実行戦略

### 採用方式：Python multiprocessing（標準ライブラリ）

Slurm が使用不可（Singularity コンテナ内）のため、
`multiprocessing.Pool` を用いて **t_w 単位でプロセス並列化**する。
各ワーカーが1つの t_w を担当し、その t_w の約300グラフを逐次処理する。

```
メインプロセス
    ├── Worker（t_w=1）  300グラフを逐次処理
    ├── Worker（t_w=2）  300グラフを逐次処理
    ├── ...
    └── Worker（t_w=N）  300グラフを逐次処理
        （最大N個まで同時実行）
```

### start method：spawn（必須）

`fork`（Linux デフォルト）は gensim Word2Vec 内部のCスレッドと相性が悪く
デッドロックの原因になるため、`spawn` を明示指定する。

| | fork | spawn |
|---|---|---|
| 仕組み | 親プロセスをそのままコピー | 新しいPythonを起動してre-import |
| 速度 | 速い | 遅い（起動コストは処理時間に対して無視できる） |
| gensim との相性 | ❌ | ✅ |

### メモリ対策：ジョブ内の明示的解放

各グラフ処理後に `del` + `gc.collect()` を呼び、
gensim の Word2Vec モデルや NetworkX グラフオブジェクトが
ループをまたいで蓄積しないようにする（`process_one_graph()` の finally 節に実装済み）。

### ティア分割：t_w の大きさに応じて Pool サイズを段階的に絞る

t_w が大きいほど部分グラフがフルグラフに近づきメモリ使用量が増える。
3ティアに分けて Pool サイズを変え、ティア同士は**順番に実行**することで
メモリ使用量のピークを抑える。

| ティア | t_w の範囲 | t_w 数 | Pool サイズ | メモリ目安 |
|---|---|---|---|---|
| small  | 1〜20 分    | 20 | 10 | 2 GB × 10 = 20 GB |
| medium | 21〜60 分、120 分 | 41 | 5  | 6 GB × 5  = 30 GB |
| large  | 180〜1440 分 | 22 | 3  | 16 GB × 3 = 48 GB |

### 実行コマンド

```bash
cd /home/A.hattori/ECMLPKDD25
source /home/A.hattori/myenv/bin/activate

python3 precompute_minimized_embeddings.py \
    --src_path   /hss01/A.hattori/all_graphs \
    --label_path /hss01/A.hattori/all_graphs \
    --out_base   /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree --comp mid \
    --log_dir    /hss01/A.hattori/log
```

### エラー検知

`pool.map()` はワーカーで発生した例外を自動でメインプロセスに伝播するため、
選択肢C（`&` + `wait`）と異なり失敗を見逃さない。

```python
try:
    with mp.Pool(processes=10) as pool:
        pool.map(process_tw, make_args(range(1, 21)))
except Exception as e:
    print(f"ワーカーで例外発生: {e}")  # 自動でキャッチ
```

### 動作確認用コマンド（小規模テスト）

```bash
# t_w=1〜3 のみ Pool(3) で処理して動作確認
# main() 内の make_args(range(1, 4)) に一時的に変更して実行
python3 precompute_minimized_embeddings.py \
    --src_path   /hss01/A.hattori/all_graphs \
    --label_path /hss01/A.hattori/all_graphs \
    --out_base   /tmp/test_minimized \
    --pick degree --comp mid \
    --log_dir    /tmp/log_test
```

---

## 8. 動作確認チェックリスト

- [ ] 出力 JSON の `nodes[i].degree_embedding` が128次元になっているか
- [ ] フルグラフの `degree_embedding` と値が異なっているか（再計算されているか）
- [ ] `t_w=1` など極小グラフでもエラーなく完了するか
- [ ] スキップされたグラフがログに記録されているか
- [ ] 出力フォルダに約300ファイルが揃っているか（`ls | wc -l`）

---

## 9. 後続タスク（本スクリプト完成後）

`train_twitter_snapshot.py` を修正し、`--all_graphs_path` に
`/hss01/A.hattori/rww_all_graphs_minimized/{t_w}min` を渡すことで、
各 `t_w` で **リークのない埋め込みを使った学習・評価** が可能になる。

具体的には `build_cache()` の代わりに、各 `t_w` で対応フォルダの
JSONを直接 `load_data` する方式に切り替える。

---

## 10. 成果物一覧

| ファイル | 種別 | 説明 |
|---|---|---|
| `precompute_minimized_embeddings.py` | スクリプト（改修） | 埋め込み再計算・保存。multiprocessing によるティア並列化を内蔵 |
| `/hss01/A.hattori/rww_all_graphs_minimized/{t_w}min/` | データ（83フォルダ） | 再計算済みグラフJSON（約24,900ファイル） |
| `/hss01/A.hattori/log/precompute_tw{t_w}.log` | ログ（t_wごと） | スキップ・エラー記録（83ファイル） |
