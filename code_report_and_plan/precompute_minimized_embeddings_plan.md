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
| `--tw_list` | （後述） | 処理する t_w 値をカンマ区切りで指定（並列実行用） |
| `--log_path` | `./log_precompute.txt` | スキップ・エラーのログ出力先 |

### スクリプト骨格

```python
# kcore_rww.py から流用する関数
from kcore_rww import get_degree, get_rww, get_embedding

def load_and_filter(file, t_w):
    """元JSONを読み込み、t_w 分以内のエッジに絞った NetworkX DiGraph を返す."""
    with open(file, 'r') as f:
        data = json.load(f)
    graph = nx.DiGraph(json_graph.node_link_graph(data))
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    # タイムスタンプでフィルタ
    timestamps = [d['timestamp'] for _, _, d in graph.edges(data=True)]
    t0 = min(timestamps)
    threshold = t0 + t_w * 60
    remove_edges = [(u, v) for u, v, d in graph.edges(data=True)
                    if d['timestamp'] > threshold]
    graph.remove_edges_from(remove_edges)

    # 孤立ノード削除
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)

    if graph.number_of_edges() == 0:
        return None  # スキップ対象
    return graph

def main():
    # t_w リストの構築
    tw_values = list(range(1, 61)) + list(range(120, 1441, 60))  # 83値

    # --tw_list 指定時はその値のみ処理（並列実行用）
    if args.tw_list:
        tw_values = [int(x) for x in args.tw_list.split(',')]

    files = glob(args.src_path + '/*_fulldata.json')
    label_path = args.label_path + '/graph_labels.json'
    with open(label_path) as f:
        graph_labels = json.load(f)

    for t_w in tw_values:
        out_dir = f"{args.out_base}/{t_w}min"
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            file_name = file.split('/')[-1][:-5]
            if file_name in EXCEPTIONS or file_name[:-9] not in graph_labels:
                continue

            graph = load_and_filter(file, t_w)
            if graph is None:
                log(f"SKIP t_w={t_w} {file_name}: 0 edges after filter")
                continue

            graph = get_degree(graph)               # ③
            walks = get_rww(graph, 'degree', 'mid') # ④
            graph = get_embedding(walks, graph, 'degree')  # ⑤

            out_json = nx.node_link_data(graph)
            out_path = f"{out_dir}/{file_name}_fulldata.json"
            with open(out_path, 'w') as f:
                json.dump(out_json, f)
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

| ケース | t_w 数 | グラフ数 | 1処理 | 合計（概算） |
|---|---|---|---|---|
| 直列実行 | 83 | 300 | 平均30秒 | **約207時間**（現実的でない） |
| 83並列（t_w単位） | 1ずつ | 300 | 平均30秒 | **約1.5時間/t_w**（計83ジョブ同時） |
| 推奨：グラフ単位並列 | 1 t_w | 1ずつ | 平均30秒 | **約150分/t_w** → 並列で短縮 |

→ **t_w × グラフ の2次元でジョブ配列化**（Slurm Job Array 推奨）

---

## 7. 並列実行戦略

### メモリ対策：ジョブ内の明示的解放

各グラフ処理後に `del` + `gc.collect()` を呼び、gensim の Word2Vec モデルや
NetworkX グラフオブジェクトがループをまたいで蓄積しないようにする。

```python
import gc

for file in files:
    graph = load_and_filter(file, t_w)
    if graph is None:
        continue
    graph = get_degree(graph)
    walks = get_rww(graph, 'degree', 'mid')
    graph = get_embedding(walks, graph, 'degree')
    save_json(graph, out_dir, file_name)

    del graph, walks   # 明示的解放
    gc.collect()       # gensim Word2Vec モデルも回収
```

### Slurm Job Array：ティア分割による同時実行数の制御

t_w が大きいほど部分グラフがフルグラフに近づきメモリ使用量が増えるため、
**t_w の大きさに応じてジョブ配列を3ティアに分割し**、それぞれ同時実行数と
要求メモリを変えて投入する。

| ティア | t_w の範囲 | ジョブ数 | 同時実行上限 | 要求メモリ |
|---|---|---|---|---|
| small  | 1〜20 分    | 20 | 10 | 2 GB |
| medium | 21〜60 分、120 分 | 41 | 5  | 6 GB |
| large  | 180〜1440 分 | 22 | 3  | 16 GB |

#### ティア1：small（t_w = 1〜20）

```bash
#!/bin/bash
#SBATCH --job-name=precomp_small
#SBATCH --mem=2G
# run/degree/precompute_small.sh

TW_LIST=($(seq 1 1 20))
TW=${TW_LIST[$SLURM_ARRAY_TASK_ID]}

python3 ../../precompute_minimized_embeddings.py \
    --src_path /hss01/A.hattori/all_graphs \
    --out_base /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree --comp mid \
    --tw_list $TW

# 投入コマンド
# sbatch --array=0-19%10 precompute_small.sh
```

#### ティア2：medium（t_w = 21〜60、120）

```bash
#!/bin/bash
#SBATCH --job-name=precomp_medium
#SBATCH --mem=6G
# run/degree/precompute_medium.sh

TW_LIST=($(seq 21 1 60) 120)
TW=${TW_LIST[$SLURM_ARRAY_TASK_ID]}

python3 ../../precompute_minimized_embeddings.py \
    --src_path /hss01/A.hattori/all_graphs \
    --out_base /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree --comp mid \
    --tw_list $TW

# 投入コマンド
# sbatch --array=0-40%5 precompute_medium.sh
```

#### ティア3：large（t_w = 180〜1440）

```bash
#!/bin/bash
#SBATCH --job-name=precomp_large
#SBATCH --mem=16G
# run/degree/precompute_large.sh

TW_LIST=($(seq 180 60 1440))
TW=${TW_LIST[$SLURM_ARRAY_TASK_ID]}

python3 ../../precompute_minimized_embeddings.py \
    --src_path /hss01/A.hattori/all_graphs \
    --out_base /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree --comp mid \
    --tw_list $TW

# 投入コマンド
# sbatch --array=0-21%3 precompute_large.sh
```

#### 全ティアの一括投入

```bash
sbatch --array=0-19%10 run/degree/precompute_small.sh
sbatch --array=0-40%5  run/degree/precompute_medium.sh
sbatch --array=0-21%3  run/degree/precompute_large.sh
```

各ティアは独立して動くため、3コマンドを順番に投入すれば同時に走る。
クラスタの空き状況に応じて `%N` の数値を調整すること。

### 動作確認用コマンド（小規模テスト）

```bash
# t_w=60 のみ処理して出力 JSON を確認
python3 precompute_minimized_embeddings.py \
    --src_path /hss01/A.hattori/all_graphs \
    --out_base /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree --comp mid \
    --tw_list 60
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
| `precompute_minimized_embeddings.py` | スクリプト（新規） | 埋め込み再計算・保存 |
| `run/degree/precompute_small.sh`   | 実行スクリプト（新規） | Slurm Job Array 投入用（t_w=1〜20、同時10、2GB） |
| `run/degree/precompute_medium.sh`  | 実行スクリプト（新規） | Slurm Job Array 投入用（t_w=21〜120、同時5、6GB） |
| `run/degree/precompute_large.sh`   | 実行スクリプト（新規） | Slurm Job Array 投入用（t_w=180〜1440、同時3、16GB） |
| `/hss01/A.hattori/rww_all_graphs_minimized/{t_w}min/` | データ（83フォルダ） | 再計算済みグラフJSON |
| `log_precompute.txt` | ログ | スキップ・エラー記録 |
