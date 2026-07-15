# 実装計画: テスト時の予測ログ記録とグラフ可視化

対象ファイル: `train_twitter_MPNN.py`

## 目的（ユーザー要件）
テスト（推論）時に、以下の2つのフローを追加する。

1. **予測ログ**: どのファイルがどの予測を受け、正解は何かをログファイルに記録する。
2. **可視化保存**: 予測ラベルに対応するフォルダを作成し、そこにテストで使われた各グラフの可視化画像を保存する。
   - 画像に含める情報: グラフのネットワーク図、正解ラベル、エッジ数、ノード数。

## 決定済み仕様（確認済み）
- **対象**: `exp=0〜4` の全5回すべて。`exp` ごとにサブフォルダを分ける。
- **枚数**: テストセットの全グラフを保存（上限なし）。
- **出力先（絶対パスに固定）**: 起動時の cwd に依存しないよう、以下の**絶対パス**を出力基底とする。
  ```
  /home/A.hattori/ECMLPKDD25/results/20260715_時間幅0で埋め込み再計算した実験_テスト時のログを可視化/
  ```
  - 既存コードの結果JSON保存先（相対 `results/`）は cwd 依存だが、**本フローの出力はこの絶対パスで固定**する。
  - 再利用のため `--analysis_out` 引数を追加し、**この絶対パスを既定値**にする（明示指定でも上書き可）。

---

## 前提となる重要事実
- 各 `Data` オブジェクトは `data.name = file_name`（元ファイル名、拡張子と末尾サフィックス除去済み）を持つ（`train_twitter_MPNN.py:137`）。
- `predict` の `DataLoader` は `shuffle=False`（`:201`）。バッチはデータセット順に反復され、バッチ内順序も保存されるため、
  返り値 `y_pred[i]` / `y_actual[i]` は `test_data[i]` と 1:1 で対応する。
  → ファイル名・正解・ノード数/エッジ数を後付けで対応付け可能。
- `y_actual[i]` は `test_data[i].y.item()` と一致（同じ順序で `batch.y` から構築されるため）。冗長だが照合に使える。
- `to_networkx` は import 済み（`:28`）。可視化に使用する。
- ノード数は `data.num_nodes`、エッジ数は `data.edge_index.shape[1]` で取得（`edge_index` は `[2, E]` 形状、`:136` で転置済み）。
- **既存の `predict` / `train_model` のロジックは変更しない**。追加は「予測後の後処理」として独立した関数で行う。

## 出力ディレクトリ構造
基底は上記の絶対パス。その下に、結果JSONの命名規則（`:492`）に合わせた設定名
`<model_name>_<rww_attr>_nodeattr<node_attr>_<data_type>_mv<multivariate>` のサブフォルダを置く。

```
<--analysis_out の絶対パス>/<設定名>/
    exp0/
        predictions.log          # ファイル名・予測・正解・正誤・ノード数・エッジ数
        pred_0/                  # 予測ラベル 0 と判定されたグラフの画像
            <グラフ名>.png
        pred_1/                  # 予測ラベル 1 と判定されたグラフの画像
            <グラフ名>.png
        ...                      # 多クラス(multivariate)時は pred_0〜pred_6
    exp1/
    ...
    exp4/
```

- 予測ラベルフォルダ名は `pred_<予測ラベルの整数値>`。
- `predictions.log` はタブ区切りテキスト。ヘッダ + 1グラフ1行:
  `file_name<TAB>predicted<TAB>actual<TAB>correct<TAB>num_nodes<TAB>num_edges`

---

## ステップ1: matplotlib をヘッドレスバックエンドに設定
- 多数の図をサーバ上で保存するため、`import matplotlib.pyplot as plt`（`:40`）の**直前**に
  `import matplotlib; matplotlib.use('Agg')` を追加。
- ファイル名サニタイズ用に `import re` を追加。
- 影響: `getReport` 内の `plt.show()`（`:323`）は現状 `__main__` から呼ばれていないため実害なし。

## ステップ2: 可視化用ヘルパー関数を追加
`save_graph_image(data, name, pred, actual, out_path)` を新設（`getReport` 定義の後、`__main__` の前）。

- `G = to_networkx(data, to_undirected=False)` で有向グラフに変換。
- レイアウトは規模に応じて分岐（全グラフ保存で巨大グラフの計算時間が発散するのを防ぐ）:
  - `num_nodes <= 500`: `nx.spring_layout(G, seed=1)`
  - それ以外: `nx.random_layout(G, seed=1)`（高速）
- `nx.draw` で描画（`node_size` 小、`arrows=True`、`with_labels=False`、エッジ細め）。
- タイトルにユーザー要件の情報を明記:
  `f"{name}\nactual={actual}  pred={pred}\nnodes={num_nodes}  edges={num_edges}"`
- `plt.savefig(out_path, dpi=100, bbox_inches='tight')` → `plt.close(fig)`（メモリリーク防止）。

## ステップ3: ログ + 可視化を束ねる関数を追加
`log_and_visualize(test_data, y_pred, y_actual, run_dir)` を新設。

- `os.makedirs(run_dir, exist_ok=True)`。
- `run_dir/predictions.log` を開き、ヘッダを書く。
- `for i, data in enumerate(test_data):`
  - `name = getattr(data, 'name', f'graph_{i}')`
  - `pred = int(y_pred[i])`, `actual = int(y_actual[i])`
  - `num_nodes = data.num_nodes`, `num_edges = int(data.edge_index.shape[1])`
  - `correct = (pred == actual)`
  - ログ1行を書き込む。
  - `label_dir = run_dir/f'pred_{pred}'` を `makedirs(exist_ok=True)`。
  - ファイル名サニタイズ: `safe = re.sub(r'[^\w\-.#]', '_', name)`（`/` 等のパス破壊文字を除去。日本語/トルコ語文字 `\w` は保持）。
    - 同一 `run_dir` 内での万一の衝突に備え、`safe` に索引 `i` を付与（例 `f"{i:04d}_{safe}.png"`）してユニーク性を担保。
  - `save_graph_image(data, name, pred, actual, label_dir/ファイル名)` を呼ぶ。
- 長時間処理になり得るので、進捗表示（`\r{i+1}/{len}`）を出す。

## ステップ4: 引数追加 + `__main__` から呼び出し
- `argparse` に `--analysis_out` を追加。既定値は上記の絶対パス:
  `/home/A.hattori/ECMLPKDD25/results/20260715_時間幅0で埋め込み再計算した実験_テスト時のログを可視化`
- ループ前に基底ディレクトリを定義:
  `analysis_base_dir = os.path.join(args.analysis_out, f"{model_name}_{rww_attr}_nodeattr{node_attr}_{args.data_type}_mv{multivariate}")`
- `y_pred, y_actual, y_scores = predict(...)`（`:443`）の**直後**に:
  ```python
  run_dir = os.path.join(analysis_base_dir, f'exp{exp}')
  log_and_visualize(test_data, y_pred, y_actual, run_dir)
  ```
- 既存の `evaluate` / 結果JSON保存フローは一切変更しない。

---

## 検証方法
- 小さいデータセット（`--data_type small`）かつ1エポック相当で試走し、
  `results/test_analysis/<設定名>/exp0〜exp4/` に `predictions.log` と `pred_*/` 画像が生成されることを確認。
- `predictions.log` の行数がテストグラフ数 + 1（ヘッダ）と一致するか、
  各行の `predicted` と、対応する `pred_<label>/` フォルダへの画像振り分けが整合するかを確認。
- 生成画像を1枚開き、ネットワーク図・actual・nodes・edges がタイトルに出ているか目視確認。

## 未決事項 / 補足
- ラベルは**整数値のまま**表示・フォルダ名にする（例 `pred_0`, `actual=1`）。
  クラス名（`getReport` の `["News", "Politics", ...]` 等）への写像が必要なら追加可能だが、現要件には含めない。
- 全グラフ保存 × 5回 × 巨大グラフの場合、可視化に時間がかかる可能性あり（ステップ2のレイアウト分岐で緩和）。
