# 実装計画: `train_twitter_MPNN.py` の修正（問題4・9を除く）

対象ファイル: `train_twitter_MPNN.py`, `models.py`
方針: レビュー (`train_twitter_MPNN_review.md`) で挙げた問題のうち、**④検証セット追加** と **⑨`getReport` のラベル改変** を除く全てを修正する。

## 対象問題
①損失の多重活性化 / ②train/eval未設定 / ③no_grad未使用 / ⑤同一分割 / ⑥毎エポックシャッフル無 / ⑦バッチサイズ1 / ⑧退化した重み / ⑩int64キャスト

## 前提となる重要事実
`models.py` の `GCN` / `GCN_edge` / `GCN_News` の `self.out` Sequential は**末尾が `Sigmoid()`**。
そのため `model.out(pooled_output)` は既に [0,1] に潰れた値を返し、さらに `F.softmax` がかかる**多重活性化**になっている。問題1の修正は `models.py` 側にも及ぶ。

---

## ステップ1: 損失をロジット入力に修正（問題1）★最優先・依存多
**`models.py`:**
- `GCN`, `GCN_edge`, `GCN_News` の `self.out` Sequential から**末尾の `Sigmoid()` を削除**し、生ロジットを返すようにする。

**`train_twitter_MPNN.py` / `train_model`:**
- `pred = F.softmax(model.out(pooled_output), dim=1)` → `logits = model.out(pooled_output)`（softmax削除）。
- `BCEWithLogitsLoss`（2値）はロジット＋float型ターゲットを期待 → 既存の one-hot `[0,1]` float ターゲットはそのまま使える。
- `CrossEntropyLoss`（多クラス）はロジット＋クラスindex（long）を期待 → `graph.y`（クラスindex）をそのまま渡せる形に統一。

**`predict`:**
- `model.out` が生ロジットを返すようになるため、確率化の `F.softmax(pred, dim=1)` は**そのまま残す**（argmax・y_scores用）。整合を確認。

> 注: ④検証セットは追加しない方針なので、損失の正しさだけを担保する。

---

## ステップ2: train/eval モードと no_grad（問題2・3）
- `train_model` の各エポック先頭で `model.train()` を呼ぶ。
- `predict` の冒頭で `model.eval()`、推論本体を `with torch.no_grad():` で囲む。

---

## ステップ3: ミニバッチ化（問題7）＋毎エポックシャッフル（問題6）
インポート済みの `torch_geometric.loader.DataLoader` を使用。
- `train_model` を `DataLoader(train_data, batch_size=B, shuffle=True)`（`shuffle=True` が⑥も解決）でループする形に書き換え。`B` は引数化（既定32など）。
- `global_mean_pool(pred, batch=None)` → **`global_mean_pool(pred, batch.batch)`** に変更（バッチ内の複数グラフを正しくプーリング）。
- ターゲット生成をバッチ対応に修正:
  - 多クラス: `batch.y`（shape `[B]`）をそのまま `CrossEntropyLoss` へ。
  - 2値: 現在の「1グラフ用 one-hot」生成を、バッチ全体の one-hot（`[B,2]`）生成に置き換え。
- `predict` も `DataLoader(test_data, batch_size=B)` 化を検討（eval モード前提でバッチ統計に影響しない）。`y_scores`/`y_pred` 収集をバッチ対応に。

> 依存: ステップ1（softmax除去）と密接。プーリング→`out`→loss の流れをまとめて書き換える。

---

## ステップ4: 実験ごとに異なる分割（問題5）
- `load_split_data` 呼び出しを **5回ループの内側**へ移動し、`seed_everything(exp)` の**後**に実行（runごとに決定的かつ異なる分割）。
- これに伴い、`num_node_features`/`num_edge_features` に依存する **`conv_dictionary` 構築・`criterion` 構築・モデル構築を全てループ内へ移動**（グローバル特徴次元はデータ読み込みで確定するため）。
- `train_test_split` に `random_state=exp` を明示し再現性を担保。

> ファイル後半の最大の構造変更。`main` のループ本体を再編成する。

---

## ステップ5: クラス重みの修正（問題8）
- `weights = np.exp(-label_counts)` を**逆頻度ベース**に置換:
  例 `weights = label_counts.sum() / (len(label_counts) * label_counts)` または `weights = 1.0 / label_counts` を正規化。
- アンダーフロー（exp(-count)≈0）を解消。多クラス時のみ適用。

---

## ステップ6: GINE の int64 キャスト修正（問題10）
- `x_val = torch.tensor(graph.x).to(torch.int64)` を削除し、`GINEConv`（Linear入力）が期待する **float のまま** `model(graph.x, ...)` を渡す。
- `torch.tensor(既存テンソル)` の警告も解消（必要なら `.clone().detach()`）。
- GINE 経路は `train_model` / `predict` 両方にあるため双方修正。

---

## 変更ファイルと影響範囲
| ファイル | 変更箇所 |
|---|---|
| `models.py` | `GCN`/`GCN_edge`/`GCN_News` の `out` 末尾 Sigmoid 削除 |
| `train_twitter_MPNN.py` | `train_model`（大）, `predict`（中）, `main` のループ再編（大）, 重み計算, GINE分岐 |

## 依存関係・実施順
1 → 2 → 3 → 6 を先に（学習/推論ループの整合）、その後 4（構造移動）→ 5（重み）。
ステップ1とステップ3は同じ箇所を触るため一括実施が安全。

## 検証方法
- 各モデル（GCN/GAT/SAGE/GIN/GINE）で短いエポック数を指定し、例外なく1 run 完走することを確認。
- 損失が `nan`/負にならず単調傾向で下がること、`results/*.json` が従来通り出力されることを確認。
- 多クラス（`--multivariate 1`）と2値の両経路を最低1回ずつ実行。

## リスク / 留意点
- `out` の Sigmoid 削除はモデルの出力スケールを変えるため、**過去の学習済み結果・数値とは比較不能**になる（再学習前提）。
- バッチ化で `BatchNorm`（`models.py` 現状は未使用だが）導入時に挙動が変わる点に注意。
- ⑤でループ内に重い処理（毎回のデータ読み込み）が入り実行時間が約5倍になる。許容可否を要確認。

## 未確定事項（実装前に確認したい）
- ⑤でデータ読み込みを5回繰り返す（実行時間増）の許容可否。
- ミニバッチの既定バッチサイズ。
