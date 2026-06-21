# `train_twitter_MPNN.py` レビュー — 機械学習上の問題点

対象ファイル: `train_twitter_MPNN.py`
種別: 機械学習プログラムとして一般的におかしい/まずい点の指摘（**修正は未実施**）

---

## 重大（学習が正しく行われていない可能性が高い）

### 1. softmax をかけた出力を、ロジットを期待する損失関数に渡している（二重活性化）
`train_model` 内:
```python
pred = F.softmax(model.out(pooled_output), dim=1)   # 確率に変換
loss = criterion(pred, label)
```
- 2値: `criterion = BCEWithLogitsLoss()` は**ロジット**を期待し内部で sigmoid をかける。softmax 済みの確率に更に sigmoid をかけることになり誤り。
- 多クラス: `criterion = CrossEntropyLoss()` も**ロジット**を期待し内部で log_softmax をかける。softmax 済みの値を渡すと softmax の二重適用になる。

**あるべき姿:** 損失には `model.out(pooled_output)` の生のロジットを渡し、softmax は推論時の確率化のみに使う。最大の問題。

### 2. `model.train()` / `model.eval()` を一度も呼んでいない
学習・推論ともにモード切替がない。Dropout / BatchNorm を含む場合、推論時 (`predict`) にも Dropout が効き、BatchNorm がバッチ統計を使うため評価が不安定・不正確になる。

### 3. `predict` に `torch.no_grad()` がない
推論で勾配を構築し続けるためメモリの無駄。大きなデータでは OOM の原因にもなる。

---

## 方法論的に問題

### 4. 検証 (validation) セットが存在しない
`val_data` は常に `None`。train/test の2分割のみで、ハイパーパラメータ選択や early stopping ができず、エポックは100で固定。実質テストセットが唯一の評価指標になっている。

### 5. 5回の「実験」がすべて同じ train/test 分割を使っている
`load_split_data` はループ前に1回だけ呼ばれ、`train_test_split(..., random_state` 未指定)。`seed_everything` はループ内で呼ばれるためモデル初期化だけが変わり、データ分割は固定。結果の分散 (±) が初期化由来だけになり、汎化性能のばらつきを過小評価する。分割自体も実行ごとに非決定的（シード設定前に分割している）。

### 6. エポックごとのシャッフルがない
`random.shuffle(train_data)` は最初の1回だけ。各エポックで毎回同じ順序でデータを舐めるため SGD の利点が薄れる。

### 7. 実質バッチサイズ1
`for i in range(len(train_data))` で1グラフずつ `opt.step()`。`DataLoader` はインポートされているが未使用。勾配が非常にノイジーで非効率。ミニバッチ化（`batch` 引数を使った `global_mean_pool`）が望ましい。

---

## その他のバグ

### 8. クラス重みが `np.exp(-label_counts)` で退化する
```python
weights = np.exp(-label_counts)
```
カウントが少し大きい（例: 20以上）だけで `exp(-count)` がほぼ 0 にアンダーフローし、最小クラス以外の重みが事実上消える。通常は逆頻度（`1/count` など）を使うべきで、この式は意図した重み付けにならない。

### 9. `getReport` が正解ラベルを書き換えている
```python
y_actual[-1] = 2
y_actual[-5] = 2
```
評価のグラウンドトゥルースを人為的に改変している（混同行列にクラス2を出すための小細工に見える）。本関数は `main` から呼ばれていないが、明確に不正。

### 10. GINE 経路で特徴量を int64 にキャスト
```python
x_val = torch.tensor(graph.x).to(torch.int64)
```
浮動小数点のノード特徴量を整数に切り捨てており情報が失われる。加えて既存テンソルに `torch.tensor(...)` を使うと警告/コピーが発生（`.clone().detach()` 推奨）。

---

## 優先度の提案
特に **1（損失への二重活性化）** と **2（train/eval モード未設定）** は、報告されている指標の妥当性自体に関わるため最優先で対応すべき。
