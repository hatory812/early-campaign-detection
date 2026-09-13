# 実装計画: スナップショット縮小化による精度変化の実験

## 1. 実験の理解（要約）

各グラフ `G_i` を「拡散開始から `t_w` 分以内」に縮小し、その縮小データセット `G'` でモデルを訓練して精度を測る。`t_w` を 1→60 分まで1分刻みで変え、**Accuracy / Precision / Recall / F1 が t_w に対してどう変化するか**の曲線を得る。

縮小規則:

- `t_0 = min(エッジの timestamp)`（**グラフごと**に算出、Unix秒）
- しきい値 `θ = t_0 + t_w*60`（秒）
- `E_ie = { timestamp ≤ θ のエッジ }`（「以内」= `≤`）
- `V_ie = E_ie の両端ノード`（孤立ノードは落ちる）
- `G_i' = (V_ie, E_ie)`

**データ確認で判明した重要事実**: グラフの時間幅は最大約1440分。60分以内に入るエッジは多くのグラフで全体の数%未満で、グラフによっては60分でもほぼ0%。→ 小さい `t_w` では縮小グラフが極小になるものがある（拡散初期を見る本実験の意図どおり）。`timestamp == t_0` のエッジは必ず残るので、各 `G_i'` は最低1エッジを持つ。

## 2. 設計方針

既存の `train_twitter_MPNN.py` を土台に、新規スクリプト **`train_twitter_snapshot.py`** を作成（既存スクリプトは非破壊で温存）。

**最大の性能ポイント**: JSONの読込（312ファイル、最大14万エッジ）は重い。`t_w` ごとに312ファイルを再読込すると 60×312 回になるため、**JSONパースは一度だけ**行い、軽量キャッシュをメモリに保持。各 `t_w` ではキャッシュをnumpyマスクでフィルタするだけにする（埋め込み再計算なし）。

### 決定事項

- **特徴量の扱い**: ノード特徴量（degree_embedding 等）とエッジ特徴量（edge_attr）は、元々フルグラフ全体に対する RWW で事前計算済み。これを**流用**し、生き残ったノード/エッジだけ抜き出す（再計算しない）。トポロジ縮小の純粋効果を見る設定。
- **モデル範囲**: まず **GCN 1モデル**で t_w=1..60 を掃引し、傾向を確認してから他モデルへ拡張。

### 2.1 一度だけ作る生キャッシュ `build_cache()`

`load_data` を改修。312ファイルを1回ループし、グラフごとに以下を `float32` numpy 配列で保持（dict回避でメモリ圧縮）:

- `src[], dst[]`（元ノードID）, `ts[]`（timestamp）, `edge_attr[E, F_e]`
- `node_feat[N, F_n]`（`rww_attr`/`node_attr` の組合せで選んだ特徴。既存 `process_data` の分岐をそのまま流用）
- `label`, `t0 = ts.min()`, `name`

### 2.2 スナップショット構築 `build_snapshot(cache_i, t_w) -> Data`

- `mask = ts <= t0 + t_w*60`
- 残エッジの端点 `V = unique(src[mask] ∪ dst[mask])` → `0..k-1` に再ラベル（dictマップ）
- `x = node_feat[V]`, `edge_index = remap(src/dst[mask])`, `edge_attr = edge_attr[mask]`, `y = label`
- PyG `Data` を返す（既存と同じ構造なので `train_model`/`predict`/`evaluate` をそのまま再利用可能）

### 2.3 メインループ（t_w 掃引）

```python
cache = build_cache(...)          # 1回だけ
results_by_tw = []
for t_w in range(1, 61):
    data_list = [build_snapshot(c, t_w) for c in cache]   # 順序固定
    label_list = [c.label for c in cache]
    # 既存の 5 experiments ループ（split_data → train_model → predict → evaluate）
    # acc/prec/rec/f1 の mean±std を集計
    results_by_tw.append({t_w, mean, std})
```

- **公平性**: `cache` を固定順で作り `data_list` の順序を `t_w` 間で不変にする。`split_data(seed=exp)` は stratify + 固定 random_state なので、**同じグラフが t_w 間で一貫して train/test に割り当てられる**（縮小度合いだけが変数になる）。
- `num_node_features` は埋め込み次元で `t_w` 不変 → モデルは現状どおり run ごとに再構築でOK。

### 2.4 結果出力

- `results/snapshot_{model}_{rww}_nodeattr{na}.json`（t_wごとの per-run と mean/std）
- 併せて `…_tw.csv`（列: `t_w, acc_mean, acc_std, prec_…, rec_…, f1_…`）
- 任意で `…_tw.png`（4指標 vs t_w の折れ線グラフ）

### 2.5 引数追加

既存引数に加え:

- `--tw_min`（既定1）, `--tw_max`（既定60）, `--tw_step`（既定1） … テスト時に範囲を絞れるように
- 実行例:

  ```bash
  python3 train_twitter_snapshot.py --model GCN --data_type all \
    --hidden_dim 1024 --lr 0.0001 --output_dim 2 \
    --all_graphs_path /hss01/A.hattori/rww_all_graphs/degree/mid \
    --multivariate 0 --rww_attr degree --node_attr 1
  ```

## 3. 留意点・リスク

- **実行時間**: データ準備は一度（数分）だが、訓練は 60×(5 runs×100ep) ＝ 現状の約60倍。GCN単体でも長時間。まず `--tw_max 3` 等で動作確認 → 全掃引、を推奨。
- **極小グラフ**: 小さい `t_w` で1〜数ノードの `G_i'` が出る。バッチ化・pooling は問題ないが、精度曲線の低 `t_w` 側はノイズが大きくなる想定（これ自体が知見）。
- **特徴量の未来情報リーク**: 埋め込みは全体グラフ由来。これは「トポロジ縮小の純粋効果」を見る設定であり、`t_w` 以降の情報が特徴に残る点は結果解釈時の前提として明記する。

## 4. 成果物

1. `train_twitter_snapshot.py`（新規）
2. `run/degree/snapshot.sh`（実行スクリプト雛形）
3. 結果: `results/snapshot_GCN_degree_*.{json,csv,png}`
