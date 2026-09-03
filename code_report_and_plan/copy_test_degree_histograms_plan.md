# exp2 のテスト対象グラフの次数分布画像を集める計画

## 目的

各時間幅 t_w について、`exp2` の試行でテストに使われたグラフだけを取り出し、
その次数分布ヒストグラム画像 (`20260804_グラフノードの入出次数の統計` で生成済み) を
学習結果側のディレクトリに集約する。予測の当たり外れ (`predictions.log` の `correct` 列) と
グラフの次数分布を同じ場所で突き合わせられるようにするのが狙い。

## 入出力

`*min` は t_w に対応する (例: t_w=5 なら `5min`)。

| 役割 | パス |
|---|---|
| 入力 A: テスト対象の一覧 | `results/20260805_時間幅0-60で埋め込み再計算した実験_テスト時のログを可視化_自己ループのみのグラフを除外しない/<t_w>min/GCN_degree_nodeattr1_all_mv0/exp2/predictions.log` |
| 入力 B: 次数分布画像 | `results/20260804_グラフノードの入出次数の統計/<t_w>min/per_graph/<file_name>.png` |
| 出力 | `results/20260805_時間幅0-60で埋め込み再計算した実験_テスト時のログを可視化_自己ループのみのグラフを除外しない/<t_w>min/results/degree_distribution_exp2/` |

出力先 `degree_distribution_exp2/` は未作成なので新規に掘る (`<t_w>min/results/` には
現状 `GCN_degree_nodeattr1_all_mv0.json` のみが置かれている)。

## 処理手順 (t_w ごとに繰り返し)

1. **テストに用いたグラフファイル名を取得**
   `<t_w>min/.../exp2/predictions.log` をタブ区切りとして読み、ヘッダ行
   (`file_name predicted actual correct num_nodes num_edges`) を除いた各行の
   `file_name` 列を集める。`file_name` は拡張子なし (`..._fulldata`) で、1 t_w あたり 62 件。
2. **対応する画像をコピー**
   `20260804_.../<t_w>min/per_graph/<file_name>.png` を出力先へコピーする。
   ファイル名は変更しない。`shutil.copy2` を使い、タイムスタンプを保つ。

## 事前調査で分かっていること

実データで確認済み (2026-09-03)。実装前に把握しておく点を挙げる。

### 対象となる t_w は 53 個 (0-60 のうち 8 個が欠損)

`t_w = 16, 18, 22, 28, 30, 31, 32, 40` はディレクトリが空で `predictions.log` が無い。
原因は学習時の `torch.AcceleratorError: CUDA error: out of memory` (`logs/tw16.log`)。
この 8 個は処理対象外とし、エラーではなく「スキップした t_w」として報告する。

### ファイル名の照合は単純一致で足りる (ただし正規化フォールバックは入れる)

`predictions.log` の `file_name` と `per_graph` の PNG 名は、Unicode 正規化形式が一致しており、
素の文字列比較と NFC 正規化後の比較で一致件数が完全に同じだった (t_w = 0, 1, 5, 60 で確認)。
ただし本リポジトリは NFD/NFC 混在が既知の問題 (`CLAUDE.md` の「既知のバグ」、
`graph_labels.json` は 313 キー中 151 キーが NFD) なので、**素の一致 → 失敗したら NFC 正規化で再照合**
の 2 段構えにしておく。追加コストはほぼ無い。

### 画像が存在しないグラフがある — 原因は「その t_w でエッジ 0 本」

t_w ごとに 1〜2 件、テスト対象なのに PNG が無いグラフが出る。実例:

| t_w | 画像が無いグラフ | `predictions.log` の n_nodes / n_edges |
|---|---|---|
| 0 | `Kıyamet_Geliyor___2023-03-21_campaign_fulldata` | 1 / 0 |
| 0 | `PopiGram_Açıldı__2023-03-20_campaign_fulldata` | 1 / 0 |
| 60 | `PopiGram_Açıldı__2023-03-20_campaign_fulldata` | 1 / 0 |

`plot_degree_histograms.py` はエッジ 0 本のグラフに対して図を出力しないため。
異常ではなく仕様なので、**欠落を検出したら `predictions.log` の `num_edges` を確認し、
`0` なら正常な欠落として記録、`0` 以外なら警告として目立たせる**。
(`num_edges` は `predictions.log` に列として入っているので追加の I/O は不要。)

### 除外リストの NFC バグに起因するケースは exp2 では発生しない

`35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata` は学習側では除外が効かず
(NFD/NFC 不一致)、次数分布側では除外が効いている、という非対称がある。
すなわち「テストに現れるのに PNG が絶対に無い」候補になりうるが、実際に
全 53 t_w の `exp2` テスト分割を調べたところ **1 度も出現しなかった**。
上記の `num_edges` 判定に引っかかった場合の分岐として扱えば足りる。

## 実装方針

新規スクリプト `copy_test_degree_histograms.py` をリポジトリ直下に置く
(`plot_degree_histograms.py` などと同じ階層)。

```
python3 copy_test_degree_histograms.py \
    --pred_root  results/20260805_時間幅0-60で埋め込み再計算した実験_テスト時のログを可視化_自己ループのみのグラフを除外しない \
    --degree_root results/20260804_グラフノードの入出次数の統計 \
    --exp exp2 [--run_name GCN_degree_nodeattr1_all_mv0] [--dry_run]
```

- `--exp` を引数にして `exp0`〜`exp4` に流用できるようにする (出力先も `degree_distribution_<exp>`)。
- `--dry_run` でコピー件数と欠落だけ出す。まずこれで確認してから実コピーする。
- 既存ファイルは上書き。再実行しても結果が変わらないようにする。

## 規模の見積もり

1 t_w あたり 62 件 (うち 1〜2 件は欠落) × 53 t_w ≒ **3,200 ファイル**。
画像 1 枚あたり平均 37 KB なので合計 **約 120 MB**。`/home` の空きは 7.8 TB あり問題無い。
`results/` は `.gitignore` 済みなのでコミット対象にはならない。

## 検証

1. `--dry_run` で、全 t_w のコピー予定件数が 60〜62 に収まり、欠落がすべて
   `num_edges == 0` で説明できることを確認する。
2. 実行後、任意の t_w について
   `ls <出力先> | wc -l` と `predictions.log` の行数 - 1 - 欠落件数 が一致することを確認する。
3. 出力された PNG 名の集合が `predictions.log` の `file_name` の部分集合であることを確認する
   (余計なグラフが混ざっていないことの確認)。
