# 実験1（二値分類）再現に向けた実装計画

対象論文: *Density-aware Walks for Coordinated Campaign Detection (DECODE)*
対象実験: **実験1 = Campaign vs Non-campaign の二値分類**（Table 3 / Table 4 / Fig.2）
出典: `REPRODUCTION_ISSUES.md` の指摘事項に対する対応方針の確定版。

本ドキュメントは**着手前の計画**であり、コードの変更はまだ行っていない。

---

## 1. 対応する項目（11件）

| # | 重要度 | 対象ファイル | 具体的にやること |
|---|--------|-------------|-----------------|
| **B-1** | 🔴 | `train_twitter_MPNN.py:17` | `from SimpleConv import SimpleConv` を**削除**（未使用かつインポート失敗の原因） |
| **D-1** | 🔴 | `train_twitter_MPNN.py:257` | `pooled_output = global_mean_pool(...)` のインデントを修正し `IndentationError` を解消 |
| **E-1** | 🔴 | `train_twitter_MPNN.py` | `node_attr` を **int 化**し、`process_data` 内の `node_attr == 0/1` 比較が機能するようにする（現状は文字列vs整数で常に False → RWW埋め込みが使われない） |
| **E-2** | 🔴 | `train_twitter_MPNN.py`（学習側） | 分岐の指標名を**生成側名称に統一**: `'core'`→`'kcore'`、`'truss'`→`'ktruss'`。埋め込み属性キー（`structural_embedding`/`truss_embedding`/`degree_embedding`）は既に一致しているため変更しない |
| **E-3** | 🟡 | `train_twitter_MPNN.py:218-219, 259, 271` | `softmax` の出力にさらに `sigmoid` を掛ける**二重適用を解消**（train / predict 両方） |
| **E-4** | 🟡 | `train_twitter_MPNN.py:228` | E-3 の修正に合わせ、AUC用 `y_scores` を正しい確率から取得する |
| **E-5** | 🟡 | `kcore_rww.py:35` | `remove_zn` の `neighbors.pop(n)` → `neighbors.remove(n)`（ノードIDをインデックスとして誤削除する不具合） |
| **E-6** | 🟡 | `kcore_rww.py` | 密度の正規化と閾値比較を**論文 Algorithm 1 通り**にする。`φ(v) > τ` の比較が意味を持つよう密度スケールを調整し、高密度ノードでは高密度側、それ以外は低密度側を選好する挙動を成立させる |
| **G-1** | 🔴 | `run/degree/binary.sh`, `run/core/binary.sh`, `run/truss/binary.sh` | 未定義変数 `$rww_attr` → `$rww` に修正（**binary のみ**。multiclass.sh は対象外） |
| **G-2** | 🟡 | 同上 binary.sh 3本 | `#GINE` コメントを実体（`SAGE`）に合わせて訂正 |
| **G-3** | 🟡 | `kcore_rww.py`（`convert_file_id_to_name` / 呼び出し側） | 未登録IDを `try/except` で**スキップ**する（`seq 1 323` に対し mapping/label に無いIDで落ちないように） |
| **H-1** | 🟡 | `train_twitter_MPNN.py` | `args` の各値＋結果（5 seed 各回の指標＋平均±標準偏差）を1つの **JSON** にまとめ、`results/` 配下に保存する |

> 着手対象は **B-1, D-1, E-1〜E-6, G-1, G-2, G-3, H-1** の計11項目。

---

## 2. 変更しない項目（明示）

以下は今回**一切変更しない**:

- **A-1, A-2, A-3** … LENデータ／RWW埋め込み／mtxファイルの欠落（外部前提）
- **B-2** … `GCN_News` 等の未使用クラス
- **C-1〜C-5** … `requirements.txt`（gensim/matplotlib欠落・名称誤り・jax不要・wheel依存）
- **F-1, F-2, F-3** … 出力先ハードコード／閾値τの未配線／出力ディレクトリ作成
- **G-4** … 作業ディレクトリ前提
- **H-2, H-3, H-4, H-5** … epochsハードコード／exceptions／val_data未使用／乱数性

---

## 3. 実装上の注意・依存関係

- **E-2 と G-1 は連動。** G-1 修正で binary.sh は `$rww`（= `degree` / `kcore` / `ktruss`）を渡すようになる。E-2 で学習側がこの名称をそのまま受けられるようにすることで整合する。
- **E-3 と E-4 は連動。** 二重活性化の解消（E-3）に伴い、AUC用スコア（E-4）の取得元も同時に直す必要がある。
- **E-6 は「落ちるバグ」ではなくスケール設計の問題。** `get_core_numbers` / `get_degree` / `get_truss_numbers` の正規化方法と `get_rww` の閾値比較ロジックの両方に関わる。τ（0.5 / median / mid）と比較可能な密度スケールにすることが要点。

---

## 4. 確定済みの設計判断（ユーザー指示）

| 論点 | 決定 |
|------|------|
| E-2 の統一方向 | **学習側を生成側名称（`kcore`/`ktruss`）に合わせる** |
| E-6 の正とする挙動 | **論文 Algorithm 1 の通り**（`φ(v)>τ`で高密度側を選好） |
| G-3 の対処方法 | **`try/except` で未登録IDをスキップ** |
| H-1 の保存仕様 | **`args` の各値と結果を JSON にまとめ `results/` に保存** |
| G-1 のスコープ | **binary.sh のみ**（multiclass は対象外） |

---

> 本ドキュメントは計画のみ。コード・設定の変更は未実施。
