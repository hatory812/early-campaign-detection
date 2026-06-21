# 実験1（二値分類）再現に向けた問題点一覧

対象論文: *Density-aware Walks for Coordinated Campaign Detection (DECODE)*
対象実験: **実験1 = Campaign vs Non-campaign の二値分類**（論文 Table 3 / Table 4 / Fig.2）
本ドキュメントは「再現を妨げる/結果を歪める要因」を洗い出したもの。**修正は未実施**。

凡例: 🔴 致命的（このままでは実行不可 or 結果が論文と別物） / 🟡 要注意（再現性・正確性に影響） / ⚪ 軽微・整理事項

---

## A. 欠落しているデータ・外部リソース

| # | 重要度 | 内容 |
|---|--------|------|
| A-1 | 🔴 | **LEN データ本体が無い。** `all_graphs/`, `small_graphs/`, 各 `graph_labels*.json` がリポジトリに存在しない（確認済み: 3ディレクトリとも未存在）。README記載の配布元 https://erdemub.github.io/large-engagement-network/ から取得が必要。 |
| A-2 | 🔴 | **RWW埋め込み済みグラフが無い。** `train_twitter_MPNN.py` はノード属性 `degree_embedding` 等を読む前提だが、生のLENグラフにはこの属性が無い。先に `kcore_rww.py` を全グラフに対し実行して埋め込み付きグラフを生成する必要がある（→ F章の経路問題に注意）。 |
| A-3 | 🟡 | **k-truss 用の事前計算ファイルが無い。** `k_truss.py:15` が参照する `./mtx_files/hierarchy/Hierarchy_raw/23/K_values/*.mtx_23_K_values` が存在しない。Sariyuce et al. の nucleus コード（README記載）で別途生成が必要。※実験1の最良値はdegreeベースなので truss は必須ではないが、Table 3/4 の Truss 列を埋めるには必要。 |

---

## B. 欠落しているファイル・モジュール

| # | 重要度 | 内容 |
|---|--------|------|
| B-1 | 🔴 | **`SimpleConv.py` が存在しない。** `train_twitter_MPNN.py:17` の `from SimpleConv import SimpleConv` でインポートエラー（`ModuleNotFoundError`）になり、スクリプト全体が起動しない。しかも `SimpleConv` はファイル内で一切使用されていない（残骸インポート）。 |
| B-2 | 🟡 | **`models.py` の `GCN_News` 等が未使用。** import はされるが実験1では使われない。実害は小さいが整理対象。 |

---

## C. 依存関係（requirements.txt）の問題

| # | 重要度 | 内容 |
|---|--------|------|
| C-1 | 🔴 | **`gensim` が記載されていない。** `kcore_rww.py:11` の `from gensim.models.word2vec import Word2Vec` に必須。RWW埋め込み生成が不可能。 |
| C-2 | 🟡 | **`matplotlib` が記載されていない。** `train_twitter_MPNN.py:41` で import。実験1の本筋では描画関数は呼ばれないが import 時に必要。 |
| C-3 | 🔴 | **パッケージ名が誤り: `pytorch-geometric==2.2.0`。** PyPIの正しい配布名は `torch-geometric`。`pip install -r requirements.txt` が失敗/誤パッケージ取得になる。 |
| C-4 | 🟡 | **`jax==0.4.16` は不要。** コード中で未使用。インストール負荷・依存衝突の元。 |
| C-5 | 🟡 | **torch 1.13.0 / PyG 2.2.0 の補助wheel依存。** `torch-scatter`/`torch-sparse` 等のCUDA整合wheelが別途必要になる場合がある（環境依存）。現環境には torch も PyG も未インストール。 |

---

## D. 実行を妨げる致命的コードバグ

| # | 重要度 | 箇所 | 内容 |
|---|--------|------|------|
| D-1 | 🔴 | `train_twitter_MPNN.py:257` | **インデント不整合による構文エラー（IndentationError）。** `pooled_output = global_mean_pool(...)` の行頭スペースが前後と揃っておらず、Pythonがパースに失敗する。ファイル自体が実行不能。 |

---

## E. 実行はできても結果を歪める論理バグ

| # | 重要度 | 箇所 | 内容 |
|---|--------|------|------|
| E-1 | 🔴 | `train_twitter_MPNN.py:367` + `process_data()` 110-123 | **`node_attr` の型不一致で RWW埋め込みが永遠に使われない。** `args.node_attr` は文字列（"0"/"1"）のまま渡されるが、`process_data` 内は整数比較 `node_attr == 1` / `== 0`。文字列 vs 整数は常に `False` のため、全 `elif` を素通りして `else`（生のnode_attrのみ）に落ちる。**＝どの密度埋め込みも特徴量に入らず、ベースラインと同じ入力で学習してしまう。** 実験1の核心（RWWの効果）が再現できない。 |
| E-2 | 🔴 | `kcore_rww.py` の `pick` 値 vs `train_twitter_MPNN.py` の `rww_attr` 値 | **密度指標名の不一致。** 埋め込み生成側は `kcore`/`ktruss`/`degree`、学習側の分岐は `core`/`truss`/`degree` を期待（`process_data` 110-121）。`kcore`→`core`, `ktruss`→`truss` が一致しないため、core/truss 指定時は分岐に合致せず `else`（生特徴）に落ちる。degree のみ辛うじて一致。Table 3/4 の Core/Truss 列が再現不可。 |
| E-3 | 🟡 | `train_twitter_MPNN.py:218-219`, `259-271` | **softmax の出力にさらに sigmoid を二重適用。** `F.softmax(...)` 済みの値に `torch.sigmoid()` を掛けており確率として無意味。さらに `BCEWithLogitsLoss`（内部でもう一度sigmoid相当を適用）に渡しているため、損失計算が論文の想定と異なる。学習自体は進むが、報告値の正確な再現を阻害。 |
| E-4 | 🟡 | `train_twitter_MPNN.py:228`, `predict()` | **AUC用スコアが歪む。** `y_scores` が `sigmoid(softmax(logits))` から取られるため、Fig.2 の ROC/AUC を忠実に再現できない恐れ。 |
| E-5 | 🟡 | `kcore_rww.py:35` `remove_zn()` | **`neighbors.pop(n)` の誤用。** `neighbors` はリストで `n` はノードID。`list.pop` はインデックス削除なので、ノードIDをインデックスとして誤削除/IndexErrorの可能性。正しくは `remove(n)`。ウォーク生成（＝埋め込み）の質に影響。 |
| E-6 | 🟡 | `kcore_rww.py:74` `get_rww()` + 各 `get_*` の正規化 | **正規化済み密度値と固定閾値 τ=0.5 の不整合。** `get_core_numbers`/`get_degree`/`get_truss_numbers` は密度を「総和=1」に正規化するため各値は概ね 1/N ≪ 0.5。固定 τ=0.5 だと `φ(v) > τ` がほぼ常に偽となり、常に低密度側分岐（`1-φ`）を選ぶ。論文Algorithm 1の意図（高密度ノードでは高密度側を選好）と乖離する可能性。τ=0.5（degree最良条件）の再現で要検証。 |

---

## F. 埋め込み生成 → 学習の配線（パイプライン）不整合

| # | 重要度 | 内容 |
|---|--------|------|
| F-1 | 🔴 | **出力先がハードコードかつ学習側入力と不一致。** `kcore_rww.py:207` は `/vscratch/grp-erdem/atulanan/graphs/{pick}/{comp_parameter}` に書き出すが、学習側は `--all_graphs_path`（例 `./all_graphs`）配下の `*_fulldata.json` を読む。生成物が学習側ディレクトリに入らないため、手動でパスを合わせない限り埋め込みが読み込まれない。 |
| F-2 | 🟡 | **閾値 τ（0.5/mid/median）が学習側に伝わらない。** 学習スクリプトに `--comp`/閾値引数が無く、`all_graphs_path` 配下を無差別に読むだけ。論文は τ をモデル別に選ぶが、その選択はディレクトリを手で差し替える運用前提になっている（Table 3/4 の τ列を体系的に再現しづらい）。 |
| F-3 | 🟡 | **出力ディレクトリの事前作成が無い。** `kcore_rww.py:210` は `open(...,'w')` 前に出力ディレクトリ作成（`os.makedirs`）をしないため、ディレクトリ未作成だと `FileNotFoundError`。 |

---

## G. 実行スクリプト（run/*）のバグ

| # | 重要度 | 箇所 | 内容 |
|---|--------|------|------|
| G-1 | 🔴 | `run/degree/binary.sh`, `run/core/binary.sh`, `run/truss/binary.sh` | **未定義変数 `$rww_attr` を渡している。** 定義しているのは `rww="degree"`（等）なのに、`--rww_attr $rww_attr` と未定義変数を参照。空文字列が渡り、学習側は `else`（生特徴）に落ちる。正しくは `$rww`。実験1の全 binary.sh が該当。 |
| G-2 | 🟡 | 各 `binary.sh` の `#GINE` コメント | **コメントとモデルの不一致。** `#GINE` の直下で実体は `m4="SAGE"`。実験1の最良モデルは GraphSAGE なので、SAGE自体は走るが可読性・取り違えの温床。 |
| G-3 | 🟡 | `run/*/gen_rww.sh` の `seq 1 323` | **存在しないIDでの停止リスク。** mapping は323件だがLENの有効ネットワークは314件。`convert_file_id_to_name`（`kcore_rww.py:14`）は未登録IDで `KeyError`、ラベル無しは `run_model` 側でスキップ。例外処理が無くループ全体が落ちる可能性。 |
| G-4 | ⚪ | `run/*/binary.sh` 全般 | **作業ディレクトリ前提が暗黙。** `python3 ../train_twitter_MPNN.py` かつ `./small_graphs` 等の相対パス。`run/<metric>/` 内からの実行を前提にしており、READMEに明記が無い。 |

---

## H. 再現性・運用上のギャップ（バグではないが要対応）

| # | 重要度 | 内容 |
|---|--------|------|
| H-1 | 🟡 | **結果・モデルの保存が無い。** `train_twitter_MPNN.py` は標準出力に print するのみ。5seed×複数設定の集計を残す仕組みが無い。 |
| H-2 | 🟡 | **エポック数がハードコード（`epochs = 100`、L400）。** 引数化されておらず論文の探索条件（hidden∈{128,256,512,1024}, lr∈{1e-3,1e-4,1e-5}）はスクリプトに固定値が散在。ハイパラ探索の再現は手作業。 |
| H-3 | ⚪ | **`exceptions` リスト（L173-175）がハードコード。** 特定グラフ名を除外。LEN配布版とファイル名がズレると挙動が変わる。 |
| H-4 | ⚪ | **`val_data` は常に None。** バリデーション処理はコメントアウト済みで未使用。実験1の結果には影響しないが未完成コード。 |
| H-5 | ⚪ | **乱数性。** `seed_everything(exp)` で seed 0-4 を5回実行。`train_test_split` は seed の影響下にあるが、データ取得元のバージョン差で分割が変わり得る。 |

---

## 再現に必要な作業の最小手順（参考）

上記を踏まえると、実験1（特に最良条件 GraphSAGE + degree, τ=0.5）を再現するには概ね以下が必要:

1. **データ取得**（A-1）: LEN から `all_graphs/`・`graph_labels.json` を配置。
2. **依存修正**（C-1〜C-4）: `gensim`・`matplotlib` 追加、`pytorch-geometric`→`torch-geometric` 修正、`jax` 除去。
3. **起動阻害の除去**（B-1, D-1）: `SimpleConv` import 削除、L257 のインデント修正。
4. **RWWが効くようにする**（E-1, E-2, G-1）: `node_attr` の int 変換、`core/kcore`・`truss/ktruss` の名称統一、binary.sh の `$rww_attr`→`$rww`。
5. **配線**（F-1〜F-3）: `kcore_rww.py` の出力先を学習側 `all_graphs_path` に合わせる（or 引数化）、出力ディレクトリ作成。
6. **埋め込み生成 → 学習**: `gen_rww.sh`（degree）→ `binary.sh`（degree）。
7. （任意）**正確な数値一致**を狙うなら E-3〜E-6 の二重活性化・スコア・正規化/閾値も要検討。

> 注: 本ドキュメントは指摘のみ。コード・設定の変更は行っていない。
