# ルートディレクトリ Python ファイル概要

現在の階層 (`ECMLPKDD25/`) に含まれる各 Python ファイルの役割をまとめる。

## `k_truss.py`
Twitter の拡散グラフ (DiGraph) に対して **k-truss 分解**を計算するモジュール。

- `get_k_truss()`: 全エッジについて所属する三角形 (3-クリーク) 数を数え、削減処理で各エッジの truss 値を求める。
- `load_k_truss()`: 事前計算済みの `.mtx_23_K_values` ファイルから truss 値を読み込み、グラフのエッジ順に対応付ける (`kcore_rww.py` から呼ばれる)。
- `coarsen_graph()`: 未実装 (グラフ粗視化のプレースホルダー)。

## `kcore_rww.py`
グラフの **構造的埋め込み (structural embedding)** を作る中核モジュール。

- ノードに `degree` / `kcore` / `ktruss` の正規化された重要度スコアを付与 (`get_degree`, `get_core_numbers`, `get_truss_numbers`)。
- そのスコアを重みとした **バイアス付きランダムウォーク (RWW)** を各ノードから実行 (`get_rww`)。
- ウォーク列を Word2Vec に通し、ノードごとの埋め込みベクトルを得る (`get_embedding`)。
- CLI 実行時は、graphId 指定のグラフ 1 つを読み込み・埋め込み計算・保存するバッチ処理 (`run_model`)。

## `models.py`
PyTorch Geometric ベースの **GNN 分類モデル定義**。

- `GCN`: 2 層グラフ畳み込み (Conv1/Conv2 は外部から注入、GCN/GAT/SAGE/GIN を差し替え可能) + 出力用 MLP。ktruss/kcore 重み付けネットワーク (`w_ktruss_network` など) も保持。
- `GCN_edge`: エッジ属性を使う GINE 用モデル。
- `GCN_News`: ニュース分類向けに Dropout を挟んだシンプルなバリアント。

## `precompute_minimized_embeddings.py`
**時間窓 t_w で切り詰めた部分グラフ**ごとに埋め込みを事前計算するバッチスクリプト。

- 元の全体グラフ JSON を読み込み、`t0〜t0+t_w分` のエッジだけ残す → 孤立ノード除去 → 再ラベル。
- 部分グラフ上で degree を再計算し、RWW + Word2Vec で埋め込みを再計算 (未来情報リークを防ぐため)。
- t_w の値域をメモリ使用量に応じて 3 ティア (small/medium/large) に分け、`multiprocessing.Pool` (spawn 方式) で並列実行し、結果を `rww_all_graphs_minimized/{t_w}min/` に保存。

## `report_snapshot_stats.py`
`train_twitter_snapshot.py` と同じ縮小規則を使い、**学習は行わず**各 t_w での縮小グラフのノード数・エッジ数の統計 (平均・標準偏差・分位点など) を集計して CSV/JSON/Markdown/PNG でレポート出力するスクリプト。

## `train_twitter_MPNN.py`
**フルグラフ**を対象にした本体の学習・評価スクリプト。

- 全 JSON を読み込み、kcore/degree/ktruss いずれかの埋め込み＋ノード属性を特徴量として PyG `Data` を構築。
- GCN/GAT/SAGE/GIN/GINE モデルを選択・訓練し (`train_model`)、5 回試行して精度・適合率・再現率・F1 を集計。
- 結果を `results/` に JSON として保存。多変量分類 (`multivariate`) やニュース分類 (`classify_news`) にも対応。

## `train_twitter_snapshot.py`
`train_twitter_MPNN.py` を土台にした **t_w 掃引実験** スクリプト。

- JSON パースは一度だけ行い軽量 numpy キャッシュ (`build_cache`) を作成、各 t_w ではタイムスタンプでマスクするだけで済むよう最適化。
- 各 t_w ごとに縮小グラフ (`build_snapshot`) を作り、5 回訓練・評価して Accuracy/Precision/Recall/F1 の平均・標準偏差を算出。
- t_w に対する精度変化の曲線を CSV/JSON/PNG として逐次保存 (長時間実行を想定)。

## `train_minimized_snapshot_mpnn.py`
`precompute_minimized_embeddings.py` が生成した縮小・埋め込み再計算済みデータセット (`rww_all_graphs_minimized/{t_w}min/`, t_w=1..60) を、**`train_twitter_MPNN.py` に t_w ごとに投入して回すドライバ (オーケストレーター)** スクリプト。自身は学習をせず、`train_twitter_MPNN.py` を subprocess として繰り返し起動する。

- `train_twitter_MPNN.py` を一切改変せず、t_w ごとに作業ディレクトリ (cwd) を分離して起動 → 出力名に t_w を含まない MPNN 側の結果 JSON の上書き事故を防ぐ。
- 冪等性: 結果 JSON が既にある t_w はスキップし、中断後の再開に強い。
- 並列化は `multiprocessing.Pool` (spawn)。ただしワーカーは torch/CUDA に触れず subprocess の起動・待機・ログ記録のみを行う。
- メモリ安全性重視の設計 (過去に `train_twitter_snapshot.py` でホスト全体の OOM killer が発火した実績あり): 既定は `num_workers=1` (逐次)、空きメモリ監視・警告、`--force_cpu` / `--test_tw` / `--dry_run` を用意。

## 全体の関係

`k_truss.py` と `kcore_rww.py` で構造特徴と埋め込みを作り (`models.py` の GNN への入力)、`precompute_minimized_embeddings.py` で時間窓ごとの埋め込みを事前計算し、`train_twitter_MPNN.py` (フルグラフ学習) と `train_twitter_snapshot.py` (t_w 掃引実験) で分類モデルを学習・評価、`report_snapshot_stats.py` でその縮小の程度を統計的に確認する、という一連のパイプラインになっている。縮小データセットを使った t_w 掃引は、オンザフライ縮小の `train_twitter_snapshot.py` に対し、事前計算データを `train_twitter_MPNN.py` に流し込む `train_minimized_snapshot_mpnn.py` という二通りの経路がある。
