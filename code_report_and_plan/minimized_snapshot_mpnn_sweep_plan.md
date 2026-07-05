# 実装計画: 縮小化データセット (t_w=1〜60分) を用いた `train_twitter_MPNN.py` の並列一括実行

## 目的

`/hss01/A.hattori/rww_all_graphs_minimized/{t_w}min/`（`precompute_minimized_embeddings.py` が生成した、各時間幅 t_w でエッジ・ノードを縮小しノード特徴量を再計算済みのデータセット、t_w=1〜60）を、
`train_twitter_MPNN.py` に t_w ごとに読み込ませて学習・評価する実験を、**`train_twitter_MPNN.py` 自体は改変せず**、Python の `multiprocessing` で OOM を起こさない範囲のプロセス数で並列実行する。

## 前提として確認済みの事実

### データセットの制約
- `precompute_minimized_embeddings.py` は `--pick degree`（実質固定、docstringにも「degree のみ使用」と明記）でしか埋め込みを作っていない。
  → `rww_all_graphs_minimized/` 配下には `degree_value` / `degree_embedding` しかなく、`kcore_value`/`ktruss_value`/`structural_embedding`/`truss_embedding` は存在しない。
  → **この実験では `--rww_attr degree` 固定**（`kcore`/`ktruss` を指定すると `train_twitter_MPNN.py` の `process_data` 内で `KeyError` になる）。
- ファイル名は元の `*_fulldata.json` 形式を維持しているため、`train_twitter_MPNN.py` の `load_data`（`glob(path + '/*_fulldata.json')`）にそのまま適合する。

### `train_twitter_MPNN.py` の出力パスの制約（★最重要）
- 結果ファイルは **相対パス** `results/{model}_{rww_attr}_nodeattr{node_attr}_{data_type}_mv{multivariate}.json` に保存される（`train_twitter_MPNN.py` 末尾）。
- ファイル名に **t_w が含まれない**ため、同じ作業ディレクトリで60回実行すると毎回上書きされて消えてしまう。
- **対策**: `train_twitter_MPNN.py` は改変せず、t_w ごとに**別々の作業ディレクトリ (cwd)** を指定して `subprocess` として起動する。相対パス `results/...` はその cwd 基準で解決されるため、自動的に t_w ごとの出力先に分離できる。

### メモリ・GPUに関する既知のリスク（★安全なプロセス数決定に直結）
- `code_report_and_plan/snapshot_oom_investigation.md` に記録の通り、過去に `train_twitter_snapshot.py` がホストの物理メモリを食い尽くし、**OOM killer が同一セッションスコープの tmux ごと巻き添えで kill**した実績がある。
- 現在のホストメモリ状況（計画作成時点で確認）:
  - `free -h`: total 119Gi, **available は約6.8Gi** と非常に逼迫（他要因で埋まっている可能性）。Swap 15Gi中8.7Gi使用済み。
  - `nproc`: 20（論理コア数上は多重実行が可能に見えるが、メモリがボトルネック）。
- GPU (`nvidia-smi` より `NVIDIA GB10`) は Grace Blackwell系の**統合メモリアーキテクチャ**で、CPU/GPUがメモリプールを共有していると見られる（`memory.total`/`memory.used` が `N/A` として返る点からも通常のディスクリートGPUと異なる可能性が高い）。
  → `torch.cuda.is_available()` が True の環境で複数プロセスが同時に CUDA コンテキストを張ると、**GPUメモリ消費もホストの同一メモリプールを圧迫**しうる。ディスクリートGPUを前提にした「GPUメモリとCPUメモリは別物」という想定はできない。
- `rww_all_graphs_minimized/` のディレクトリサイズは t_w に対して単調増加（実測: `1min` = 77MB, `60min` = 3.5GB）。t_wが大きいほど1ワーカーあたりのメモリ使用量も増える。

これらから、**固定の少数プロセス数 + t_wが大きい領域はさらに絞る** という、`precompute_minimized_embeddings.py` のティア分割と同様の考え方が必要と判断。

## 設計方針

1. **`train_twitter_MPNN.py` は無改変**。オーケストレータ側から `subprocess.run` で t_w ごとに独立プロセスとして起動する。
2. **t_w ごとに作業ディレクトリを分離**し、出力の上書き事故を防ぐ（前述）。
3. **`multiprocessing.Pool` で並列化**。ワーカー自身は torch/CUDA に触れず「別プロセスの起動と待機・ログ記録」だけを行うため、fork/spawn どちらでも安全だが、`precompute_minimized_embeddings.py` の前例に倣い **spawn を明示**して事故を防ぐ。
4. **冪等性**: 既に該当 t_w の結果 JSON が存在する場合はスキップ（`precompute_minimized_embeddings.py` と同じ設計）。中断・再実行に強くする。
5. **プロセス数はティア制 or 保守的な固定値**とし、既定値は小さめ（例: 2〜4）に設定。現在の空きメモリが数GB規模であることを踏まえ、**デフォルトで積極的な並列化はしない**方針とし、`--num_workers` で明示的に上書きできるようにする。
6. **CPU専用実行オプション（`--force_cpu`）を用意**。GB10 の統合メモリ特性を踏まえ、GPU競合を避けたい場合は `CUDA_VISIBLE_DEVICES=''` を子プロセスの環境変数に設定して CPU 実行に固定できるようにする。
7. 各ワーカーの標準出力/標準エラーは t_w ごとのログファイルへリダイレクトし、進捗・成否をメインログに集約する（`precompute_minimized_embeddings.py` の `setup_logger` パターンを踏襲）。
8. 実行前に `/proc/meminfo` の `MemAvailable` を読み取り、現在の空きメモリを起動時ログに記録し、必要に応じて警告を出す（強制停止はしない: 情報提供に留める）。

## スクリプト仕様（作成予定: `train_minimized_snapshot_mpnn.py`）

### 配置
プロジェクトルート直下（`precompute_minimized_embeddings.py` と同階層）。

### 主なCLI引数
| 引数 | 既定値 | 説明 |
|---|---|---|
| `--src_root` | `/hss01/A.hattori/rww_all_graphs_minimized` | t_w ディレクトリの親 |
| `--out_root` | `./results/minimized_snapshot_mpnn` | t_w ごとの作業ディレクトリ（=各回の cwd）の親。実結果は `{out_root}/{t_w}min/results/...json` に生成される |
| `--tw_min` / `--tw_max` / `--tw_step` | 1 / 60 / 1 | 掃引範囲 |
| `--num_workers` | 2〜4程度（要調整） | `multiprocessing.Pool` のプロセス数 |
| `--model` | `GCN` | `train_twitter_MPNN.py --model` にそのまま渡す |
| `--lr`, `--hidden_dim`, `--output_dim`, `--rww_attr`(既定`degree`固定推奨), `--node_attr`, `--batch_size`, `--multivariate`, `--classify_news` | README記載の既定値に準拠 | そのまま `train_twitter_MPNN.py` へ委譲 |
| `--force_cpu` | False | 子プロセスの `CUDA_VISIBLE_DEVICES` を空にしてCPU実行に固定 |
| `--test_tw` | None | カンマ区切りでt_wを指定し `Pool(1)` で逐次実行するテストモード（`precompute_minimized_embeddings.py` と同様） |
| `--dry_run` | False | 実際には実行せず、組み立てたコマンドとcwdだけを表示して確認する |
| `--log_dir` | `{out_root}/logs` | t_wごとのログ + メインログの出力先 |

### 処理フロー
1. `mp.set_start_method('spawn')`
2. `--tw_min`〜`--tw_max` の t_w 一覧を作成（`--test_tw` 指定時はそちらを優先）。
3. 各 t_w について:
   - `src_dir = {src_root}/{t_w}min` の存在確認。無ければ警告してスキップ。
   - `run_dir = {out_root}/{t_w}min` を作成（`train_twitter_MPNN.py` 実行時の cwd）。
   - 期待される出力 `run_dir/results/{model}_{rww_attr}_nodeattr{node_attr}_all_mv{multivariate}.json` が既に存在すればスキップ（冪等性）。
   - コマンド組み立て:
     ```
     python3 {絶対パス}/train_twitter_MPNN.py \
       --model {model} --lr {lr} --hidden_dim {hidden_dim} --output_dim {output_dim} \
       --data_type all --all_graphs_path {src_dir} \
       --multivariate {multivariate} --classify_news {classify_news} \
       --rww_attr {rww_attr} --node_attr {node_attr} --batch_size {batch_size}
     ```
   - `subprocess.run(cmd, cwd=run_dir, env=env, stdout=logfh, stderr=STDOUT)` を実行。
     - `env` は `--force_cpu` 指定時のみ `CUDA_VISIBLE_DEVICES=''` を追加。
4. `multiprocessing.Pool(processes=num_workers)` で 3. を t_w 単位に並列実行。
5. 全体の成功/スキップ/失敗件数をメインログに出力。

### プロセス数（`--num_workers`）の決め方
- ディレクトリサイズの実測（`1min`=77MB, `60min`=3.5GB）から、t_wが大きいほど1ワーカーあたりのメモリ使用量が増える。
- 実装前の準備作業として、`10min`, `20min`, `30min`, `40min`, `50min` 相当のディレクトリサイズも実測し、増加傾向を確認した上で、
  - 単一の固定値にするか、
  - `precompute_minimized_embeddings.py` のように **t_w 範囲でティア分割**（例: t_w=1〜20 は並列多め、21〜60 は並列少なめ）にするか
  を決定する。
- 現在ホストの空きメモリが逼迫している（約6.8GB）ため、**初回実行時は `--num_workers` を小さく（2程度）・`--test_tw` で数点だけ試す**ことを強く推奨する運用ルールをスクリプトのdocstringにも明記する。

## 未確定事項（実装前に確認・決定したい）

1. `10/20/30/40/50min` ディレクトリの実サイズ（増加傾向の確認、ティア分割要否の判断材料）。
2. 現在ホストの空きメモリが約6.8GBまで逼迫している原因（他プロセスの利用状況）。実行前に解消 or 別ホスト/タイミングでの実行を検討すべきか。
3. GPU (`NVIDIA GB10`) が統合メモリ方式であることの最終確認（`nvidia-smi` で `memory.total`/`memory.used` が `N/A` だった点の裏付け）。これにより `--force_cpu` を既定にすべきか判断する。
4. `--num_workers` の既定値（2 / 4 など）と、ティア分割を導入するかどうかの最終決定。
5. 5回の試行 (`for exp in range(5)`) は `train_twitter_MPNN.py` 内部でそのまま実行されるため変更しないが、1 t_w あたりの実行時間見積り（モデル×5試行×100エポック）を確認し、全60 t_w × 並列数での総所要時間を見積もる。

## リスクと対策まとめ

| リスク | 対策 |
|---|---|
| 出力ファイルの上書き | t_wごとにcwdを分離（コード改変なしで解決） |
| ホストメモリ枯渇によるOOM killer（tmuxごと巻き添え） | 保守的な`--num_workers`既定値、事前のメモリ実測ログ、`--test_tw`での小規模検証を先に実施 |
| GPU(統合メモリ)への並列アクセス集中 | `--force_cpu`オプションで必要に応じてCPU実行に切替可能にする |
| 縮小データセットが`rww_attr=degree`以外に対応していない | オーケストレータ側で`rww_attr`既定値を`degree`にし、他の値を指定した場合は警告を出す |
| 途中でプロセスが落ちた場合の再実行コスト | 冪等性チェック（結果JSON存在時はスキップ）で再実行時に完了済み分を飛ばせるようにする |

## 実装ステップ

1. `train_minimized_snapshot_mpnn.py` の雛形作成（argparse, ログ設定, ディレクトリ探索）。
2. コマンド組み立て・`subprocess.run`によるt_w単体実行部分の実装（`--dry_run`で動作確認）。
3. `multiprocessing.Pool`による並列化・冪等性チェックの実装。
4. `--test_tw`（例: `1,10,30,60`）で少数t_wを`--num_workers 1`〜`2`で試験実行し、メモリ使用量を`watch free -h`等で監視しながら安全な並列数を実測で決定。
5. 決定した`--num_workers`（必要ならティア分割）を既定値としてスクリプトに反映。
6. 全体実行（t_w=1〜60）。
