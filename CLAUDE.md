# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリの位置づけ

上流は DECODE (Density-aware Walks for Coordinated Campaign Detection, ECML/PKDD 2025) の公式実装 (`README.md` 参照)。本リポジトリはそれをフォークし、**「カスケードグラフをどれだけ観測すれば組織的キャンペーンを検出できるか」= 観測期間 t_w の掃引実験**を追加したもの (`研究概要.md`)。上流由来コード (`train_twitter_MPNN.py`, `kcore_rww.py`, `models.py`, `k_truss.py`, `run/`) には再現を妨げるバグが多数あり、調査・修正の記録が `code_report_and_plan/` にある (`REPRODUCTION_ISSUES.md` → `IMPLEMENTATION_PLAN.md` の順に読むと経緯が分かる)。

## 実行環境

ホストの `python3` には torch も PyG も入っていない。**Singularity/Apptainer コンテナ内で実行する**:

```bash
cd /home/A.hattori && ./runSingularity_gnn03.sh    # ホストごとに .sif が違う (gnn01/gnn02/gnn03/now91)
# 中身: singularity run --nv --bind /hss01:/hss01 --bind ${PWD}:/${PWD} ./env_gnn_arm.sif
```

長時間実験は tmux 上で回す (`tmux attach -t session1`)。`requirements.txt` は上流のままで実態と合っていない (gensim 欠落、`pytorch-geometric` はパッケージ名誤り、jax 不要) ので、環境構築の参照元にしないこと。

## データ配置 (リポジトリ外・`.gitignore` 済み)

| パス | 内容 |
|---|---|
| `/hss01/A.hattori/all_graphs/` | LEN データ本体 (`*_fulldata.json`) と `graph_labels*.json`。**ラベルファイルの場所は各スクリプトにハードコード** |
| `/hss01/A.hattori/rww_all_graphs/{degree,kcore,ktruss}/{0.5,mid,median}/` | RWW 埋め込み付きフルグラフ。学習の入力はここ (実験で常用しているのは `degree/mid`) |
| `/hss01/A.hattori/rww_all_graphs_minimized/<t_w>min/` | t_w で縮小したうえで埋め込みを**再計算**したデータセット |

1 ファイル最大 4GB、`degree/mid` 全体で約 112GB。`json.load` は簡単に数十 GB 食うので、全件走査するコードを書くときは規模を先に見積もること。

## よく使うコマンド

```bash
# RWW 埋め込み生成 (all_graphs → rww_all_graphs)。density metric ごとに run/{degree,core,truss}/gen_rww.sh
python3 kcore_rww.py --path ./all_graphs --graphId 1 --pick degree --comp 0.5

# フルグラフでの二値分類 (5 試行、結果は ./results/*.json)。run/degree/ で実行する前提のパス
cd run/degree && bash binary.sh
python3 ../../train_twitter_MPNN.py --model GCN --data_type all --hidden_dim 1024 --lr 0.0001 \
  --output_dim 2 --all_graphs_path /hss01/A.hattori/rww_all_graphs/degree/mid \
  --multivariate 0 --rww_attr degree --node_attr 1

# t_w 掃引 (オンザフライ縮小・埋め込み流用)
cd run/degree && bash snapshot.sh

# 学習なしで各 t_w の |V|,|E| 統計だけ出す
cd run/degree && bash snapshot_size_report.sh

# t_w ごとの結果 JSON を 1 本の集計 CSV にまとめる (掃引の後処理その 1)
python3 aggregate_snapshot_results.py results/<実験ディレクトリ> --tw_min 0 --tw_max 60

# 既存の集計 CSV から図を再生成 (--baseline で全 campaign 予測のベースラインを重ねる)
python3 plot_snapshot_metrics.py results/<実験ディレクトリ> --metrics f1 accuracy --baseline

# 縮小グラフの入出次数ヒストグラム (mmap+正規表現で JSON をパースせず走査)
python3 plot_degree_histograms.py --t-w 1min 2min --jobs 8 --out-root results/<実験ディレクトリ>
```

テストスイートやリンタは無い。検証は「小さい t_w / 少数グラフで 1 回回して出力を見る」形で行う。

## アーキテクチャ

パイプラインは 3 段:

1. **構造埋め込み生成** — `kcore_rww.py` がノードに密度スコア (degree / kcore / ktruss、min-max 正規化) を付け、それでバイアスした重み付きランダムウォーク (RWW) を回し、Word2Vec でノード埋め込みを得る。埋め込みはグラフ JSON のノード属性として書き戻される。属性名は metric ごとに違う: `degree_embedding` / `structural_embedding` (kcore) / `truss_embedding`。ktruss だけは Sariyuce et al. の nucleus コードで事前計算した `.mtx_23_K_values` を `k_truss.py:load_k_truss()` が読む前提。
2. **特徴量の組み立て** — 学習側は `--rww_attr {degree,kcore,ktruss}` × `--node_attr {0,1}` の分岐でノード特徴を作る (`train_twitter_MPNN.py:process_data`, `train_twitter_snapshot.py:_node_feature`)。`node_attr=1` は `node_attr + <metric>_embedding` の連結、`0` は埋め込みのみ。**この分岐は 2 ファイルに重複しているので片方だけ直さないこと。**
3. **学習・評価** — `models.py` の `GCN` は conv 層を外から注入する薄いラッパで、GCN/GAT/SAGE/GIN を差し替えられる (GINE だけ `GCN_edge`)。`train_twitter_MPNN.py` が seed 0..4 の 5 試行を回し、試行ごとに train/test を再分割して平均±標準偏差を出す。

### ファイル名とラベルの対応

グラフのファイル名は `<トピック名>_{campaign,noncampaign}_fulldata.json`。ラベル引きは `file_name[:-9]` (`_fulldata` を落とす) を `graph_labels.json` のキーにする、という約束が全スクリプトに散在している。壊れているグラフは各スクリプト内の `exceptions` / `EXCEPTIONS` リスト (`Gomis_noncampaign_fulldata` など 6 件) でスキップ。このリストは `train_twitter_MPNN.py` / `train_twitter_snapshot.py` / `precompute_minimized_embeddings.py` / `plot_snapshot_metrics.py` / `plot_degree_histograms.py` の **5 ファイルに複製されている**ので、除外対象を増減させるときは全部直すこと。`graph_name_mapping.json` はグラフ名 → 連番 ID (1..323) の対応で、`kcore_rww.py --graphId` 実行用。

**既知のバグ: 除外リストが 1 件効いていない。** ディスク上のファイル名は Unicode NFD (`ş` = `s` + 結合セディーユ) だが、`.py` 中のリテラルは NFC で書かれているため、生の文字列比較では `35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata` が一致せず素通りする (`#Hıdırellez` の `ı` は単一コードポイントなので影響なし)。`unicodedata.normalize('NFC', ...)` を挟んでいるのは `plot_degree_histograms.py` だけで、学習側の 4 ファイルは未対応。過去の実験結果はこのグラフを含んだまま集計されている可能性がある。

### t_w 掃引の 2 経路 — 未来情報リークに注意

* **`train_twitter_snapshot.py`**: フルグラフ上で計算済みの埋め込みを**流用**し、タイムスタンプでエッジをマスクするだけ。JSON パースは 1 回で済むぶん速いが、**縮小グラフのノードがグラフ全体の情報を持つ埋め込みを保持したまま = 未来情報リーク**であることが実験途中で判明した (`results/20260627_.../intro.txt`)。トポロジ縮小の純粋効果を見る設定としてのみ有効。
* **`precompute_minimized_embeddings.py` + `train_minimized_snapshot_mpnn.py`**: 縮小後のグラフで RWW と Word2Vec を**再計算**してから `train_twitter_MPNN.py` を t_w ごとに subprocess 起動する。リークが無いのはこちら。2026-07 以降の `results/` はすべてこの経路。掃引後の集計・作図は `aggregate_snapshot_results.py` → `plot_snapshot_metrics.py` の順。

### ブランチ構成

2026-09-01 に `feature/each-embedding` を main へマージ (`b6a6778`) したため、**両経路とも main に揃っている**。それ以前の CLAUDE.md にあった「再計算パイプラインは feature ブランチにのみ存在」という制約は解消済み。

| ブランチ | 位置づけ |
|---|---|
| `main` | 統合先。origin (`hatory812/early-campaign-detection`) へ push する |
| `feature/each-embedding` | 縮小グラフでの埋め込み再計算パイプライン。`b6a6778` で main にマージ済み |
| `feature/analysis-scripts` | 掃引結果の集計・作図 (`aggregate_snapshot_results.py`, `plot_snapshot_metrics.py`) と次数分布調査 (`degree_extract.py`, `plot_degree_histograms.py`)。fast-forward で main にマージ済み |

かつて存在した `feature/experimental-script` (t_w 掃引の第 1 世代) と `fix` (上流コードの再現性バグ修正群) は、独自コミットを持たない履歴上の位置マーカーになっていたため削除した。両者の成果は main の履歴に含まれている。

リモートは `origin` の 1 本だけで、フォーク元 `erdemUB/ECMLPKDD25` を追える remote は登録されていない (必要なら `git remote add upstream https://github.com/erdemUB/ECMLPKDD25.git`)。

### メモリ

`train_twitter_snapshot.py` は 312 グラフを全部メモリに常駐させ、さらに t_w ごとに全スナップショットを複製するため、t_w=600 で必ずホスト全体の OOM killer が発火し tmux セッションごと落ちた (`code_report_and_plan/snapshot_oom_investigation.md`)。`train_minimized_snapshot_mpnn.py` はその反省から既定 `num_workers=1`、空きメモリ監視、`--dry_run` / `--test_tw` を持つ。大きい t_w を扱うコードを触るときはメモリ増加が t_w に対して単調である前提で見積もること。JSON 全体をパースせずに済むなら `degree_extract.py` の mmap + 正規表現方式を使う (前提条件はファイル冒頭の docstring に書いてある)。

## 出力の置き場

学習スクリプトは**カレントディレクトリ直下の `results/`** に固定名 (`{model}_{rww_attr}_nodeattr{n}_{data_type}_mv{mv}.json`、掃引は `snapshot_{model}_{rww_attr}_nodeattr{n}.{json,csv,png}`) で書く。t_w が名前に入らないため、同じ cwd で複数実験を回すと上書きされる (`train_minimized_snapshot_mpnn.py` が t_w ごとに cwd を分けているのはこのため)。

実験結果は `results/<YYYYMMDD>_<日本語の実験名>/` に 1 実験 1 ディレクトリで残す (`train_minimized_snapshot_mpnn.py` は `--out_root` にこのパスを直接渡し、`--log_dir` 配下に t_w ごとのログを吐く)。`results/` は `.gitignore` 済みなのでコミットされない。実験の意図や中断理由は同ディレクトリの `intro.txt` に残す。コミットメッセージは日本語で `add:` / `fix:` / `move:` 接頭辞。

## プロンプトに対する回答

私が研究を進めていく中でわからないことがあればあなたに質問する場合があります。
その質問は次のようにカテゴライズされると思います。
- 語彙の未知：技術用語や社内用語、専門用語などその言葉の意味がわからない
- 構造の未知：処理フローや流れ、あるものとあるもののつながり、論理の展開が理解できていない。
- その他：その他の質問。

私から質問された場合、まず私の質問を上記の項目でカテゴライズしてください。
その後に各項目の質問に対して次のように回答してください。
- 語彙の未知：その語彙が現れるコンテキストを踏まえて、200文字以内で説明してください。
- 構造の未知：その構造を一言で表したのち、その構造の詳細を1000文字以内で説明してください。
- その他：あなたが考える最適な回答を作成し、なるべく簡潔に説明してください。