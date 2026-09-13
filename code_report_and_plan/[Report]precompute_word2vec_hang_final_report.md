# `precompute_minimized_embeddings.py` 実行停止（ハング）の最終原因調査レポート

調査日: 2026-07-01
関連ドキュメント: [[Report]word2vec_thread_crash_investigation.md]([Report]word2vec_thread_crash_investigation.md)（tmux上のエラー出力に関する先行調査）

## 概要

`python3 precompute_minimized_embeddings.py --test_tw 5,40,300`（テストモード、Pool(1)）を実行したところ、tmux 上に散発的な gensim 由来の `TypeError` が出力される事象が確認された（先行調査: [[Report]word2vec_thread_crash_investigation.md]([Report]word2vec_thread_crash_investigation.md)）。

先行調査ではこの事象を「ワーカーログにも `try/except` にも記録されないサイレント障害」と結論づけていたが、実行中プロセスを `/hss01/A.hattori/log` のログおよび `ps` で追跡した結果、**実際にはサイレント障害にとどまらず、パイプライン全体がハングして完全に停止する**ことが確認された。本レポートは両方の調査結果を統合した最終結論と修正方針をまとめる。

## 事象タイムライン

| 時刻 | 内容 |
|---|---|
| 13:27:52 | `precompute_minimized_embeddings.py --test_tw 5,40,300` 起動。`precompute_main.log` に `TEST MODE: t_w=[5, 40, 300], Pool(1)` 記録 |
| 13:27:53 | `precompute_tw5.log` に `worker START t_w=5min` 記録。以降、対象グラフごとに Word2Vec の学習ログが順次記録される |
| 13:28:52 | `precompute_tw5.log` の最終行。**語彙サイズ1**（`training model with 4 workers on 1 vocabulary`）のグラフの学習中、epoch 24/100 まで進んだところでログが途絶える |
| 14:22〜14:23 | ログ調査時点。`precompute_tw5.log` は13:28:52から**54分以上更新なし**。プロセス（メイン PID 2433413 / spawn ワーカー PID 2433454）はまだ生存しており、`ps -T` で worker プロセスのスレッド数を確認したところ **39スレッド**（`Word2Vec(workers=4)` 1回分の想定スレッド数＝5前後を大幅に超過）を抱えたまま `sleeping` 状態だった |
| 15:08 | 再確認したところ、当該プロセス（PID 2433413 / 2433454）は**消滅**していた。`precompute_main.log` に完了ログなし、`precompute_tw5.log` も1930行・13:28:52で止まったまま。`worker DONE` は一度も記録されず、t_w=5 の処理すら完走していない |

**この時系列から、「一時的に遅い」のではなく「該当グラフの学習でハングし、そのまま自己回復せずに終了した（もしくは外部から強制終了された）」ことが確定した。**

## 根本原因（先行調査の要約）

詳細は [[Report]word2vec_thread_crash_investigation.md]([Report]word2vec_thread_crash_investigation.md) を参照。要点のみ以下に記す。

1. `load_and_filter`（`precompute_minimized_embeddings.py:74`）の孤立ノード除去は `nx.isolates()` を使うが、**自己ループのみを持つノードを孤立ノードと判定できない**（自己ループがあると次数が0にならないため）。
2. そのため t_w フィルタ後に「自己ループしか残らないノード1個」の部分グラフが有効なグラフとして通過してしまう。
3. `get_rww`/`get_degree`（`kcore_rww.py`）は計算前に自己ループを除去するため、このノードは実質次数0＝孤立扱いとなり、長さ1のウォーク `[node]` が1本だけ生成される。
4. これが唯一のウォークの場合、`Word2Vec` の**語彙サイズが1**になる。
5. gensim 4.4.0 の `Word2Vec(hs=1)` には、語彙サイズ1のときだけ Huffman木の `code`/`point` 属性が `np.ndarray` 化されず Python の空リストのまま残るバグが存在する。
6. `get_embedding()`（`kcore_rww.py:161`）は `workers=4` でマルチスレッド学習するため、上記の不正なリストに Cython 側 (`train_batch_cbow`) が `np.PyArray_DATA()` でアクセスしようとして、**ワーカースレッド内で `TypeError` が発生**する。

## 実害の確定（今回の追加調査で判明）

先行調査ではスレッド内例外は「メインスレッドに伝播せず、`process_one_graph` の `try/except` にも引っかからないためサイレントに処理が継続する」と推測していた。しかし実際のプロセスを追跡した結果：

- スレッド内で例外が発生した後、**メインスレッドの `Word2Vec.train()` がそのジョブの完了通知を待ち続け、ハングしている**と考えられる（`precompute_tw5.log` が特定のグラフのepoch途中で完全に停止し、それ以降1件もログが出ていないことと整合する）。
- worker プロセスが39スレッドという異常な数を抱えていたことから、**このハングが今回1回限りではなく、それまでの複数グラフ処理でクラッシュ→スレッドリークが蓄積した結果**である可能性が高い（`process_tw` は対象グラフを `for` ループで逐次処理しており、各グラフごとに新しい `Word2Vec` インスタンス＝新しいスレッド群が生成されるため）。
- 最終的にプロセスは完了せず消滅しており、**t_w=5 の処理すら完走できていない**（`n_ok`/`n_skip` の集計ログである `worker DONE` が一度も出力されていない）。

**結論**: これは「サイレントなデータ品質劣化」ではなく、**パイプライン全体を停止させうる致命的な不具合**である。特にテストモード（`Pool(1)`）では1つのハングが全体を止めるが、本番モード（ティア別 Pool）でも、ハングしたプロセスがプールのスロットを永久に占有し続けるため、そのティアの他の t_w の処理も巻き込まれて完了しなくなるリスクがある。

## 修正方針（最終版・優先度順）

### 1.（最優先・根本原因）`load_and_filter` で自己ループを isolates 判定前に除去する

`precompute_minimized_embeddings.py:106` 付近：

```python
graph.remove_edges_from(remove_edges)
graph.remove_edges_from(nx.selfloop_edges(graph))   # 追加
isolates = list(nx.isolates(graph))
graph.remove_nodes_from(isolates)

if graph.number_of_edges() == 0:
    return None
```

自己ループのみのノードを `load_and_filter` の時点で正しく孤立ノードとして除去し、語彙サイズ1になる部分グラフ自体を Word2Vec に渡さないようにする。これにより6章の不具合の発生条件そのものを消す。

### 2.（防御的修正）`kcore_rww.py` の `Word2Vec(workers=4)` を `workers=1` に変更

`multiprocessing.Pool` によるプロセス並列化が既に t_w 単位で行われているため、`Word2Vec` 内のスレッド並列は不要。シングルスレッド化により、スレッド間の非同期更新（Hogwildスタイル）に起因する今回のようなハング・スレッドリーク経路自体をなくす。1と併用することで二重に安全になる。

### 3.（設定の妥当性）`negative=0` を明示する

`hs=1` のみを意図しているにもかかわらず `negative` を無指定にしているため、gensim デフォルトの `negative=5` が同時に有効になっている（全ワーカーログに `Both hierarchical softmax and negative sampling are activated` の warning が出続けている）。今回のハングの直接原因ではないが、意図した学習設定との乖離であり、あわせて修正すべき。

### 4.（任意）ランダムウォーク生成のシード固定

`kcore_rww.py` の `random.choices`（RWWの経路選択）にシードがなく、実行のたびに埋め込みが変わる。再現性が必要であれば `process_tw` 冒頭などで `random.seed()` を固定する。

## 今後のアクション

1. 上記1・2・3の修正を `precompute_minimized_embeddings.py` / `kcore_rww.py` に適用する
2. 修正後、`--test_tw 5,40,300` で再実行し、以下を確認する
   - `precompute_tw{t_w}.log` に `worker DONE` が出力され、正常終了すること
   - tmux 上に `TypeError: Cannot convert list to numpy.ndarray` が出力されないこと
   - ワーカーログから `Both hierarchical softmax and negative sampling are activated` warning が消えること
3. 既に生成済みの `rww_all_graphs_minimized/` 配下の出力について、修正前の不完全な実行分（今回の t_w=5 のように完走していないもの）が残っていないか確認し、必要であれば再生成する
