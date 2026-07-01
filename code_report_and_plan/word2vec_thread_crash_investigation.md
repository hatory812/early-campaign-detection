# `precompute_minimized_embeddings.py` 実行中に tmux 上へ散発的に出る `TypeError: Cannot convert list to numpy.ndarray` の調査結果

調査日: 2026-07-01

## 症状

- `python3 precompute_minimized_embeddings.py --test_tw 5,40,300` の実行中、tmux 画面に散発的に以下の例外が出力される。

```
Exception in thread Thread-8126 (_worker_loop):
Traceback (most recent call last):
  File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.12/threading.py", line 1010, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/venv/lib/python3.12/site-packages/gensim/models/word2vec.py", line 1166, in _worker_loop
    tally, raw_tally = self._do_train_job(data_iterable, alpha, thread_private_mem)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/gensim/models/word2vec.py", line 957, in _do_train_job
    tally += train_batch_cbow(self, sentences, alpha, work, neu1, self.compute_loss)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "gensim/models/word2vec_inner.pyx", line 649, in gensim.models.word2vec_inner.train_batch_cbow
TypeError: Cannot convert list to numpy.ndarray
```

- 対応するワーカーログ（`/hss01/A.hattori/log/precompute_tw{t_w}.log`）にはこの例外は**一切記録されない**。

## 結論

**gensim 4.4.0 の `Word2Vec`（`hs=1`）に存在する、語彙サイズ1のときに限って Huffman 木の `code`/`point` 属性が `np.ndarray` ではなく素の Python リストのままセットされてしまうバグ**が原因。

`kcore_rww.py` の `get_embedding()` が `workers=4` のマルチスレッドで学習しているため、この不正なリストに対して Cython 側 (`train_batch_cbow`) が `np.PyArray_DATA()` を呼ぶタイミングでワーカースレッドがクラッシュする。

さらに重要な点として、**この例外はスレッド内で発生するため `process_one_graph` の `try/except`（`precompute_minimized_embeddings.py:142`）では捕捉できず、パイプラインは「成功」として処理を続行してしまう**（サイレント障害）。

## 発生条件（語彙サイズ1になる部分グラフ）

「語彙サイズ1」＝ `get_rww()` が生成するランダムウォークが1本だけ（＝部分グラフのノード数が実質1個）になるケース。

`load_and_filter()`（`precompute_minimized_embeddings.py:74`）で t_w によるエッジフィルタ後、**セルフループのみを持つノード1個の部分グラフ**が残る場合にこれが起こる：

1. セルフループ (`u == v`) はエッジ数としてカウントされるため、`graph.number_of_edges() == 0` チェック（L109）を通過する。
2. `nx.isolates()`（L106）はセルフループを持つノードを孤立ノードとみなさない（in/out 次数が 0 ではないため）ので、除去されずに残る。
3. `get_rww()`（`kcore_rww.py:38`）内で無向化した `u_graph` からセルフループを `remove_edges_from(nx.selfloop_edges(...))` で除去する（L40）ため、結局そのノードは次数0の孤立ノード扱いになる。
4. その結果 `random_walks` にはそのノード1個だけの長さ1のウォーク `[node]` が1本だけ追加され、`Word2Vec` の語彙サイズが1になる。

## 根本原因（gensim 側のバグ）

`gensim/models/word2vec.py` の `_assign_binary_codes()`：

```python
def _assign_binary_codes(wv):
    heap = _build_heap(wv)
    ...
    stack = [(heap[0], [], [])]
    while stack:
        node, codes, points = stack.pop()
        if node[1] < len(wv):  # leaf node
            # 語彙サイズが1のとき、ルート = 唯一の葉なのでここに即座に入り、
            # 空リスト [] がそのまま code/point としてセットされる（np.array化されない）
            k = node[1]
            wv.set_vecattr(k, 'code', codes)
            wv.set_vecattr(k, 'point', points)
            ...
        else:
            # 通常はここで np.array に変換される
            points = np.array(list(points) + [node.index - len(wv)], dtype=np.uint32)
            stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
            stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))
```

語彙サイズが2以上なら常に `else` 分岐（`np.array` へ変換）を通るが、**語彙サイズ1のときだけ**最初の pop で `node[1] < len(wv)` が真になり、初期値の空リスト `[]`（Python list）がそのまま `code`/`point` に設定されてしまう。

`gensim/models/word2vec_inner.pyx` の `train_batch_cbow`（L649 付近）：

```cython
c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
```

`vocab_codes[word_index]` が Python list だと `np.PyArray_DATA()` が `TypeError: Cannot convert list to numpy.ndarray` を送出する。

ただし通常は `sample=0.001` によるダウンサンプリングで、この唯一の単語は各エポックでほぼ確率的にスキップされる（ログ上は `training on 1 raw words (0 effective words)` が繰り返される）。まれに（1エポックあたり約3%）ダウンサンプリングを生き残ると上記コードパスに到達しクラッシュする。`epochs=100` かつテスト対象グラフ数が多いため、実行全体で見ると散発的にヒットする。

### 再現コード（apptainer コンテナ内、`env_gnn_arm.sif`）

```python
import logging
logging.disable(logging.CRITICAL)
from gensim.models.word2vec import Word2Vec

walks = [[0]]  # ノード1個・長さ1のウォーク1本 => 語彙サイズ1
for trial in range(20):
    model = Word2Vec(walks, hs=1, alpha=0.0001, epochs=100, vector_size=128,
                      window=5, min_count=1, workers=4, seed=trial)
```

数回〜十数回の試行で `Exception in thread ... TypeError: Cannot convert list to numpy.ndarray` を安定して再現できた。

## 実害（なぜ危険か）

`threading.Thread` はスレッド内で発生した例外を `run()` 内で握りつぶし、標準エラーに出力するだけで **`join()` を呼んでいるメインスレッドには伝播しない**。そのため：

- `Word2Vec.train()`（メインスレッド）は例外を検知できず、正常終了したものとして処理が続く。
- `process_one_graph`（`precompute_minimized_embeddings.py:117`）の `try/except` にも引っかからず、`success=True` として扱われる。
- ワーカーログにも記録されず、`n_ok` としてカウントされ、embedding が保存される。
- クラッシュしたスレッドが担当していたバッチ分の学習が欠落した状態で `model.wv` が使われる可能性があり、**サイレントなデータ品質劣化**につながりうる。

## 副次的な問題

ワーカーログに以下の warning が全件で出ている：

```
Both hierarchical softmax and negative sampling are activated. This is probably a mistake. You should set either 'hs=0' or 'negative=0' to disable one of them.
```

`get_embedding()`（`kcore_rww.py:161`）で `hs=1` のみ明示しているが `negative` を指定していないため、デフォルトの `negative=5` も同時に有効になっている。意図しない設定と思われる。

## 対処の方向性

1. **（推奨）語彙サイズ1になる部分グラフを学習前にスキップする**
   `process_one_graph`（`precompute_minimized_embeddings.py:117`）で、`get_rww()` 呼び出し後（または `u_graph` のノード数取得後）に語彙サイズ（ウォーク本数）が1以下なら `get_embedding()` を呼ばずスキップ扱いにする。学習しても意味のあるベクトルにならないデータでもあるため、根本原因を踏まずに済む。
2. **`negative=0` を明示する**（`kcore_rww.py:170` 付近）
   `hs=1` のみを使う意図であれば `negative=0` を追加し、warning の解消とコードパスの単純化を行う。
3. **`workers=1` に変更する**（`kcore_rww.py:170`）
   並列化は既に `multiprocessing.Pool`（t_w 単位）で行われているため、`Word2Vec` 内のスレッド並列は不要。`workers=1` にすればこの種のマルチスレッド起因の問題を回避できる（ただし上記の gensim 側バグ自体は `workers=1` でも理論上再現しうるため、対策1と併用が望ましい）。

## 確認方法

- 対象ワーカーログ (`/hss01/A.hattori/log/precompute_tw{t_w}.log`) 内で `"collected 1 word types"` を検索すると、語彙サイズ1になった箇所を特定できる。
- 修正後は、上記再現コード（`walks = [[0]]`, `epochs=100`, 複数 seed で試行）が例外を出さなくなることで確認できる。
