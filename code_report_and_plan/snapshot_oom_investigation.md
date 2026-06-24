# `run/degree/snapshot.sh` が t_w=600 で必ず止まる件の調査結果

調査日: 2026-06-24

## 症状

- `run/degree/snapshot.sh` を実行すると、**必ず時間幅 t_w=600 の結果が出たところで実験が終了する**。
- tmux 上で実行しているが、t_w=600 の結果が出たところで tmux が **`no session`** になる（セッションごと消える）。

## 結論

**ホストの物理メモリ枯渇によるグローバル OOM kill**。
Linux カーネルの OOM killer が発火し、python プロセスだけでなく、**同じ session スコープ内にある tmux サーバごと巻き込んで kill する**ため、セッションが消えて `no session` になる。

t_w=600 で「必ず」死ぬのは運やバグではなく、**メモリ使用量が t_w に対して単調増加し、600 が使用可能メモリ（RAM+Swap）を超える最初の点**だから、という決定的な現象。

## 根拠（調査ログ）

### 1. cgroup の OOM イベント

セッションの cgroup (`/user.slice/user-1000014.slice/session-28333.scope`) の `memory.events`:

```
low 0
high 0
max 0
oom 0
oom_kill 3
oom_group_kill 0
```

- `oom_kill 3` … OOM killer が実際に 3 回プロセスを kill している。
- `memory.max = max` / `MemoryMax = infinity` … cgroup の割当上限ではなく、**ホスト全体のメモリ枯渇（グローバル OOM）**であることを示す。

### 2. システムメモリ

```
               total        used        free      shared  buff/cache   available
Mem:           119Gi       5.3Gi       114Gi       704Ki       993Mi       114Gi
Swap:           15Gi       3.0Gi        12Gi
```

利用可能上限は RAM 119GB + Swap 15GB ≒ 134GB。

### 3. データセット規模

`/hss01/A.hattori/rww_all_graphs/degree/mid`:

- `*_fulldata.json` が **312 ファイル / 合計 約112GB**。
- 単一ファイルの最大は **3.9GB**（例: `Anadolu_Ajansı_noncampaign_fulldata.json` 3.9G, `Alevi_... 3.6G`, `#23Nisan_... 2.8G` など複数 GB 級が多数）。

## メモリが膨らむ仕組み（`train_twitter_snapshot.py`）

1. **`build_cache`（L88〜）** が 312 グラフを**一度に全部パースして numpy で常駐**させる。
   - 全 t_w で使い回して再パースを避ける設計だが、その代償として「全グラフ常駐」を強制し、メモリの底上げ（固定の大コスト）になっている。
2. **`run_one_tw`（L300〜）** の `data_list = [build_snapshot(c, t_w) for c in cache]`（L305）で、各 t_w ごとに**全グラフのスナップショットを丸ごと複製**する。
   - t_w が大きいほど、しきい値 `t0 + t_w*60` 秒以内に生き残るエッジ・ノードが増える → 複製サイズが増大。
3. したがって総メモリは

   ```
   総メモリ ≒ cache(固定・大) + data_list(t_w に比例して増加) + 学習中のバッファ
   ```

   となり、t_w=600（10時間ぶんのスナップショット）で 134GB(RAM+Swap) を超えて OOM が発火する。

## なぜ tmux ごと落ちるのか

OOM killer 発火時、対象プロセスは同一 session スコープ内から選ばれる。tmux サーバも同じスコープに属しているため巻き添えで kill され、結果として `no session` になる。

## 対処の方向性

`cache` を全 t_w で使い回す設計が「全グラフ常駐」を強制し、メモリの天井を作っているのが根本原因。

1. **案1（推奨）: ストリーミング化**
   全グラフを常駐させず、1 グラフずつパース → 必要な t_w 範囲のスナップショットだけ抽出 → 解放。再パースコストは増えるが、メモリ上限が一定になる。
2. **案2: キャッシュ軽量化 / 明示解放**
   `node_feat`/`edge_attr` の dtype を必要最小に。各 t_w 終了時に `data_list` を `del` + `gc.collect()` で明示解放（現状は次の t_w 開始まで前の分が残り、重なる瞬間がある）。
3. **案3: t_w 分割実行 + tmux を OOM 対象外に**
   `--tw_min 60 --tw_max 540` と `--tw_min 600 --tw_max 1440` のようにプロセスを分け、各プロセスを使い捨てて cache 常駐を解放する。あわせて tmux サーバに `oom_score_adj=-1000` を設定すれば、少なくとも「セッション消滅」は防げる。
4. **その場しのぎ**: Swap 増設、または大容量 RAM ノードでの実行。

## 確認方法

実行中に別ペインで監視すると挙動を裏取りできる:

```bash
watch -n 5 'free -h; cat /sys/fs/cgroup$(cat /proc/self/cgroup | cut -d: -f3)/memory.events 2>/dev/null'
```

`oom_kill` のカウントが t_w=600 到達時にインクリメントされるはず。
