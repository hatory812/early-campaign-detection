"""各グラフのノード入次数 / 出次数のヒストグラムを描く。

rww_all_graphs_minimized/<t_w>/ 以下の *_fulldata.json を読み、グラフ 1 個につき
PNG 1 枚 (左: 入次数, 右: 出次数) を出力する。あわせて全グラフ分の統計 CSV と、
全ノードをプールした集約ヒストグラム (campaign / noncampaign 別) も出力する。

グラフは有向・多重辺なしとして扱う。次数は edges から直接数え、孤立ノード (次数 0)
も nodes に載っていれば 0 として集計に含める。JSON は 1 ファイル最大 4GB あるので
json.load は使わず degree_extract.degrees_from_file で構造キーだけを拾う。

次数分布は裾が非常に長い (t_w=1440min の入次数は最大 21536) ため、最大次数が
SYMLOG_X_MIN_MAX_DEGREE を超えたら横軸を symlog にして棒ではなく点で描く。線形軸
のままだと幅 1 の棒がサブピクセルに潰れてハブが消え、軸の右 9 割が空白に見える。

注意: 図中の文字は ASCII とトルコ語 (Latin Extended-A) のみにしている。既定フォント
の DejaVu Sans に日本語グリフが無く、日本語を入れると豆腐 (□) になるため。

使い方:
    # 単一の時間幅
    python3 plot_degree_histograms.py --t-w 1min \
        --out-dir "results/20260804_グラフノードの入出次数の統計/1min"

    # 複数の時間幅をまとめて (出力先は <out-root>/<t_w>/)
    python3 plot_degree_histograms.py --t-w 2min 3min --jobs 8 \
        --out-root "results/20260804_グラフノードの入出次数の統計"
"""

import argparse
import csv as csvlib
import glob
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from degree_extract import degrees_from_file

DEFAULT_DATA_ROOT = '/hss01/A.hattori/rww_all_graphs_minimized'
DEFAULT_LABEL_PATH = '/hss01/A.hattori/all_graphs/graph_labels.json'

# train_twitter_MPNN.py / plot_snapshot_metrics.py と同じ除外リスト。
# ディスク上のファイル名は Unicode NFD なので、比較前に両側を NFC 正規化する
# (既存コードは生の文字列比較なのでトルコ語名が一致せず素通りしていた)。
EXCEPTIONS = {
    'graph_labels',
    'Gomis_noncampaign_fulldata',
    '#Hıdırellez_noncampaign_fulldata',
    '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata',
    'Haluk_noncampaign_fulldata',
    '#ErdenTimurSezonu_noncampaign_fulldata',
    'Gustavo_noncampaign_fulldata',
}

LABEL_NAMES = {0: 'noncampaign', 1: 'campaign'}
IN_COLOR = '#4C72B0'
OUT_COLOR = '#DD8452'
AGG_COLORS = (IN_COLOR, OUT_COLOR)
# 次数の裾が広いグラフは線形軸だと 0 の棒しか見えないので log 軸に切り替える。
LOG_SCALE_MAX_DEGREE = 20
# 入次数はハブが極端で、t_w=1440min では最大 21536 に達する。線形の横軸だと
# 幅 1 の棒がサブピクセルに潰れてハブが「描かれていない」ように見え、しかも
# 軸の大半が空白になる。最大次数がこれを超えたら symlog 軸 + 点描に切り替える
# (linthresh=1 なので次数 0 のノードも落とさずに描ける)。図の幅 600px 程度に対し
# 300 なら棒 1 本が 2px 残るので、この辺りが棒で描ける限界。
SYMLOG_X_MIN_MAX_DEGREE = 300


def nfc(s):
    return unicodedata.normalize('NFC', s)


def safe_filename(name):
    """ファイル名に使えない文字を _ に置換する。"""
    return re.sub(r'[/\\:*?"<>|]', '_', name)


def int_bins(hi):
    """0..hi の整数次数を棒の中心に置くためのビン境界。"""
    return np.arange(-0.5, hi + 1.5, 1.0)


def _plot_counts(ax, counts, color, symlog_x, label=None, alpha=1.0):
    """bincount 済みの度数 counts を描く。軸の設定は _style_axis 側で行う。

    symlog_x のときは棒ではなく点で描く。棒だと横軸が数万まで伸びたときに
    幅 1 の棒がサブピクセルになって消えるが、点なら 1 ノードでも必ず見える。
    """
    if symlog_x:
        nz = np.nonzero(counts)[0]
        ax.plot(nz, counts[nz], marker='.', markersize=4, linestyle='none',
                color=color, alpha=alpha, label=label)
    else:
        hi = len(counts) - 1
        ax.bar(np.arange(hi + 1), counts, width=1.0, color=color, alpha=alpha,
               edgecolor='white', linewidth=0.6 if hi < 60 else 0, label=label)


def _style_axis(ax, hi, symlog_x, title):
    """横軸の目盛りと範囲を最大次数 hi に合わせる (hi は同一 ax 上の全系列の最大)。"""
    if symlog_x:
        ax.set_xscale('symlog', linthresh=1, linscale=0.5)
        # 右端に無駄な余白を作らないよう、実在する最大次数のすぐ外で切る。
        ax.set_xlim(-0.3, hi * 1.3)
        # 既定の指数表記 (10^4) だと次数として読みづらいので 0, 1, 10, ... と出す。
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{v:.0f}'))
    else:
        ax.set_xlim(-0.6, hi + 0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if hi > LOG_SCALE_MAX_DEGREE:
        ax.set_yscale('log')
    ax.set_xlabel('degree')
    ax.set_ylabel('number of nodes')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


def _draw_hist(ax, counts, color, symlog_x, title):
    _plot_counts(ax, counts, color, symlog_x)
    _style_axis(ax, len(counts) - 1, symlog_x, title)


def plot_one(name, label, in_counts, out_counts, n_nodes, n_edges, in_mean, out_mean,
             out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # 同じ図の左右で棒と点が混ざると読みづらいので、軸方式は 2 パネル通しで決める
    # (横軸の範囲だけはパネルごとの最大次数に合わせる)。
    symlog_x = max(len(in_counts), len(out_counts)) - 1 > SYMLOG_X_MIN_MAX_DEGREE
    _draw_hist(axes[0], in_counts, IN_COLOR, symlog_x,
               f'In-degree  (mean {in_mean:.2f}, max {len(in_counts) - 1})')
    _draw_hist(axes[1], out_counts, OUT_COLOR, symlog_x,
               f'Out-degree  (mean {out_mean:.2f}, max {len(out_counts) - 1})')
    fig.suptitle(f'{name}   [{LABEL_NAMES.get(label, label)}]   N={n_nodes}, E={n_edges}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def process_file(job):
    """1 グラフぶんの PNG を書き、集計用のレコードを返す (ワーカで実行)。"""
    path, label, per_graph_dir = job
    file_name = os.path.basename(path)[:-5]
    in_deg, out_deg, n_edges = degrees_from_file(path)

    in_counts = np.bincount(in_deg)
    out_counts = np.bincount(out_deg)
    plot_one(file_name, label, in_counts, out_counts, len(in_deg), n_edges,
             in_deg.mean(), out_deg.mean(),
             os.path.join(per_graph_dir, safe_filename(file_name) + '.png'))

    return {
        'name': file_name, 'label': label, 'n_nodes': len(in_deg), 'n_edges': n_edges,
        'in_mean': float(in_deg.mean()), 'in_std': float(in_deg.std()),
        'in_max': int(in_deg.max()), 'in_zero_ratio': float((in_deg == 0).mean()),
        'out_mean': float(out_deg.mean()), 'out_std': float(out_deg.std()),
        'out_max': int(out_deg.max()), 'out_zero_ratio': float((out_deg == 0).mean()),
        'in_counts': in_counts, 'out_counts': out_counts,
    }


def pool_counts(records, key, label):
    """同じラベルのグラフの bincount を足し合わせる (長さは最大次数に合わせる)。"""
    sel = [r[key] for r in records if r['label'] == label]
    if not sel:
        return np.zeros(1, dtype=np.int64)
    total = np.zeros(max(len(c) for c in sel), dtype=np.int64)
    for c in sel:
        total[:len(c)] += c
    return total


def plot_aggregate(records, out_path, t_w):
    """全グラフのノードをプールした入次数 / 出次数分布 (ラベル別)。"""
    labels = sorted({r['label'] for r in records})
    # 軸は両ラベルを通した最大次数に合わせる (片方だけで決めると切れる)。棒か点かは
    # plot_one と同じく 2 パネル通しで決め、図の中で表現が混ざらないようにする。
    pooled_all = {(key, lbl): pool_counts(records, key, lbl)
                  for key in ('in_counts', 'out_counts') for lbl in labels}
    symlog_x = max(len(c) - 1 for c in pooled_all.values()) > SYMLOG_X_MIN_MAX_DEGREE

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], 'in_counts', 'In-degree'),
                           (axes[1], 'out_counts', 'Out-degree')):
        pooled = {lbl: pooled_all[(key, lbl)] for lbl in labels}
        hi = max(len(c) - 1 for c in pooled.values())
        for i, lbl in enumerate(labels):
            counts = pooled[lbl]
            _plot_counts(ax, counts, AGG_COLORS[i % len(AGG_COLORS)], symlog_x,
                         label=f'{LABEL_NAMES.get(lbl, lbl)} (n={int(counts.sum())})',
                         alpha=0.55)
        _style_axis(ax, hi, symlog_x, f'{title} pooled over all graphs')
        ax.set_yscale('log')
        ax.legend()

    fig.suptitle(f'Degree distribution over all graphs (t_w = {t_w})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def write_csv(records, csv_path):
    cols = ['name', 'label', 'n_nodes', 'n_edges',
            'in_mean', 'in_std', 'in_max', 'in_zero_ratio',
            'out_mean', 'out_std', 'out_max', 'out_zero_ratio']
    with open(csv_path, 'w', newline='') as f:
        w = csvlib.writer(f)
        w.writerow(cols)
        for r in sorted(records, key=lambda x: x['name']):
            w.writerow([f'{r[c]:.4f}' if isinstance(r[c], float) else r[c] for c in cols])


def run_one_window(t_w, args, labels, exceptions):
    data_dir = os.path.join(args.data_root, t_w)
    files = sorted(glob.glob(os.path.join(data_dir, '*_fulldata.json')))
    if not files:
        print(f'[{t_w}] グラフが見つかりません: {data_dir}')
        return False

    out_dir = args.out_dir or os.path.join(args.out_root, t_w)
    per_graph_dir = os.path.join(out_dir, 'per_graph')
    os.makedirs(per_graph_dir, exist_ok=True)

    jobs, skipped = [], []
    for path in files:
        file_name = os.path.basename(path)[:-5]
        base_name = nfc(file_name)[:-9]
        if (not args.no_exceptions and nfc(file_name) in exceptions) or base_name not in labels:
            skipped.append(file_name)
            continue
        jobs.append((path, int(labels[base_name]), per_graph_dir))

    t0 = time.time()
    records = []
    total = len(jobs)

    def report(i):
        # ログにリダイレクトすると \r が潰れないので、区切りのいい所だけ出す。
        if (i + 1) % max(1, total // 10) == 0 or i + 1 == total:
            print(f'[{t_w}] {(i + 1) / total * 100:5.1f}%  ({i + 1}/{total}) '
                  f'{time.time() - t0:.0f}s', flush=True)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, rec in enumerate(ex.map(process_file, jobs, chunksize=1)):
                records.append(rec)
                report(i)
    else:
        for i, job in enumerate(jobs):
            records.append(process_file(job))
            report(i)

    write_csv(records, os.path.join(out_dir, 'degree_summary.csv'))
    plot_aggregate(records, os.path.join(out_dir, 'degree_histogram_all_graphs.png'), t_w)

    print(f'[{t_w}] 完了 {len(records)}/{len(files)} グラフ '
          f'(スキップ {len(skipped)}) {time.time() - t0:.0f}s -> {out_dir}', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t-w', nargs='+', default=['1min'],
                    help='時間幅フォルダ名 (例: 1min 2min)。all で 1min 以外の全部')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--label-path', default=DEFAULT_LABEL_PATH)
    ap.add_argument('--out-dir', default=None,
                    help='単一の時間幅だけ処理する場合の出力先')
    ap.add_argument('--out-root', default=None,
                    help='出力ルート。実際の出力先は <out-root>/<t_w>/')
    ap.add_argument('--jobs', type=int, default=1, help='並列ワーカ数')
    ap.add_argument('--skip-existing', action='store_true',
                    help='degree_summary.csv が既にある時間幅を飛ばす')
    ap.add_argument('--no-exceptions', action='store_true',
                    help='除外リストを適用せず全グラフを描く')
    args = ap.parse_args()

    if not args.out_dir and not args.out_root:
        ap.error('--out-dir か --out-root のどちらかが必要です')

    tws = args.t_w
    if tws == ['all']:
        names = [d for d in os.listdir(args.data_root)
                 if re.fullmatch(r'\d+min', d) and d != '1min']
        tws = sorted(names, key=lambda s: int(s[:-3]))
    if args.out_dir and len(tws) > 1:
        ap.error('--out-dir は時間幅 1 個のときだけ使えます (複数なら --out-root)')

    with open(args.label_path) as f:
        raw_labels = json.load(f)
    labels = {nfc(k): v for k, v in raw_labels.items()}
    exceptions = {nfc(x) for x in EXCEPTIONS}

    print(f'対象 {len(tws)} 時間幅: {", ".join(tws)}')
    t0 = time.time()
    for t_w in tws:
        if args.skip_existing and args.out_root and os.path.exists(
                os.path.join(args.out_root, t_w, 'degree_summary.csv')):
            print(f'[{t_w}] 既存のためスキップ')
            continue
        run_one_window(t_w, args, labels, exceptions)
    print(f'全体 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
