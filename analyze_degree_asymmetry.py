"""「campaign は入次数が多く、noncampaign は出次数が多い」という仮説を時間幅ごとに検証する。

入力は results/20260804_グラフノードの入出次数の統計/<t_w>min/degree_summary.csv
(列: name, label, n_nodes, n_edges, in_mean, in_std, in_max, in_zero_ratio,
 out_mean, out_std, out_max, out_zero_ratio)。

注意: 有向グラフでは sum(in) == sum(out) == |E| なので **in_mean と out_mean は常に一致する**
(全 305 グラフで確認済み)。したがって「入次数が多い/出次数が多い」は平均では判定できず、
分布の形 (最大値・標準偏差・次数 0 の割合) で見るしかない。本スクリプトはその方針を取る。

各 t_w について
* クラス別に in_max / out_max の中央値
* 各特徴量単独の AUROC (campaign を陽性、0.5 = 無情報)
* 非対称性スコア asym = log((out_max + 1) / (in_max + 1)) の AUROC
を出し、CSV と図にする。

使い方:
    python3 analyze_degree_asymmetry.py [--src <次数分布ディレクトリ>]
        [--out_dir <出力先>] [--tw_max 60]
"""

import argparse
import csv as csvlib
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SRC = 'results/20260804_グラフノードの入出次数の統計'

# dataviz スキルの検証済みリファレンス配色をスロット順に使う (slot 1..4)。
# ホストに node が無く validate_palette.js を実行できないため、値は変更せずそのまま用いる。
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
C_GRID, C_REF = '#c8c7c0', '#52514e'


def auroc(scores, y):
    """campaign(=1) を陽性とした AUROC。同値は平均順位で扱う。"""
    s, y = np.asarray(scores, float), np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    order = s.argsort()
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = (s == v)
        ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def analyze_tw(path):
    rows = list(csvlib.DictReader(open(path, encoding='utf-8')))
    col = lambda k: np.array([float(r[k]) for r in rows])
    y = np.array([1 if r['label'] == '1' else 0 for r in rows])
    nV = col('n_nodes')

    feats = {
        'in_max': col('in_max'),
        'out_max': col('out_max'),
        'in_std': col('in_std'),
        'out_std': col('out_std'),
        'in_zero_ratio': col('in_zero_ratio'),
        'out_zero_ratio': col('out_zero_ratio'),
        'in_max_per_nV': col('in_max') / nV,
        'out_max_per_nV': col('out_max') / nV,
        # 大きいほど out 寄り。仮説が正しければ campaign 側で小さくなるはず。
        'asym': np.log((col('out_max') + 1) / (col('in_max') + 1)),
        'n_nodes': nV,
    }

    cam, non = y == 1, y == 0
    rec = {
        'n': len(rows), 'n_campaign': int(cam.sum()), 'n_noncampaign': int(non.sum()),
        'cam_in_max_med': np.median(col('in_max')[cam]),
        'cam_out_max_med': np.median(col('out_max')[cam]),
        'non_in_max_med': np.median(col('in_max')[non]),
        'non_out_max_med': np.median(col('out_max')[non]),
        # in より out が大きいグラフの割合 (仮説では noncampaign 側で高いはず)
        'cam_out_gt_in': float(np.mean(col('out_max')[cam] > col('in_max')[cam])),
        'non_out_gt_in': float(np.mean(col('out_max')[non] > col('in_max')[non])),
    }
    for name, v in feats.items():
        rec[f'auroc_{name}'] = auroc(v, y)
    return rec


def plot(tws, recs, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # 上段: クラス別・方向別の in_max/out_max 中央値。
    # 色 = 方向 (in/out)、線種 = クラス という複合エンコードにして色数を 2 に抑える。
    get = lambda k: [r[k] for r in recs]
    ax1.plot(tws, get('cam_in_max_med'), color=C_BLUE, lw=2, label='in-degree max, campaign')
    ax1.plot(tws, get('non_in_max_med'), color=C_BLUE, lw=2, ls='--',
             label='in-degree max, noncampaign')
    ax1.plot(tws, get('cam_out_max_med'), color=C_ORANGE, lw=2, label='out-degree max, campaign')
    ax1.plot(tws, get('non_out_max_med'), color=C_ORANGE, lw=2, ls='--',
             label='out-degree max, noncampaign')
    ax1.set_ylabel('median of per-graph max degree')
    ax1.set_title('Max in-degree exceeds max out-degree in BOTH classes')
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.grid(True, ls='--', alpha=0.3, color=C_GRID)

    # 下段: 各特徴量単独の AUROC。0.5 を基準線に置く。
    for key, color, label in [
        ('auroc_in_max', C_BLUE, 'in-degree max'),
        ('auroc_out_max', C_ORANGE, 'out-degree max'),
        ('auroc_in_std', C_AQUA, 'in-degree std'),
        ('auroc_out_std', C_YELLOW, 'out-degree std'),
    ]:
        ax2.plot(tws, get(key), color=color, lw=2, label=label)
    ax2.axhline(0.5, color=C_REF, ls=':', lw=1.5, label='no information (0.5)')
    ax2.set_xlabel('t_w')
    ax2.set_ylabel('AUROC (campaign = positive)')
    ax2.set_title('Above 0.5 means LARGER value indicates campaign')
    ax2.legend(loc='lower right', frameon=False, fontsize=9)
    ax2.grid(True, ls='--', alpha=0.3, color=C_GRID)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--out_dir', default='results/20260903_入出次数の非対称性による仮説検証')
    ap.add_argument('--tw_max', type=int, default=60)
    args = ap.parse_args()

    tws, recs = [], []
    for t_w in range(0, args.tw_max + 1):
        path = os.path.join(args.src, f'{t_w}min', 'degree_summary.csv')
        if not os.path.exists(path):
            continue
        tws.append(t_w)
        recs.append(analyze_tw(path))

    if not tws:
        raise SystemExit(f'degree_summary.csv が見つかりません: {args.src}')

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'degree_asymmetry_by_tw.csv')
    fields = ['t_w'] + list(recs[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csvlib.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t_w, rec in zip(tws, recs):
            w.writerow({'t_w': t_w, **{k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in rec.items()}})
    print(f'Saved {csv_path}')

    plot(tws, recs, os.path.join(args.out_dir, 'degree_asymmetry_by_tw.png'))

    # 標準出力に結論を出す。
    print(f'\n--- 全 {len(tws)} 時間幅の平均 AUROC (campaign = 陽性) ---')
    for k in [k for k in recs[0] if k.startswith('auroc_')]:
        print(f'  {k[6:]:<16}{np.nanmean([r[k] for r in recs]):.3f}')
    print('\n"in が大きいほど campaign" が正しければ auroc_in_max > 0.5、')
    print('"out が大きいほど noncampaign" が正しければ auroc_out_max < 0.5 になるはず。')


if __name__ == '__main__':
    main()
