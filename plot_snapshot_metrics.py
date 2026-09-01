"""時間幅ごとの集計 CSV (snapshot_{model}_{rww}_nodeattr{n}.csv) から
メトリクス vs t_w の図を再生成するスクリプト。

エラーバーの端は T 字 (capsize 付き) で描画する。

--baseline を付けると「全グラフを campaign(=1) と予測する」多数派クラス分類器の
スコアを重ねて描く。t_w ごとにデータセットの陽性率が変わる (短い時間幅ではエッジが
0 本になり出力されないグラフがあるため) ので、各 t_w のデータを実際に数えて求める。

使い方:
    python3 plot_snapshot_metrics.py <csv or dir> [<csv or dir> ...]
        [--metrics f1 accuracy] [--capsize 3] [--baseline]
"""

import argparse
import csv as csvlib
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# メトリクス名 -> グラフタイトルの表記
METRIC_TITLES = {
    'f1': 'F1 score',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
}

# ベースライン計算用。train_twitter_MPNN.py / precompute_minimized_embeddings.py と
# 同じラベル・除外リストを使う。
DEFAULT_LABEL_PATH = '/hss01/A.hattori/all_graphs/graph_labels.json'
DEFAULT_DATA_ROOT = '/hss01/A.hattori/rww_all_graphs_minimized'
EXCEPTIONS = {
    'graph_labels',
    'Gomis_noncampaign_fulldata',
    '#Hıdırellez_noncampaign_fulldata',
    '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata',
    'Haluk_noncampaign_fulldata',
    '#ErdenTimurSezonu_noncampaign_fulldata',
    'Gustavo_noncampaign_fulldata',
}


def stratified_test_counts(neg, pos, test_size=0.20):
    """sklearn の StratifiedShuffleSplit と同じテスト件数の配分を返す。

    n_test = ceil(test_size * N) を各クラスの比率で按分し、floor した残りを
    小数部の大きいクラスから 1 件ずつ配る。戻り値は (test_neg, test_pos, n_test)。
    """
    n = neg + pos
    n_test = math.ceil(test_size * n)
    raw = [n_test * neg / n, n_test * pos / n]
    base = [int(math.floor(r)) for r in raw]
    order = sorted(range(2), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:n_test - sum(base)]:
        base[i] += 1
    return base[0], base[1], n_test


def all_positive_baseline(t_w, label_path=DEFAULT_LABEL_PATH, data_root=DEFAULT_DATA_ROOT):
    """t_w のデータセットで「全部を campaign(=1) と予測」した場合のスコアを返す。

    accuracy = precision = テスト集合の陽性率 P, recall = 1.0, f1 = 2P/(1+P)。
    データが見つからなければ None。
    """
    data_dir = os.path.join(data_root, f'{t_w}min')
    if not os.path.isdir(data_dir):
        return None
    with open(label_path) as f:
        labels = json.load(f)

    pos = neg = 0
    for path in glob.glob(os.path.join(data_dir, '*_fulldata.json')):
        name = os.path.basename(path)[:-5]
        if name in EXCEPTIONS or name[:-9] not in labels:
            continue
        if labels[name[:-9]]:
            pos += 1
        else:
            neg += 1
    if pos + neg == 0:
        return None

    _, test_pos, n_test = stratified_test_counts(neg, pos)
    p = test_pos / n_test
    return {'accuracy': p, 'precision': p, 'recall': 1.0, 'f1': 2 * p / (1 + p)}


def baseline_series(tws, metric):
    """t_w のリストに対応するベースライン値のリスト (欠損は None)。"""
    return [(all_positive_baseline(int(t)) or {}).get(metric) for t in tws]


def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csvlib.DictReader(f))
    return rows


def plot_metric(rows, metric, out_path, label, capsize=3, baseline=False):
    tws, means, stds = [], [], []
    for r in rows:
        if r.get(f'{metric}_mean') in (None, ''):
            continue
        tws.append(float(r['t_w']))
        means.append(float(r[f'{metric}_mean']))
        stds.append(float(r.get(f'{metric}_std') or 0.0))

    plt.figure(figsize=(12, 7))
    plt.errorbar(tws, means, yerr=stds, fmt='o', color='C0',
                 ecolor='dimgray', elinewidth=1, capsize=capsize, capthick=1,
                 markersize=5, label=f'{metric}_mean ± std')

    if baseline:
        # 陽性率は t_w ごとに ±1 件ずれる階段状の値なので、線形補間せず step で描く。
        base = baseline_series(tws, metric)
        pts = [(t, b) for t, b in zip(tws, base) if b is not None]
        if pts:
            bt, bv = zip(*pts)
            # 凡例は ASCII のみ。既定フォント (DejaVu Sans) に日本語グリフが無く、
            # 日本語を入れると豆腐 (□) になるため。
            plt.step(bt, bv, where='mid', color='C3', linestyle='--', linewidth=1.8,
                     label='baseline: predict all as campaign')
        plt.legend(loc='best')

    plt.xlabel('t_w')
    plt.ylabel(f'{metric}_mean')
    title = METRIC_TITLES.get(metric, metric)
    plt.title(f'{title} vs t_w ({label})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f'Saved {out_path}')


def label_from_name(base):
    # snapshot_GCN_degree_nodeattr1 -> "GCN, degree, nodeattr1"
    m = re.match(r'snapshot_(.+?)_(.+?)_(nodeattr\d+)$', base)
    if m:
        return f'{m.group(1)}, {m.group(2)}, {m.group(3)}'
    return base


def process_csv(path, metrics, capsize, baseline=False):
    rows = load_csv(path)
    base = os.path.splitext(path)[0]
    label = label_from_name(os.path.basename(base))
    available = {c[:-5] for c in rows[0] if c.endswith('_mean')} if rows else set()
    targets = [m for m in metrics if m in available] if metrics else sorted(available)
    for metric in targets:
        plot_metric(rows, metric, f'{base}_{metric}.png', label, capsize, baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='集計 CSV かそれを含むディレクトリ')
    ap.add_argument('--metrics', nargs='*', default=None,
                    help='描画するメトリクス (既定: CSV 内の全 *_mean)')
    ap.add_argument('--capsize', type=float, default=3)
    ap.add_argument('--baseline', action='store_true',
                    help='全グラフを campaign(=1) と予測する多数派クラス分類器のスコアを重ねる')
    args = ap.parse_args()

    for p in args.paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith('.csv'):
                    process_csv(os.path.join(p, name), args.metrics, args.capsize,
                                args.baseline)
        else:
            process_csv(p, args.metrics, args.capsize, args.baseline)


if __name__ == '__main__':
    main()
