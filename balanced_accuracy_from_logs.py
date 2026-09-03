"""テスト時の predictions.log から時間幅ごとの Balanced Accuracy を算出し、
散布図 + エラーバーを描くスクリプト。

Balanced Accuracy = (TPR + TNR) / 2 で、クラス不均衡下でも
「全部 campaign と答えるだけ」の自明な分類器が必ず 0.5 になる指標。
既存の accuracy / f1 の図 (plot_snapshot_metrics.py) はこの性質を持たないため、
同じ試行の予測ログから後付けで算出する。

入力は
    <実験ディレクトリ>/<t_w>min/<run_name>/exp*/predictions.log
で、列は file_name / predicted / actual / correct / num_nodes / num_edges の TSV。
試行 (exp*) ごとに Balanced Accuracy を出し、その平均 ± 標準偏差 (np.std, ddof=0)
を t_w ごとの 1 点として描く。集計の丸めは train_twitter_MPNN.py に合わせて 3 桁。

使い方:
    python3 balanced_accuracy_from_logs.py results/<実験ディレクトリ>
        [--run_name GCN_degree_nodeattr1_all_mv0] [--out_dir <出力先>]
        [--capsize 3] [--no_baseline]
"""

import argparse
import csv as csvlib
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN_NAME = 'GCN_degree_nodeattr1_all_mv0'
# 出力ファイル名は既存の図 (snapshot_{model}_{rww}_nodeattr{n}_{metric}.png) に合わせる。
OUT_STEM = 'snapshot_GCN_degree_nodeattr1_balanced_accuracy'


def balanced_accuracy_from_log(path):
    """predictions.log 1 本から Balanced Accuracy を返す。

    片方のクラスがテスト分割に 1 件も無い場合は TPR/TNR の一方が未定義になるため
    None を返す (層化分割なので通常は起こらない)。
    """
    tp = tn = fp = fn = 0
    with open(path, encoding='utf-8') as f:
        reader = csvlib.DictReader(f, delimiter='\t')
        for row in reader:
            actual = int(row['actual'])
            pred = int(row['predicted'])
            if actual == 1:
                tp += (pred == 1)
                fn += (pred == 0)
            else:
                tn += (pred == 0)
                fp += (pred == 1)
    if (tp + fn) == 0 or (tn + fp) == 0:
        return None
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def collect(root, run_name):
    """t_w 昇順に (t_w, mean, std, n_exp, 試行ごとの値) を返す。"""
    rows = []
    for tw_dir in glob.glob(os.path.join(root, '*min')):
        m = re.match(r'^(\d+)min$', os.path.basename(tw_dir))
        if not m:
            continue
        t_w = int(m.group(1))
        logs = sorted(glob.glob(os.path.join(tw_dir, run_name, 'exp*', 'predictions.log')))
        vals = [v for v in (balanced_accuracy_from_log(p) for p in logs) if v is not None]
        if not vals:
            # CUDA OOM 等で学習自体が落ちた t_w はディレクトリが空になる。
            continue
        rows.append((t_w, float(np.mean(vals)), float(np.std(vals)), len(vals), vals))
    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(rows, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csvlib.writer(f)
        w.writerow(['t_w', 'balanced_accuracy_mean', 'balanced_accuracy_std', 'n_exp'])
        for t_w, mean, std, n, _ in rows:
            w.writerow([t_w, round(mean, 3), round(std, 3), n])
    print(f'Saved {out_path}')


def plot(rows, out_path, label, capsize=3, baseline=True):
    tws = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]

    plt.figure(figsize=(12, 7))
    plt.errorbar(tws, means, yerr=stds, fmt='o', color='C0',
                 ecolor='dimgray', elinewidth=1, capsize=capsize, capthick=1,
                 markersize=5, label='balanced_accuracy_mean ± std')

    if baseline:
        # 定数分類器 (全部 campaign / 全部 noncampaign) の Balanced Accuracy は
        # 陽性率によらず常に 0.5。t_w に依存しないので水平線で引く。
        # 凡例は ASCII のみ (既定フォントに日本語グリフが無く豆腐になるため)。
        plt.axhline(0.5, color='C3', linestyle='--', linewidth=1.8,
                    label='baseline: any constant predictor (= 0.5)')
    plt.legend(loc='best')

    plt.xlabel('t_w')
    plt.ylabel('balanced_accuracy_mean')
    plt.title(f'Balanced Accuracy vs t_w ({label})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f'Saved {out_path}')


def label_from_run_name(run_name):
    # GCN_degree_nodeattr1_all_mv0 -> "GCN, degree, nodeattr1"
    m = re.match(r'^(.+?)_(.+?)_(nodeattr\d+)_', run_name)
    if m:
        return f'{m.group(1)}, {m.group(2)}, {m.group(3)}'
    return run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='実験ディレクトリ (<t_w>min/ を含む階層)')
    ap.add_argument('--run_name', default=DEFAULT_RUN_NAME)
    ap.add_argument('--out_dir', default=None, help='既定は <root>/学習結果')
    ap.add_argument('--capsize', type=int, default=3)
    ap.add_argument('--no_baseline', action='store_true')
    args = ap.parse_args()

    rows = collect(args.root, args.run_name)
    if not rows:
        raise SystemExit(f'predictions.log が見つかりません: {args.root}/*min/{args.run_name}/exp*/')

    out_dir = args.out_dir or os.path.join(args.root, '学習結果')
    os.makedirs(out_dir, exist_ok=True)
    write_csv(rows, os.path.join(out_dir, OUT_STEM + '.csv'))
    plot(rows, os.path.join(out_dir, OUT_STEM + '.png'),
         label_from_run_name(args.run_name),
         capsize=args.capsize, baseline=not args.no_baseline)

    n_exps = {r[3] for r in rows}
    print(f'{len(rows)} 件の t_w を集計 (試行数: {sorted(n_exps)})')


if __name__ == '__main__':
    main()
