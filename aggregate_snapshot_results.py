"""train_minimized_snapshot_mpnn.py のスイープ結果 (t_w ごとの JSON) を
1 つの集計 CSV にまとめるスクリプト。

入力: {out_root}/{t_w}min/results/{model}_{rww_attr}_nodeattr{node_attr}_{data_type}_mv{mv}.json
      (train_twitter_MPNN.py が書き出す形式。'metric_names' / 'mean' / 'std' を持つ)
出力: {out_root}/{out_subdir}/snapshot_{model}_{rww_attr}_nodeattr{node_attr}.csv
      (列: t_w, {metric}_mean, {metric}_std, ...。plot_snapshot_metrics.py の入力形式)

結果 JSON が無い t_w (学習が失敗した t_w など) は行をスキップし、標準出力に一覧を出す。

使い方:
    python3 aggregate_snapshot_results.py <out_root> [--tw_min 0] [--tw_max 60]
"""

import argparse
import csv
import json
import os


def collect(out_root, tw_values, fname):
    """t_w ごとの結果 JSON を読み、(rows, metric_names, missing) を返す。"""
    rows, metric_names, missing = [], None, []
    for t_w in tw_values:
        path = os.path.join(out_root, f'{t_w}min', 'results', fname)
        if not os.path.exists(path):
            missing.append(t_w)
            continue
        with open(path) as f:
            rec = json.load(f)
        if metric_names is None:
            metric_names = rec['metric_names']
        row = {'t_w': t_w}
        for m in metric_names:
            row[f'{m}_mean'] = rec['mean'][m]
            row[f'{m}_std'] = rec['std'][m]
        rows.append(row)
    return rows, metric_names, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_root', help='スイープの出力ルート ({t_w}min/ を含むディレクトリ)')
    ap.add_argument('--tw_min', type=int, default=0)
    ap.add_argument('--tw_max', type=int, default=60)
    ap.add_argument('--tw_step', type=int, default=1)
    ap.add_argument('--model', default='GCN')
    ap.add_argument('--rww_attr', default='degree')
    ap.add_argument('--node_attr', type=int, default=1)
    ap.add_argument('--data_type', default='all')
    ap.add_argument('--multivariate', type=int, default=0)
    ap.add_argument('--out_subdir', default='学習結果',
                    help='CSV の出力先サブディレクトリ (out_root からの相対)')
    args = ap.parse_args()

    fname = (f'{args.model}_{args.rww_attr}_nodeattr{args.node_attr}'
             f'_{args.data_type}_mv{args.multivariate}.json')
    tw_values = list(range(args.tw_min, args.tw_max + 1, args.tw_step))
    rows, metric_names, missing = collect(args.out_root, tw_values, fname)

    if not rows:
        raise SystemExit(f'結果 JSON が 1 つも見つかりません: {args.out_root}/*min/results/{fname}')

    out_dir = os.path.join(args.out_root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f'snapshot_{args.model}_{args.rww_attr}_nodeattr{args.node_attr}.csv')

    header = ['t_w']
    for m in metric_names:
        header += [f'{m}_mean', f'{m}_std']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {out_path} ({len(rows)} rows)')
    if missing:
        print(f'結果 JSON が無い t_w ({len(missing)} 件, 行をスキップ): '
              + ', '.join(str(t) for t in missing))


if __name__ == '__main__':
    main()
