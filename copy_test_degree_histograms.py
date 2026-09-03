"""ある試行 (exp*) のテスト対象グラフについて、次数分布ヒストグラム画像を集約する。

各時間幅 t_w について
    <pred_root>/<t_w>min/<run_name>/<exp>/predictions.log
からテストに用いたグラフのファイル名を取得し、対応する
    <degree_root>/<t_w>min/per_graph/<file_name>.png
を
    <pred_root>/<t_w>min/results/degree_distribution_<exp>/
にコピーする。予測の当たり外れと次数分布を同じ場所で突き合わせるのが目的。

計画は code_report_and_plan/copy_test_degree_histograms_plan.md を参照。
実データで確認済みの前提が 2 つある:

* 学習が CUDA OOM で落ちた t_w はディレクトリが空で predictions.log が無い
  (20260805 の実験では 16/18/22/28/30/31/32/40 の 8 個)。エラーにせずスキップする。
* エッジ 0 本のグラフには per_graph の画像が存在しない (plot_degree_histograms.py が
  図を出力しないため)。predictions.log の num_edges 列で照合し、0 なら正常な欠落、
  0 以外なら警告として報告する。

ファイル名の照合は素の一致を優先し、外れたときだけ NFC 正規化で再照合する
(本リポジトリはファイル名の NFD/NFC 混在が既知の問題のため)。

使い方:
    python3 copy_test_degree_histograms.py --pred_root <実験ディレクトリ> \
        --degree_root <次数分布ディレクトリ> [--exp exp2] [--dry_run]
"""

import argparse
import csv as csvlib
import glob
import os
import re
import shutil
import unicodedata

DEFAULT_RUN_NAME = 'GCN_degree_nodeattr1_all_mv0'


def nfc(s):
    return unicodedata.normalize('NFC', s)


def read_test_entries(log_path):
    """predictions.log から [(file_name, num_edges), ...] を返す。"""
    with open(log_path, encoding='utf-8') as f:
        return [(r['file_name'], int(r['num_edges']))
                for r in csvlib.DictReader(f, delimiter='\t')]


def build_png_index(per_graph_dir):
    """per_graph の PNG を「素の名前」「NFC 正規化した名前」の 2 通りで引けるようにする。"""
    exact, normalized = {}, {}
    for name in os.listdir(per_graph_dir):
        if not name.endswith('.png'):
            continue
        exact[name] = name
        normalized.setdefault(nfc(name), name)
    return exact, normalized


def process_tw(t_w, pred_root, degree_root, run_name, exp, dry_run):
    """1 つの t_w を処理し、(コピー数, 正常な欠落, 警告付き欠落) を返す。"""
    log_path = os.path.join(pred_root, f'{t_w}min', run_name, exp, 'predictions.log')
    per_graph_dir = os.path.join(degree_root, f'{t_w}min', 'per_graph')
    out_dir = os.path.join(pred_root, f'{t_w}min', 'results', f'degree_distribution_{exp}')

    entries = read_test_entries(log_path)
    exact, normalized = build_png_index(per_graph_dir)

    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)

    copied, empty_missing, unexpected = 0, [], []
    for file_name, num_edges in entries:
        png = f'{file_name}.png'
        src_name = exact.get(png) or normalized.get(nfc(png))
        if src_name is None:
            # エッジ 0 本なら図が作られないのが仕様。それ以外は要調査。
            (empty_missing if num_edges == 0 else unexpected).append((file_name, num_edges))
            continue
        if not dry_run:
            shutil.copy2(os.path.join(per_graph_dir, src_name),
                         os.path.join(out_dir, src_name))
        copied += 1

    return copied, empty_missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_root', required=True, help='predictions.log を含む実験ディレクトリ')
    ap.add_argument('--degree_root', required=True, help='per_graph を含む次数分布ディレクトリ')
    ap.add_argument('--exp', default='exp2')
    ap.add_argument('--run_name', default=DEFAULT_RUN_NAME)
    ap.add_argument('--dry_run', action='store_true', help='コピーせず件数と欠落だけ出す')
    args = ap.parse_args()

    tws = sorted(int(m.group(1))
                 for m in (re.match(r'^(\d+)min$', os.path.basename(p))
                           for p in glob.glob(os.path.join(args.pred_root, '*min')))
                 if m)

    total_copied, skipped_tw, total_empty, total_unexpected = 0, [], 0, []
    for t_w in tws:
        log_path = os.path.join(args.pred_root, f'{t_w}min', args.run_name,
                                args.exp, 'predictions.log')
        per_graph_dir = os.path.join(args.degree_root, f'{t_w}min', 'per_graph')
        if not os.path.exists(log_path):
            # 学習が落ちた t_w。エラーにせず記録だけ残す。
            skipped_tw.append(t_w)
            continue
        if not os.path.isdir(per_graph_dir):
            print(f'[WARN] t_w={t_w}: per_graph が無い ({per_graph_dir})')
            skipped_tw.append(t_w)
            continue

        copied, empty_missing, unexpected = process_tw(
            t_w, args.pred_root, args.degree_root, args.run_name, args.exp, args.dry_run)
        total_copied += copied
        total_empty += len(empty_missing)
        total_unexpected += [(t_w, n, e) for n, e in unexpected]

        note = ''
        if empty_missing:
            note += f'  欠落(エッジ0本) {len(empty_missing)}: ' \
                    + ', '.join(n for n, _ in empty_missing)
        if unexpected:
            note += f'  [WARN] 原因不明の欠落 {len(unexpected)}: ' \
                    + ', '.join(f'{n}(edges={e})' for n, e in unexpected)
        print(f't_w={t_w:3d}  copied={copied:3d}{note}')

    verb = 'コピー予定' if args.dry_run else 'コピー'
    print(f'\n--- 集計 ---')
    print(f'処理した t_w: {len(tws) - len(skipped_tw)} / {len(tws)}')
    if skipped_tw:
        print(f'スキップした t_w ({len(skipped_tw)}): {skipped_tw}')
    print(f'{verb}した画像: {total_copied} 枚')
    print(f'正常な欠落 (エッジ0本): {total_empty} 件')
    if total_unexpected:
        print(f'[WARN] 説明のつかない欠落: {len(total_unexpected)} 件 -> {total_unexpected}')
    else:
        print('説明のつかない欠落: 0 件')


if __name__ == '__main__':
    main()
