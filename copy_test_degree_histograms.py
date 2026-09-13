"""ある試行 (exp*) のテスト対象グラフについて、次数分布ヒストグラム画像を集約する。

各時間幅 t_w について
    <pred_root>/<t_w>min/<run_name>/<exp>/predictions.log
からテストに用いたグラフのファイル名を取得し、対応する
    <degree_root>/<t_w>min/per_graph/<file_name>.png
を
    <pred_root>/<t_w>min/results/degree_distribution_<exp>/pred_<predicted>/
にコピーする。predicted は predictions.log のテスト時の予測値 (0 or 1) で、actual ではない。
予測の当たり外れと次数分布を同じ場所で突き合わせるのが目的。

注意: 学習側 (<t_w>min/<run_name>/<exp>/pred_0, pred_1) にも同名のフォルダがあるが、
あちらは train_twitter_MPNN.py が作るカスケードグラフ自体の可視化で、ファイル名も
連番プレフィックス付き・非 ASCII 置換済み。本スクリプトが作るのは次数分布ヒストグラムで、
ファイル名は predictions.log の file_name のまま (分類基準はどちらも predicted で同じ)。

計画は code_report_and_plan/[Plan]copy_test_degree_histograms_plan.md を参照。
実データで確認済みの前提が 2 つある:

* 学習が CUDA OOM で落ちた t_w はディレクトリが空で predictions.log が無い
  (20260805 の実験では 16/18/22/28/30/31/32/40 の 8 個)。エラーにせずスキップする。
* テスト対象のグラフには必ず per_graph の画像がある。欠落はすべて警告として報告する。
  欠落が出る典型的な原因は、自己ループ修正 (1ec2d56) より前のデータセットから作った
  古い次数統計を --degree_root に指定していること (警告の edges= が 0 ばかりなら、まずこれを疑う)。

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
    """predictions.log から [(file_name, num_edges, predicted), ...] を返す。"""
    with open(log_path, encoding='utf-8') as f:
        return [(r['file_name'], int(r['num_edges']), r['predicted'])
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


def remove_flat_pngs(out_dir, dry_run):
    """出力先の直下に残っている PNG を消す。

    平置きしていた頃の出力の後始末。pred_0/ pred_1/ の中身には触らない
    (再帰的には消さない)。消した枚数を返す。
    """
    if not os.path.isdir(out_dir):
        return 0
    stale = [n for n in os.listdir(out_dir)
             if n.endswith('.png') and os.path.isfile(os.path.join(out_dir, n))]
    if not dry_run:
        for name in stale:
            os.remove(os.path.join(out_dir, name))
    return len(stale)


def process_tw(t_w, pred_root, degree_root, run_name, exp, dry_run):
    """1 つの t_w を処理し、(予測値ごとのコピー数, 欠落, 削除数) を返す。"""
    log_path = os.path.join(pred_root, f'{t_w}min', run_name, exp, 'predictions.log')
    per_graph_dir = os.path.join(degree_root, f'{t_w}min', 'per_graph')
    out_dir = os.path.join(pred_root, f'{t_w}min', 'results', f'degree_distribution_{exp}')

    entries = read_test_entries(log_path)
    exact, normalized = build_png_index(per_graph_dir)

    removed = remove_flat_pngs(out_dir, dry_run)

    copied, missing = {}, []
    for file_name, num_edges, predicted in entries:
        png = f'{file_name}.png'
        src_name = exact.get(png) or normalized.get(nfc(png))
        if src_name is None:
            missing.append((file_name, num_edges))
            continue
        # テスト時の予測値でサブフォルダに振り分ける (actual ではなく predicted)。
        sub_dir = os.path.join(out_dir, f'pred_{predicted}')
        if not dry_run:
            os.makedirs(sub_dir, exist_ok=True)
            shutil.copy2(os.path.join(per_graph_dir, src_name),
                         os.path.join(sub_dir, src_name))
        copied[predicted] = copied.get(predicted, 0) + 1

    return copied, missing, removed


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

    per_class, skipped_tw, total_missing, total_removed = {}, [], [], 0
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

        copied, missing, removed = process_tw(
            t_w, args.pred_root, args.degree_root, args.run_name, args.exp, args.dry_run)
        for k, v in copied.items():
            per_class[k] = per_class.get(k, 0) + v
        total_missing += [(t_w, n, e) for n, e in missing]
        total_removed += removed

        note = ''
        if removed:
            note += f'  平置きの旧出力を削除 {removed}'
        if missing:
            note += f'  [WARN] 欠落 {len(missing)}: ' \
                    + ', '.join(f'{n}(edges={e})' for n, e in missing)
        breakdown = ' '.join(f'pred_{k}={copied.get(k, 0):2d}' for k in sorted(copied))
        print(f't_w={t_w:3d}  copied={sum(copied.values()):3d} ({breakdown}){note}')

    verb = 'コピー予定' if args.dry_run else 'コピー'
    print(f'\n--- 集計 ---')
    print(f'処理した t_w: {len(tws) - len(skipped_tw)} / {len(tws)}')
    if skipped_tw:
        print(f'スキップした t_w ({len(skipped_tw)}): {skipped_tw}')
    print(f'{verb}した画像: {sum(per_class.values())} 枚 '
          + '(' + ', '.join(f'pred_{k}={per_class[k]}' for k in sorted(per_class)) + ')')
    if total_removed:
        removed_verb = '削除予定' if args.dry_run else '削除'
        print(f'平置きの旧出力を{removed_verb}: {total_removed} 枚')
    if total_missing:
        print(f'[WARN] 欠落: {len(total_missing)} 件 -> {total_missing}')
    else:
        print('欠落: 0 件')


if __name__ == '__main__':
    main()
