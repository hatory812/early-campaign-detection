"""pred_0 / pred_1 の次数分布図を、フォルダごとに 1 枚のコンタクトシートへ束ねる。

    <root>/<t_w>min/results/degree_distribution_<exp>/pred_<k>/*.png
        -> <root>/<t_w>min/results/degree_distribution_<exp>/summary/pred_<k>.png

計画は code_report_and_plan/copy_test_degree_histograms_plan.md を参照。

出力を summary/ サブフォルダに置くのは、
* 直下に置くと copy_test_degree_histograms.py の remove_flat_pngs() に消されるため
* pred_<k>/ の中に置くと「中身が predictions.log の file_name と 1 対 1」という性質が崩れるため。

タイルは縮小せず原寸 (1200x450) で並べる。元画像がタイトルにグラフ名・ラベル・N・E を、
各軸に目盛りラベルを持っており、縮小すると読めなくなるため。原寸でも 1 枚 1 MB 程度。

使い方:
    python3 make_degree_contact_sheets.py --root <実験ディレクトリ>
        [--exp exp2] [--cols 3] [--format png|pdf] [--dry_run]
"""

import argparse
import glob
import os
import re

import matplotlib
from PIL import Image, ImageDraw, ImageFont

# 見出し帯の高さ (px) と、そこに描く文字のサイズ。
HEADER_H = 60
FONT_SIZE = 32
FONT_PATH = os.path.join(os.path.dirname(matplotlib.__file__),
                         'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')


def load_font():
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        # フォントが見つからない環境では既定のビットマップフォントで代替する。
        return ImageFont.load_default()


def sheet_size(n_tiles, cols, tile_w, tile_h):
    rows = -(-n_tiles // cols)  # 切り上げ
    return cols * tile_w, HEADER_H + rows * tile_h, rows


def build_sheet(files, cols, title, font):
    """原寸のタイルを並べたコンタクトシートを返す。"""
    with Image.open(files[0]) as probe:
        tile_w, tile_h = probe.size
    width, height, _ = sheet_size(len(files), cols, tile_w, tile_h)

    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, (HEADER_H - FONT_SIZE) // 2), title, fill='black', font=font)

    for i, path in enumerate(files):
        with Image.open(path) as im:
            # 元画像は RGBA なので、白背景に載せてから貼る。
            sheet.paste(im.convert('RGB'),
                        ((i % cols) * tile_w, HEADER_H + (i // cols) * tile_h))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='<t_w>min/ を含む実験ディレクトリ')
    ap.add_argument('--exp', default='exp2')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--format', choices=['png', 'pdf'], default='png')
    ap.add_argument('--dry_run', action='store_true', help='作らずに枚数と画素数だけ出す')
    args = ap.parse_args()

    font = load_font()
    tws = sorted(int(m.group(1))
                 for m in (re.match(r'^(\d+)min$', os.path.basename(p))
                           for p in glob.glob(os.path.join(args.root, '*min')))
                 if m)

    made, empty, missing_tw = 0, [], []
    for t_w in tws:
        base = os.path.join(args.root, f'{t_w}min', 'results',
                            f'degree_distribution_{args.exp}')
        if not os.path.isdir(base):
            missing_tw.append(t_w)
            continue
        out_dir = os.path.join(base, 'summary')
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)

        parts = []
        for k in ('0', '1'):
            src_dir = os.path.join(base, f'pred_{k}')
            if not os.path.isdir(src_dir):
                continue
            # 実行のたびに同じ並びになるようファイル名昇順で固定する。
            files = sorted(glob.glob(os.path.join(src_dir, '*.png')))
            if not files:
                empty.append((t_w, k))
                continue

            title = f't_w={t_w}min  pred_{k}  ({len(files)} graphs)'
            out_path = os.path.join(out_dir, f'pred_{k}.{args.format}')
            if args.dry_run:
                with Image.open(files[0]) as probe:
                    tw_px, th_px = probe.size
                w, h, rows = sheet_size(len(files), args.cols, tw_px, th_px)
                parts.append(f'pred_{k}: {len(files):2d} 枚 -> {args.cols}x{rows} {w}x{h}px')
            else:
                sheet = build_sheet(files, args.cols, title, font)
                if args.format == 'pdf':
                    sheet.save(out_path, 'PDF', resolution=150.0)
                else:
                    sheet.save(out_path)
                parts.append(f'pred_{k}: {len(files):2d} 枚 -> {sheet.size[0]}x{sheet.size[1]}px '
                             f'{os.path.getsize(out_path) / 1e6:.1f}MB')
            made += 1

        if parts:
            print(f't_w={t_w:3d}  ' + '  |  '.join(parts))

    verb = '作成予定' if args.dry_run else '作成'
    print(f'\n--- 集計 ---')
    print(f'{verb}したシート: {made} 枚')
    if missing_tw:
        print(f'degree_distribution_{args.exp} が無い t_w ({len(missing_tw)}): {missing_tw}')
    if empty:
        print(f'画像が 0 枚だったフォルダ: {empty}')


if __name__ == '__main__':
    main()
