"""フルグラフ degree/mid 生成用に、担当分の graphId を大きい順に出力する。

usage: python3 make_full_mid_ids.py <part 0|1|all>   (0=gnn04, 1=gnn05, all=1 台で全件)
- 対象: graph_labels.json にあり、縮小グラフと同じ EXCEPTIONS (生の文字列比較) に当たらないグラフ
  → 縮小グラフ (308 件) と同じ集合
- 2 台への割り振り: 生データのサイズで大きい順に、合計が小さい方へ入れる (LPT)
- 保存済みのファイルは出力しない (再開時はそのまま再実行すればよい)
"""
import json, os, sys

REPO = '/home/A.hattori/ECMLPKDD25_run_revised'
SRC = '/hss01/A.hattori/all_graphs'
OUT = '/hss01/A.hattori/rww_all_graphs_revised/degree/mid'
EXCEPTIONS = [
    'graph_labels',
    'Gomis_noncampaign_fulldata',
    '#Hıdırellez_noncampaign_fulldata',
    '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata',
    'Haluk_noncampaign_fulldata',
    '#ErdenTimurSezonu_noncampaign_fulldata',
    'Gustavo_noncampaign_fulldata',
]  # precompute_minimized_embeddings.py:66-74 と同じ

part = sys.argv[1]
mapping = json.load(open(os.path.join(REPO, 'graph_name_mapping.json')))
labels = json.load(open(os.path.join(SRC, 'graph_labels.json')))
disk = set(os.listdir(SRC))

rows = []
for name, gid in mapping.items():
    fn = name + '_fulldata.json'
    if fn not in disk or name not in labels or fn[:-5] in EXCEPTIONS:
        continue
    rows.append((os.path.getsize(os.path.join(SRC, fn)), gid, fn))
rows.sort(reverse=True)

loads, parts = [0, 0], [[], []]
for size, gid, fn in rows:
    i = 0 if loads[0] <= loads[1] else 1
    loads[i] += size
    parts[i].append((gid, fn))

if len(sys.argv) > 2 and sys.argv[2] == '--summary':
    print(f'total={len(rows)} part0={len(parts[0])} ({loads[0]/2**30:.1f}GB) '
          f'part1={len(parts[1])} ({loads[1]/2**30:.1f}GB)')
    sys.exit(0)

targets = [(gid, fn) for _, gid, fn in rows] if part == 'all' else parts[int(part)]
for gid, fn in targets:
    if not os.path.exists(os.path.join(OUT, fn)):
        print(gid)
