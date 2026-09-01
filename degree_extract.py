"""グラフ JSON からノード ID とエッジ端点だけを高速に取り出す。

rww_all_graphs_minimized の JSON は 1 ファイル最大 4GB あり (ほぼ全部が node_attr /
edge_attr の float 配列)、json.load すると数十 GB のメモリを食う。次数を数えるのに
必要なのは "id" と "source"/"target" だけなので、mmap + 正規表現で構造キーだけを
拾う。

正しさの根拠: JSON 文字列中の " は必ず \\" にエスケープされるため、エスケープされて
いない '"source": ' というバイト列は本物のキーとしてしか現れない (author.description
や edge.text にツイート本文が入っていても誤検出しない)。json.dump の既定セパレータ
', ' / ': ' に依存しているので、書き出し側が変わったら validate_against_json() で
再確認すること。
"""

import json
import mmap
import re

import numpy as np

# ノード dict の "id" と、エッジ dict の "source"/"target"。
RE_NODE_ID = re.compile(rb'"id": (\d+)')
RE_EDGE = re.compile(rb'"source": (\d+), "target": (\d+)')


def degrees_from_file(path):
    """(in_deg, out_deg, n_edges) を返す。次数の並びは JSON の nodes の出現順。"""
    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # nodes 配列は edges 配列より前にあるので、先に "edges" の位置を取って
            # 走査範囲を分ける。こうすると各正規表現が余計な領域を舐めずに済む。
            edges_at = mm.find(b'"edges": [')
            if edges_at < 0:
                raise ValueError(f'edges 配列が見つかりません: {path}')

            index = {int(m.group(1)): i
                     for i, m in enumerate(RE_NODE_ID.finditer(mm, 0, edges_at))}
            n = len(index)
            in_deg = np.zeros(n, dtype=np.int64)
            out_deg = np.zeros(n, dtype=np.int64)

            n_edges = 0
            seen = set()
            for m in RE_EDGE.finditer(mm, edges_at):
                s, t = int(m.group(1)), int(m.group(2))
                n_edges += 1
                # multigraph: false なので (s, t) の重複は 1 本として数える。
                if (s, t) in seen:
                    continue
                seen.add((s, t))
                out_deg[index[s]] += 1
                in_deg[index[t]] += 1

    return in_deg, out_deg, n_edges


def degrees_from_json(path):
    """json.load で同じものを求める参照実装 (検証用、小さいファイルにのみ使う)。"""
    with open(path) as f:
        data = json.load(f)
    index = {node['id']: i for i, node in enumerate(data['nodes'])}
    in_deg = np.zeros(len(index), dtype=np.int64)
    out_deg = np.zeros(len(index), dtype=np.int64)
    seen = set()
    for e in data['edges']:
        pair = (e['source'], e['target'])
        if pair in seen:
            continue
        seen.add(pair)
        out_deg[index[e['source']]] += 1
        in_deg[index[e['target']]] += 1
    return in_deg, out_deg, len(data['edges'])


def validate_against_json(paths, verbose=True):
    """各ファイルで高速版と json.load 版が完全一致することを確認する。"""
    ok = True
    for p in paths:
        fast = degrees_from_file(p)
        ref = degrees_from_json(p)
        same = (np.array_equal(fast[0], ref[0]) and np.array_equal(fast[1], ref[1])
                and fast[2] == ref[2])
        ok &= same
        if verbose:
            import os
            print(f'{"OK  " if same else "NG  "} N={len(ref[0]):5d} E={ref[2]:6d}  '
                  f'{os.path.basename(p)}')
    return ok


if __name__ == '__main__':
    import glob
    import sys

    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        '/hss01/A.hattori/rww_all_graphs_minimized/1min/*_fulldata.json'
    files = sorted(glob.glob(pattern))
    print(f'{len(files)} ファイルを検証')
    print('全一致' if validate_against_json(files, verbose=False) else '不一致あり')
