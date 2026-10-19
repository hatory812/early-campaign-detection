"""train_twitter_MPNN.load_data のキャッシュの検証スクリプト。

小規模データ (data_path) と使い捨てのキャッシュ先 (cache_dir) を指定して実行し、
最後に OK が出れば合格。data_path のファイルの mtime は一時的に書き換える
(終了時に戻す)ので、本番データには向けないこと。

  python3 check_dataset_cache.py --data_path <小規模データ> --cache_dir <使い捨てのキャッシュ先>
"""
import argparse
import glob
import os
import shutil

import torch

import train_twitter_MPNN as M


def load(data_path, cache_dir, node_attr=1):
    data_list, label_list = M.load_data(data_path, 'degree', node_attr, 0, cache_dir=cache_dir)
    return data_list, label_list, M.num_node_features, M.num_edge_features


def assert_same(a, b, msg):
    (dl_a, ll_a, nn_a, ne_a), (dl_b, ll_b, nn_b, ne_b) = a, b
    assert len(dl_a) == len(dl_b), f'{msg}: len'
    for p, q in zip(dl_a, dl_b):
        for k in ('x', 'edge_index', 'edge_attr', 'y'):
            assert torch.equal(getattr(p, k), getattr(q, k)), f'{msg}: {k} {p.name}'
        assert p.name == q.name, f'{msg}: name'
    assert ll_a == ll_b, f'{msg}: label_list'
    assert (nn_a, ne_a) == (nn_b, ne_b), f'{msg}: dims'


def cache_files(cache_dir):
    return sorted(glob.glob(os.path.join(cache_dir, 'train_twitter_MPNN', '*.pt')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--cache_dir', required=True)
    args = ap.parse_args()
    M.multivariate = 0
    M.classify_news = 0
    shutil.rmtree(args.cache_dir, ignore_errors=True)

    # 1. キャッシュあり/なしで一致 (1 回目=保存、2 回目=復元)
    ref = load(args.data_path, '')
    assert cache_files(args.cache_dir) == [], 'cache_dir="" なのにキャッシュができた'
    first = load(args.data_path, args.cache_dir)
    assert len(cache_files(args.cache_dir)) == 1
    M.num_node_features = M.num_edge_features = None  # 復元で戻ることの確認
    second = load(args.data_path, args.cache_dir)
    assert_same(ref, first, 'ref vs save')
    assert_same(ref, second, 'ref vs hit')
    print('1 OK: cache hit == no cache')

    # 2. mtime を変えると別キャッシュ
    f = sorted(glob.glob(os.path.join(args.data_path, '*_fulldata.json')))[0]
    st = os.stat(f)
    try:
        os.utime(f, (st.st_atime, st.st_mtime + 100))
        third = load(args.data_path, args.cache_dir)
    finally:
        os.utime(f, (st.st_atime, st.st_mtime))
    assert len(cache_files(args.cache_dir)) == 2, 'mtime 変更で別キャッシュにならない'
    assert_same(ref, third, 'mtime')
    print('2 OK: mtime change -> new cache')

    # 3. 壊れた .pt は作り直す
    for p in cache_files(args.cache_dir):
        with open(p, 'wb') as fh:
            fh.write(b'broken')
    fourth = load(args.data_path, args.cache_dir)
    assert_same(ref, fourth, 'broken')
    assert sum(os.path.getsize(p) > 100 for p in cache_files(args.cache_dir)) == 1, '壊れた .pt が作り直されていない'
    print('3 OK: broken .pt rebuilt')

    # 4. node_attr=0 は別キャッシュ
    n_before = len(cache_files(args.cache_dir))
    na0 = load(args.data_path, args.cache_dir, node_attr=0)
    assert len(cache_files(args.cache_dir)) == n_before + 1
    assert na0[2] != ref[2], 'node_attr=0 で次元が変わるはず'
    print('4 OK: node_attr=0 -> separate cache')
    print('OK')


if __name__ == '__main__':
    main()
