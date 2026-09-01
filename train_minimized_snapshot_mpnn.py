"""
train_minimized_snapshot_mpnn.py

/hss01/A.hattori/rww_all_graphs_minimized/{t_w}min/ (precompute_minimized_embeddings.py が
生成した、各時間幅 t_w でエッジ・ノードを縮小しノード特徴量(degree_embedding)を再計算済みの
データセット, t_w=1..60) を train_twitter_MPNN.py に t_w ごとに読み込ませて学習・評価する.

設計方針:
  - train_twitter_MPNN.py は一切改変しない。t_w ごとに別プロセス (subprocess) として起動する。
  - train_twitter_MPNN.py は結果を相対パス
      results/{model}_{rww_attr}_nodeattr{node_attr}_{data_type}_mv{multivariate}.json
    に保存する仕様のため (ファイル名に t_w を含まない)、t_w ごとに作業ディレクトリ (cwd) を
    分離して起動することで、コード変更なしに出力の上書き事故を防ぐ。
  - 並列化は multiprocessing.Pool (spawn) で行う。ワーカー自身は torch/CUDA に触れず、
    subprocess の起動・待機・ログ記録のみを行う。
  - 冪等性: 既に結果 JSON が存在する t_w はスキップするので、中断後の再実行に強い。

★ 実行前に必ず確認すること
  (詳細: code_report_and_plan/minimized_snapshot_mpnn_sweep_plan.md,
         code_report_and_plan/snapshot_oom_investigation.md)
  - ホストの空きメモリ (`free -h` の available)。過去に train_twitter_snapshot.py で
    ホスト全体の OOM killer が発火し、同一セッションスコープの tmux サーバごと
    巻き添えで kill された実績がある。
  - rww_all_graphs_minimized/{t_w}min のディレクトリサイズは t_w にほぼ比例して増加する
    (実測: 1min=77MB, 10min=620MB, 20min=1.3GB, 30min=1.9GB, 40min=2.4GB, 50min=3.0GB,
     60min=3.5GB)。t_w が大きいワーカーほどメモリ消費が大きい。
  - GPU (nvidia-smi 上 NVIDIA GB10) は memory.total/memory.used が N/A であり、
    統合メモリ方式 (CPU/GPUでメモリプールを共有) とみられる。ディスクリートGPUを前提にした
    「GPUメモリとホストメモリは別物」という想定はできない可能性が高い。
    不安な場合は --force_cpu を使うこと。

既定値は保守的に num_workers=1 (逐次実行) にしてある。並列度を上げる場合は、
必ず --test_tw で少数の t_w を試し、`watch -n 5 free -h` 等でメモリを監視しながら
安全な --num_workers を実測してから本実行すること。

実行例（本番: 全60点を既定の逐次実行で処理、途中結果はスキップして再開可能）:
  python3 train_minimized_snapshot_mpnn.py

実行例（試験: t_w=1,30,60 のコマンド組み立てだけ確認）:
  python3 train_minimized_snapshot_mpnn.py --test_tw 1,30,60 --dry_run

実行例（試験: t_w=1,30,60 を実際に Pool(1) で逐次実行）:
  python3 train_minimized_snapshot_mpnn.py --test_tw 1,30,60

実行例（並列度を上げる場合。事前にメモリを監視しながら安全な値を確認すること）:
  python3 train_minimized_snapshot_mpnn.py --num_workers 3
"""

import os
import sys
import argparse
import logging
import subprocess
import multiprocessing as mp
from datetime import datetime

MPNN_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_twitter_MPNN.py'))


def setup_main_logger(log_path):
    """メインプロセス用のロガー (root) を stdout + ファイルに設定して返す。
    process_tw は root を触らない get_worker_logger を使うため、この設定は
    逐次モードでもワーカーに上書きされない。"""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def get_worker_logger(log_path):
    """ワーカー (t_w) 用のロガーを返す。root ロガーには触れず、その t_w 専用の
    ファイルハンドラだけを持つ名前付きロガーを使う。これにより逐次モードで
    process_tw を呼んでもメインロガー (root) を壊さない。spawn 子プロセスでも
    各自クリーンに動く。"""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger(f'worker.{log_path}')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # root に伝播させない (main.log/stdout を汚さない)
    if not logger.handlers:    # 同一パスで二重登録しない
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(fh)
    return logger


def get_available_memory_gb():
    """/proc/meminfo から MemAvailable を読み取る (GiB)。取得できなければ None。"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return None


def build_command(src_dir, run_dir, args):
    """train_twitter_MPNN.py 呼び出しコマンドを組み立てる。

    --analysis_out を run_dir ({out_root}/{t_w}min) に固定して渡すことで、
    予測ログ・可視化を t_w ごとに分離する (train_twitter_MPNN.py の既定は
    t_w 非依存の固定パスのため、スイープ時は t_w 間で上書きされてしまう)。
    """
    return [
        sys.executable, MPNN_SCRIPT,
        '--model', args.model,
        '--lr', str(args.lr),
        '--hidden_dim', str(args.hidden_dim),
        '--output_dim', str(args.output_dim),
        '--data_type', 'all',
        '--all_graphs_path', src_dir,
        '--multivariate', str(args.multivariate),
        '--classify_news', str(args.classify_news),
        '--rww_attr', args.rww_attr,
        '--node_attr', str(args.node_attr),
        '--batch_size', str(args.batch_size),
        '--analysis_out', run_dir,
    ]


def expected_output_path(run_dir, args):
    """train_twitter_MPNN.py が書き出す結果JSONの絶対パス (cwd=run_dir 前提)。"""
    fname = f"{args.model}_{args.rww_attr}_nodeattr{args.node_attr}_all_mv{args.multivariate}.json"
    return os.path.join(run_dir, 'results', fname)


def tail_lines(path, n=20):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return '(ログ読込失敗)'


def process_tw(task):
    """Pool のワーカー関数。1つの t_w に対して train_twitter_MPNN.py を1回起動する。"""
    t_w, args = task

    logger = get_worker_logger(os.path.join(args.log_dir, f'tw{t_w}.log'))

    src_dir = os.path.join(args.src_root, f'{t_w}min')
    if not os.path.isdir(src_dir):
        logger.warning(f"SKIP t_w={t_w}: source dir not found: {src_dir}")
        return (t_w, 'skip_missing_src')

    run_dir = os.path.join(args.out_root, f'{t_w}min')
    os.makedirs(run_dir, exist_ok=True)

    out_path = expected_output_path(run_dir, args)
    if os.path.exists(out_path):
        logger.info(f"SKIP t_w={t_w}: result already exists: {out_path}")
        return (t_w, 'skip_existing')

    cmd = build_command(src_dir, run_dir, args)

    env = os.environ.copy()
    if args.force_cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''

    logger.info(f"=== START t_w={t_w}min  cwd={run_dir}  {datetime.now()} ===")
    logger.info(f"cmd: {' '.join(cmd)}")

    if args.dry_run:
        logger.info("[DRY-RUN] 実行はスキップしました。")
        return (t_w, 'dry_run')

    subproc_log = os.path.join(args.log_dir, f'tw{t_w}_subprocess.log')
    with open(subproc_log, 'w', encoding='utf-8') as logfh:
        result = subprocess.run(cmd, cwd=run_dir, env=env, stdout=logfh, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        logger.info(f"=== DONE  t_w={t_w}min  {datetime.now()} ===")
        return (t_w, 'ok')
    else:
        logger.error(
            f"=== FAILED t_w={t_w}min  returncode={result.returncode}  {datetime.now()} ===\n"
            f"--- {subproc_log} tail ---\n{tail_lines(subproc_log)}")
        return (t_w, 'failed')


def main():
    parser = argparse.ArgumentParser(
        description="rww_all_graphs_minimized/{t_w}min を train_twitter_MPNN.py に順次/並列投入する.")
    parser.add_argument('--src_root', default='/hss01/A.hattori/rww_all_graphs_minimized',
                        help='t_w ディレクトリの親')
    parser.add_argument('--out_root',
                        default='/home/A.hattori/ECMLPKDD25/results/20260706_時間幅1-60で埋め込み再計算した実験_OutOfMemory回避',
                        help='t_wごとの作業ディレクトリの親。実結果は {out_root}/{t_w}min/results/ に生成される')
    parser.add_argument('--tw_min', type=int, default=1)
    parser.add_argument('--tw_max', type=int, default=60)
    parser.add_argument('--tw_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='multiprocessing.Pool のプロセス数。ホストメモリが逼迫しやすいため既定は1(逐次)。'
                             '上げる場合は --test_tw で少数を試してメモリを監視してから決めること。')
    parser.add_argument('--model', default='GCN')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--output_dim', default=2, type=int)
    parser.add_argument('--multivariate', default=0, type=int)
    parser.add_argument('--classify_news', default=0, type=int)
    parser.add_argument('--rww_attr', default='degree',
                        help='この縮小データセットは degree_embedding のみ持つため、degree 以外を'
                             '指定すると train_twitter_MPNN.py 側で KeyError になる。')
    parser.add_argument('--node_attr', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--force_cpu', action='store_true',
                        help='子プロセスの CUDA_VISIBLE_DEVICES を空にしてCPU実行に固定する。')
    parser.add_argument('--test_tw', default=None,
                        help="テスト用: 処理するt_wをカンマ区切りで指定 (e.g. '1,30,60')。"
                             "省略時は tw_min..tw_max の全域を処理する。")
    parser.add_argument('--dry_run', action='store_true',
                        help='実際には実行せず、組み立てたコマンドとcwdだけをログに出す。')
    parser.add_argument('--log_dir', default=None)
    args = parser.parse_args()

    if args.rww_attr != 'degree':
        print(f"WARNING: --rww_attr={args.rww_attr} は縮小データセットに存在しない可能性が高い"
              f"(precompute_minimized_embeddings.py は degree のみ生成)。続行しますが"
              f"train_twitter_MPNN.py 側で失敗する見込みです。")

    args.out_root = os.path.abspath(args.out_root)
    args.log_dir = args.log_dir or os.path.join(args.out_root, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)

    main_logger = setup_main_logger(os.path.join(args.log_dir, 'main.log'))
    main_logger.info(f"=== train_minimized_snapshot_mpnn.py START {datetime.now()} ===")
    main_logger.info(f"args: {vars(args)}")

    mem_gb = get_available_memory_gb()
    if mem_gb is not None:
        main_logger.info(f"現在のホスト空きメモリ (MemAvailable): {mem_gb:.1f} GiB")
        if mem_gb < 20:
            main_logger.warning(
                f"空きメモリが {mem_gb:.1f} GiB と少ないです。過去に train_twitter_snapshot.py で"
                f"ホスト全体のOOM killerが発火しtmuxごと巻き添えで落ちた実績があります"
                f"(詳細: code_report_and_plan/snapshot_oom_investigation.md)。"
                f"num_workers={args.num_workers} で実行しますが、"
                f"まず --test_tw と --dry_run で小規模に確認することを強く推奨します。")

    if args.test_tw:
        tw_values = [int(x.strip()) for x in args.test_tw.split(',')]
        main_logger.info(f"TEST MODE: t_w={tw_values}")
    else:
        tw_values = list(range(args.tw_min, args.tw_max + 1, args.tw_step))

    tasks = [(t_w, args) for t_w in tw_values]

    if args.num_workers <= 1:
        main_logger.info(f"逐次実行 (num_workers=1) で {len(tasks)} 件処理します。")
        results = [process_tw(task) for task in tasks]
    else:
        main_logger.info(f"並列実行 (num_workers={args.num_workers}) で {len(tasks)} 件処理します。")
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(process_tw, tasks)

    counts = {}
    for _, status in results:
        counts[status] = counts.get(status, 0) + 1
    main_logger.info(f"=== ALL DONE {datetime.now()} === summary: {counts}")
    for t_w, status in results:
        if status == 'failed':
            main_logger.warning(f"  t_w={t_w}: {status} (詳細は tw{t_w}.log / tw{t_w}_subprocess.log を参照)")


if __name__ == '__main__':
    main()
