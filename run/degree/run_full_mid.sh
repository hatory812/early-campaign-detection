#!/usr/bin/env bash
# フルグラフの degree/mid 埋め込みを作る (コンテナ内で実行)
# usage: bash run_full_mid.sh <part 0|1|all> [並列数 (既定 4)]   0=gnn04, 1=gnn05, all=1 台で全件
part=$1; P=${2:-4}
HERE=$(cd "$(dirname "$0")" && pwd)       # このスクリプトと make_full_mid_ids.py の置き場
D=/hss01/A.hattori/log_revised/full_mid    # ログだけを置く
mkdir -p "$D"
cd /home/A.hattori/ECMLPKDD25_run_revised/run/degree || exit 1
echo "=== START part=$part P=$P $(hostname) $(date '+%F %T') commit $(git rev-parse --short HEAD) ==="
python3 "$HERE/make_full_mid_ids.py" $part | xargs -P "$P" -I{} sh -c '
  s=$(date "+%F %T")
  python3 ../../kcore_rww.py --path /hss01/A.hattori/all_graphs \
    --out_root /hss01/A.hattori/rww_all_graphs_revised \
    --graphId {} --pick degree --comp mid > '"$D"'/id_{}.log 2>&1
  echo "id={} exit=$? start=$s end=$(date "+%F %T")"'
echo "=== END part=$part $(date '+%F %T') ==="
