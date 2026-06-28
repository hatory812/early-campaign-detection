#!/bin/bash
#SBATCH --job-name=precomp_small
#SBATCH --mem=2G
#SBATCH --output=logs/precomp_small_%A_%a.out
#SBATCH --error=logs/precomp_small_%A_%a.err

# ティア1：t_w = 1〜20 分（1分刻み、20ジョブ）
# 投入コマンド: sbatch --array=0-19%10 precompute_small.sh

TW_LIST=($(seq 1 1 20))
TW=${TW_LIST[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

source /home/A.hattori/myenv/bin/activate

echo "START: t_w=${TW}min  $(date)"

python3 ../../precompute_minimized_embeddings.py \
    --src_path   /hss01/A.hattori/all_graphs \
    --label_path /hss01/A.hattori/all_graphs \
    --out_base   /hss01/A.hattori/rww_all_graphs_minimized \
    --pick degree \
    --comp mid \
    --tw_list "$TW" \
    --log_path   /hss01/A.hattori/log/precompute_small_tw${TW}.log

echo "DONE:  t_w=${TW}min  $(date)"
