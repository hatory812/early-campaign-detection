#!/bin/bash
#SBATCH --job-name=precomp_large
#SBATCH --mem=16G
#SBATCH --output=logs/precomp_large_%A_%a.out
#SBATCH --error=logs/precomp_large_%A_%a.err

# ティア3：t_w = 180〜1440 分（60分刻み、22ジョブ）
# 投入コマンド: sbatch --array=0-21%3 precompute_large.sh

TW_LIST=($(seq 180 60 1440))
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
    --log_path   /hss01/A.hattori/log/precompute_large_tw${TW}.log

echo "DONE:  t_w=${TW}min  $(date)"
