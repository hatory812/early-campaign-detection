#!/bin/bash
#SBATCH --job-name=precomp_medium
#SBATCH --mem=6G
#SBATCH --output=logs/precomp_medium_%A_%a.out
#SBATCH --error=logs/precomp_medium_%A_%a.err

# ティア2：t_w = 21〜60 分（1分刻み）＋ 120 分（計41ジョブ）
# 投入コマンド: sbatch --array=0-40%5 precompute_medium.sh

TW_LIST=($(seq 21 1 60) 120)
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
    --log_path   /hss01/A.hattori/log/precompute_medium_tw${TW}.log

echo "DONE:  t_w=${TW}min  $(date)"
