#!/usr/bin/env bash
# 再修正版コード (fix/bugfix-revision の f95702e を固定した worktree) で縮小グラフを作り直す
cd /home/A.hattori/ECMLPKDD25_run_revised || exit 1
echo "=== START $(date '+%F %T')  commit $(git rev-parse --short HEAD) ==="
python3 precompute_minimized_embeddings.py --test_tw 0
echo "=== t_w=0 EXIT=$? $(date '+%F %T') ==="
python3 precompute_minimized_embeddings.py
echo "=== MAIN EXIT=$? $(date '+%F %T') ==="
