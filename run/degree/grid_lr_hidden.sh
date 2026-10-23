#!/usr/bin/env bash
# 学習率 × 隠れ層次元の全組み合わせで二値分類を回す (コンテナ内で実行)
# DECODE 論文 p.9 / LEN 論文 p.7: h ∈ {128,256,512,1024}, l ∈ {0.001,0.0001,0.00001}
# usage: bash grid_lr_hidden.sh <出力ディレクトリ> [モデル ... (既定 GCN GAT SAGE)]
# train_twitter_MPNN.py は cwd/results/ に固定名で書くため、組み合わせごとに cwd を分ける
# GIN は predict() だけ global_add_pool で学習時 (mean) と食い違うため既定から外している
# テスト割合は環境変数 TEST_SIZE で指定 (既定 0.25 = LEN 論文の 75/25)。例: TEST_SIZE=0.20 bash grid_lr_hidden.sh ...
out_root=$(realpath -m "$1"); shift
test_size=${TEST_SIZE:-0.25}
models=${@:-GCN GAT SAGE}
script=$(cd "$(dirname "$0")/../.." && pwd)/train_twitter_MPNN.py
all_dir="/hss01/A.hattori/rww_all_graphs_revised/degree/mid"
rww="degree"
node_attr=1

for m in $models; do
  for lr in 0.001 0.0001 0.00001; do
    for hl in 128 256 512 1024; do
      d="$out_root/test${test_size}/${m}_lr${lr}_h${hl}"
      if [ -f "$d/results/${m}_${rww}_nodeattr${node_attr}_all_mv0.json" ]; then
        echo "skip $d"; continue
      fi
      mkdir -p "$d"
      echo "=== $m lr=$lr h=$hl start $(date '+%F %T') ==="
      (cd "$d" && python3 "$script" --model "$m" --data_type all --multivariate 0 \
        --hidden_dim "$hl" --lr "$lr" --output_dim 2 --all_graphs_path "$all_dir" \
        --rww_attr "$rww" --node_attr "$node_attr" --save_images 0 --test_size "$test_size" \
        --analysis_out "$d/analysis") > "$d/train.log" 2>&1
      echo "=== $m lr=$lr h=$hl exit=$? end $(date '+%F %T') ==="
    done
  done
done
