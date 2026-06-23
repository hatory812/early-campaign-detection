all_dir="/hss01/A.hattori/rww_all_graphs/degree/mid"
data_type="all"
rww="degree"
node_attr=1

# 各 t_w における各 G_i の |V_ie| / |E_ie| とその統計レポートを出力 (訓練なし).
python3 ../../report_snapshot_stats.py --data_type $data_type \
  --all_graphs_path $all_dir --rww_attr $rww --node_attr $node_attr \
  --tw_min 1 --tw_max 60 --tw_step 1
