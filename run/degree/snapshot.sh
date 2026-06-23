all_dir="/hss01/A.hattori/rww_all_graphs/degree/mid"
data_type="all"
output_dim=2
mv=0
rww="degree"

# GCN: snapshot window t_w = 1..60 minutes
m1="GCN"
gcn_lr=0.0001
gcn_hl=1024
node_attr_gcn=1
python3 ../../train_twitter_snapshot.py --model $m1 --data_type $data_type --multivariate $mv \
  --hidden_dim $gcn_hl --lr $gcn_lr --output_dim $output_dim \
  --all_graphs_path $all_dir --rww_attr $rww --node_attr $node_attr_gcn \
  --tw_min 1 --tw_max 60 --tw_step 1
