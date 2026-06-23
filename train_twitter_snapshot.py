"""
スナップショット縮小化による精度変化の実験.

各グラフ G_i を「拡散開始 (= 最小タイムスタンプ t_0) から t_w 分以内」の
エッジ・ノードだけに縮小し, その縮小データセット G' でモデルを訓練して
Accuracy / Precision / Recall / F1 を測る. t_w を 1..60 分まで掃引し,
精度が t_w に対してどう変化するかの曲線を得る.

設計上のポイント:
  - JSON のパースは一度だけ行い, 軽量な numpy キャッシュをメモリに保持する.
    各 t_w ではキャッシュをタイムスタンプでマスクするだけ (埋め込みの再計算なし).
  - ノード/エッジの特徴量はフルグラフ由来の事前計算値を流用し, 生き残った
    ノード・エッジだけを抜き出す (トポロジ縮小の純粋効果を見る設定).
  - data_list の順序を t_w 間で固定し, split_data(seed=exp) の stratify +
    固定 random_state により, 同じグラフが t_w 間で一貫して train/test に
    割り当てられるようにする (縮小度合いだけが変数になる).

train_twitter_MPNN.py を土台にしており, 学習/推論/評価のロジックは流用している.
"""

import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import json
import argparse
import glob
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv, SAGEConv
from torch.nn import Linear, Sequential
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.optim as optim
import random
import torch.nn as nn
import os
import warnings
from models import GCN, GCN_edge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

num_node_features = None
num_edge_features = None
num_classes = None
lr = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = None
multivariate = None
classify_news = None
model_name = None
hidden_channels = None


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _node_feature(node, rww_attr, node_attr):
    """process_data と同じ分岐でノード特徴ベクトル (list) を返す."""
    if rww_attr == 'kcore' and node_attr == 1:
        return node['node_attr'] + node['structural_embedding']
    elif rww_attr == 'kcore' and node_attr == 0:
        return node['structural_embedding']
    elif rww_attr == 'degree' and node_attr == 1:
        return node['node_attr'] + node['degree_embedding']
    elif rww_attr == 'degree' and node_attr == 0:
        return node['degree_embedding']
    elif rww_attr == 'ktruss' and node_attr == 1:
        return node['node_attr'] + node['truss_embedding']
    elif rww_attr == 'ktruss' and node_attr == 0:
        return node['truss_embedding']
    else:
        return node['node_attr']


def build_cache(data_path, rww_attr, node_attr):
    """312 個の JSON を一度だけパースし, t_w フィルタ用の軽量キャッシュを作る.

    各グラフにつき以下を保持する (dict):
        src, dst : エッジ端点の位置インデックス (int32 numpy)
        ts       : エッジのタイムスタンプ (int64 numpy, Unix 秒)
        t0       : ts の最小値 (= 拡散開始時刻)
        edge_attr: [E, F_e] (float32 numpy)
        node_feat: [N, F_n] (float32 numpy, ノード位置順)
        label    : int
        name     : str
    """
    global num_node_features, num_edge_features
    print("Building cache (JSON parsed once).....")

    path = data_path
    label_path = "/hss01/A.hattori/all_graphs"
    files = list(glob.glob(path + '/*_fulldata.json'))

    if classify_news:
        with open(label_path + "/graph_labels_news.json", "r") as f:
            graph_labels = json.load(f)
    else:
        with open(label_path + "/graph_labels.json", "r") as f:
            graph_labels = json.load(f)

    exceptions = ['graph_labels', "Gomis_noncampaign_fulldata", "#Hıdırellez_noncampaign_fulldata",
                  '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata', 'Haluk_noncampaign_fulldata',
                  '#ErdenTimurSezonu_noncampaign_fulldata', 'Gustavo_noncampaign_fulldata']

    cache = []
    total = len(files)
    for i, file in enumerate(files):
        file_name = file.split('/')[-1][:-5]
        print(f"\r{(i + 1) / total * 100:.1f}% {file_name}", end='', flush=True)
        if file_name in exceptions or (file_name[:-9] not in graph_labels):
            continue
        graph_label = graph_labels[file_name[:-9]]

        with open(file, 'r') as f:
            data = json.load(f)

        # ノード: JSON の出現順に 0..N-1 の位置へマップ (process_data の relabel と同等).
        nodemap = {}
        node_feat = []
        for node in data['nodes']:
            nodemap[node['id']] = len(node_feat)
            node_feat.append(_node_feature(node, rww_attr, node_attr))
        node_feat = np.asarray(node_feat, dtype=np.float32)

        edges = data['edges']
        src = np.empty(len(edges), dtype=np.int32)
        dst = np.empty(len(edges), dtype=np.int32)
        ts = np.empty(len(edges), dtype=np.int64)
        edge_attr = np.empty((len(edges), len(edges[0]['edge_attr'])), dtype=np.float32)
        for j, e in enumerate(edges):
            src[j] = nodemap[e['source']]
            dst[j] = nodemap[e['target']]
            ts[j] = e['timestamp']
            edge_attr[j] = e['edge_attr']

        if num_node_features is None:
            num_node_features = node_feat.shape[1]
            num_edge_features = edge_attr.shape[1]

        cache.append({
            'src': src, 'dst': dst, 'ts': ts, 't0': int(ts.min()),
            'edge_attr': edge_attr, 'node_feat': node_feat,
            'label': int(graph_label), 'name': file_name,
        })

    print()
    return cache


def build_snapshot(c, t_w):
    """キャッシュ c を t_w 分でフィルタした縮小グラフ G_i' を PyG Data として返す.

    タイムスタンプが t_0 + t_w*60 (秒) 以内のエッジ E_ie とその端点 V_ie だけを残す.
    ts == t0 のエッジは必ず残るので, 縮小グラフは最低 1 エッジを持つ.
    """
    from torch_geometric.data import Data

    threshold = c['t0'] + t_w * 60
    mask = c['ts'] <= threshold
    src = c['src'][mask]
    dst = c['dst'][mask]

    # 生き残ったノード (両端ノード) を 0..k-1 へ再ラベル.
    surviving = np.unique(np.concatenate([src, dst]))
    remap = np.empty(c['node_feat'].shape[0], dtype=np.int64)
    remap[surviving] = np.arange(surviving.shape[0])

    x = torch.from_numpy(c['node_feat'][surviving])
    edge_index = torch.from_numpy(np.stack([remap[src], remap[dst]])).long()
    edge_attr = torch.from_numpy(c['edge_attr'][mask])
    y = torch.tensor([c['label']]).view(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.edge_attr = edge_attr
    data.name = c['name']
    return data


def split_data(data_list, label_list, seed=None):
    campaign_news_graphs = []
    noncampaign_news_graphs = []

    if classify_news:
        for data in data_list:
            if data.y.item():
                campaign_news_graphs.append(data)
            else:
                noncampaign_news_graphs.append(data)
        data_list = campaign_news_graphs + random.sample(noncampaign_news_graphs, len(campaign_news_graphs))
        label_list = [1] * len(campaign_news_graphs) + [0] * len(campaign_news_graphs)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data_list, label_list, stratify=label_list,
        test_size=0.20, shuffle=True, random_state=seed)
    random.shuffle(train_data)

    return train_data, test_data, None


def predict(model, test_data, args):
    model.eval()
    y_pred = []
    y_actual = []
    y_scores = []
    loader = DataLoader(test_data, batch_size=int(args.batch_size), shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if args.model == "GINE":
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                pred = model(batch.x, batch.edge_index)
            if args.model == 'GIN':
                pooled_output = global_add_pool(pred, batch.batch)
            else:
                pooled_output = global_mean_pool(pred, batch.batch)
            pred = model.out(pooled_output)
            pred = F.softmax(pred, dim=1)
            labels = batch.y
            _, predictions = torch.max(pred, 1)
            y_pred += predictions.tolist()
            y_actual += labels.tolist()
            y_scores += pred[:, 1].tolist()
    return np.array(y_pred), np.array(y_actual), np.array(y_scores)


def train_model(model, epochs, train_data, args):
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_data, batch_size=int(args.batch_size), shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            if args.model == "GINE":
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                pred = model(batch.x, batch.edge_index)
            pooled_output = global_mean_pool(pred, batch.batch)
            pred = model.out(pooled_output)
            if multivariate:
                label = batch.y.to(device)
            else:
                label = F.one_hot(batch.y, num_classes=2).float().to(device)
            criterion.to(device)
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            opt.zero_grad()
    return model


def evaluate(y_pred, y_actual):
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    return accuracy, precision, recall, f1


def build_model(args):
    """num_node_features / num_edge_features に合わせて run ごとにモデルを再構築する."""
    conv_dictionary = {
        'GCN': (GCNConv(num_node_features, hidden_channels), GCNConv(hidden_channels, hidden_channels)),
        'GAT': (GATConv(num_node_features, hidden_channels), GATConv(hidden_channels, hidden_channels)),
        'SAGE': (SAGEConv(num_node_features, hidden_channels), SAGEConv(hidden_channels, hidden_channels)),
        'GIN': (GINConv(Sequential(Linear(num_node_features, hidden_channels), nn.LeakyReLU(0.1),
                                   Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.1)),
                        train_eps=True),
                GINConv(Sequential(Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.1)),
                        train_eps=False)),
        'GINE': (GINEConv(Sequential(Linear(num_node_features, hidden_channels), nn.LeakyReLU(0.2),
                                     Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2)),
                          train_eps=True, edge_dim=num_edge_features),
                 GINEConv(Sequential(Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2),
                                     Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2)),
                          train_eps=True, edge_dim=num_edge_features)),
    }
    conv1, conv2 = conv_dictionary[model_name]
    if model_name == "GINE":
        model = GCN_edge(conv1, conv2, num_classes)
    else:
        model = GCN(conv1, conv2, num_classes)
    return model.to(device)


def run_one_tw(cache, t_w, epochs, args):
    """ある t_w について 5 回の実験を回し, 4 指標の per-run / mean / std を返す."""
    global criterion

    # 縮小グラフのデータセット G' を構築 (順序は cache 固定 → t_w 間で一貫).
    data_list = [build_snapshot(c, t_w) for c in cache]
    label_list = [c['label'] for c in cache]

    per_run = []
    for exp in range(5):
        seed_everything(exp)
        train_data, test_data, _ = split_data(data_list, label_list, seed=exp)
        criterion = BCEWithLogitsLoss()
        model = build_model(args)
        model = train_model(model, epochs, train_data, args)
        y_pred, y_actual, _ = predict(model, test_data, args)
        acc, prec, rec, f1 = evaluate(y_pred, y_actual)
        per_run.append([acc, prec, rec, f1])

    per_run = np.array(per_run)
    mean = np.round(per_run.mean(axis=0), 3)
    std = np.round(per_run.std(axis=0), 3)
    return per_run.tolist(), mean, std


def save_results(records, metric_names, out_base):
    """t_w ごとの結果を JSON / CSV / PNG に保存する."""
    os.makedirs('results', exist_ok=True)

    with open(out_base + '.json', 'w') as f:
        json.dump(records, f, indent=2)

    with open(out_base + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['t_w']
        for m in metric_names:
            header += [f'{m}_mean', f'{m}_std']
        writer.writerow(header)
        for r in records['per_tw']:
            row = [r['t_w']]
            for m in metric_names:
                row += [r['mean'][m], r['std'][m]]
            writer.writerow(row)

    # 4 指標 vs t_w の折れ線グラフ.
    try:
        tws = [r['t_w'] for r in records['per_tw']]
        plt.figure(figsize=(8, 5))
        for m in metric_names:
            means = [r['mean'][m] for r in records['per_tw']]
            plt.plot(tws, means, marker='.', label=m)
        plt.xlabel('t_w (minutes)')
        plt.ylabel('score')
        plt.title(f"{model_name} {records['args']['rww_attr']}: metrics vs snapshot window")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_base + '.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"Saved results to {out_base}.{{json,csv,png}}")


if __name__ == '__main__':
    print("Inside Main (snapshot experiment)")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN', help="Enter model name")
    parser.add_argument("--lr", default=1e-4, help="Enter learning rate")
    parser.add_argument('--hidden_dim', default=128, help="Enter hidden dimensions")
    parser.add_argument('--output_dim', default=2, help="Enter output dimensions")
    parser.add_argument('--data_type', default='all', help="Either small or All")
    parser.add_argument('--multivariate', default=0, help="Multivariate classification flag")
    parser.add_argument("--classify_news", default=0, help="Classify the news graphs")
    parser.add_argument("--small_graphs_path", help="Mention path to small graphs")
    parser.add_argument("--all_graphs_path", help="Mention path to all graphs")
    parser.add_argument("--rww_attr", default="degree", help="Mention what feature for rww")
    parser.add_argument("--node_attr", default="1", help="Whether node features should be used")
    parser.add_argument("--batch_size", default=32, help="Mini-batch size")
    parser.add_argument("--tw_min", default=1, type=int, help="Minimum t_w (minutes)")
    parser.add_argument("--tw_max", default=60, type=int, help="Maximum t_w (minutes)")
    parser.add_argument("--tw_step", default=1, type=int, help="Step for t_w sweep")
    args = parser.parse_args()

    model_name = args.model
    hidden_channels = int(args.hidden_dim)
    num_classes = int(args.output_dim)
    multivariate = int(args.multivariate)
    classify_news = int(args.classify_news)
    lr = float(args.lr)
    rww_attr = args.rww_attr
    node_attr = int(args.node_attr)

    if multivariate:
        raise SystemExit("This snapshot experiment script supports binary classification only (multivariate=0).")

    data_path = args.all_graphs_path if args.data_type != 'small' else args.small_graphs_path
    epochs = 100

    cache = build_cache(data_path, rww_attr, node_attr)
    print(f"Cache built: {len(cache)} graphs")
    print(f"Number of node features: {num_node_features}, edge features: {num_edge_features}")

    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    tw_values = list(range(args.tw_min, args.tw_max + 1, args.tw_step))

    records = {'args': vars(args), 'epochs': epochs,
               'num_node_features': num_node_features,
               'num_edge_features': num_edge_features,
               'metric_names': metric_names, 'per_tw': []}

    out_base = os.path.join(
        'results', f"snapshot_{model_name}_{rww_attr}_nodeattr{node_attr}")

    for t_w in tw_values:
        per_run, mean, std = run_one_tw(cache, t_w, epochs, args)
        rec = {
            't_w': t_w,
            'per_run': [[float(v) for v in run] for run in per_run],
            'mean': {name: float(mean[i]) for i, name in enumerate(metric_names)},
            'std': {name: float(std[i]) for i, name in enumerate(metric_names)},
        }
        records['per_tw'].append(rec)
        print(f"t_w={t_w:2d}min | "
              + ", ".join(f"{m}={rec['mean'][m]:.3f}±{rec['std'][m]:.3f}" for m in metric_names))
        # 途中経過も逐次保存 (長時間実行のため).
        save_results(records, metric_names, out_base)
