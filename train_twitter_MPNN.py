import sys
# sys.path.append('/projects/academic/erdem/atulanan/twitter_analytics/CRaWl/')
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
import json
import argparse
import glob
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv, SAGEConv
from torch.nn import LazyLinear, Linear, Sequential, Dropout, LeakyReLU, Sigmoid
import torch.optim as optim
import random
from random import sample
from torch_geometric.utils import train_test_split_edges
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import Dataset, Data
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, BatchNorm1d as BN)
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling, global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx
from torch.nn.functional import normalize
import torch.nn as nn
from statistics import mean, stdev
import os
import pickle
import shutil
import warnings
import math
import time
from models import GCN, GCN_edge, GCN_News
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend: safe for saving many figures on a server.
import matplotlib.pyplot as plt
import copy
import re

warnings.filterwarnings("ignore")

data_path = None
num_node_features = None
num_edge_features = None
num_classes = None
lr = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = None
multivariate = None
noise_graphs = []
classify_news = None
ctpooling = None
model_name = None
maxcore = None
lf = None
hidden_channels = None

temporal = None


# opt = optim.Adam(model.parameters(), lr=1e-3)

def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mean_time(time_ranges):
    time_ranges_in_seconds = [tr.total_seconds() for tr in time_ranges]
    mean_seconds = mean(time_ranges_in_seconds)
    stdev_seconds = stdev(time_ranges_in_seconds)

    mean_time = pd.to_timedelta(mean_seconds, unit='s')
    stdev_time = pd.to_timedelta(stdev_seconds, unit='s')

    return mean_time, stdev_time

def process_data(files, graph_labels, exceptions, rww_attr, node_attr):
    data_list = []
    label_list = []

    campaign_news_graphs = []
    noncampaign_news_graphs = []

    total = len(files)
    for i, file in enumerate(files):
        # Get the graph label
        file_name = file.split('/')[-1][:-5]
        print(f"\r{(i + 1) / total * 100:.1f}% {file_name}", end='', flush=True)
        if file_name not in exceptions and (file_name[:-9] in graph_labels):
            graph_label = graph_labels[file_name[:-9]]
            # label_counter[graph_label] += 1
            with open(file, 'r') as f:
                data = json.load(f)
            graph = nx.DiGraph(json_graph.node_link_graph(data))

            mapping = {node: i for i, node in enumerate(graph.nodes())}
            graph = nx.relabel.relabel_nodes(graph, mapping)
            label_list.append(graph_label)
            y = [graph_label]
            y = torch.tensor(y)
            # x = torch.tensor([graph.nodes[node]['node_attr'] for node in graph.nodes()])
            if rww_attr == 'kcore' and node_attr == 1:
                x = torch.tensor([graph.nodes[node]['node_attr'] + graph.nodes[node]['structural_embedding'] for node in graph.nodes()])
            elif rww_attr == 'kcore' and node_attr == 0:
                x = torch.tensor([graph.nodes[node]['structural_embedding'] for node in graph.nodes()])
            elif rww_attr == 'degree' and node_attr == 1:
                x = torch.tensor([graph.nodes[node]['node_attr'] + graph.nodes[node]['degree_embedding'] for node in graph.nodes()])
            elif rww_attr == 'degree' and node_attr == 0:
                x = torch.tensor([graph.nodes[node]['degree_embedding'] for node in graph.nodes()])
            elif rww_attr == 'ktruss' and node_attr == 1:
                x = torch.tensor([graph.nodes[node]['node_attr'] + graph.nodes[node]['truss_embedding'] for node in graph.nodes()])
            elif rww_attr == 'ktruss' and node_attr == 0:
                x = torch.tensor([graph.nodes[node]['truss_embedding'] for node in graph.nodes()])
            else:
                x = torch.tensor([graph.nodes[node]['node_attr'] for node in graph.nodes()])

            global num_edge_features, num_node_features

            edge_index = torch.tensor([e for e in graph.edges], dtype=torch.long)
            edge_attr = torch.tensor([graph.edges[edge]['edge_attr'] for edge in graph.edges()])

            num_node_features = x.shape[1]
            num_edge_features = edge_attr.shape[1]

            data = Data(x=x, edge_index=edge_index, y=y)
            data.y = data.y.view(-1)
            data.edge_attr = edge_attr
            data.edge_index = torch.transpose(data.edge_index, 0, 1)
            data.name = file_name
            data_list.append(data)

    print()
    return data_list, label_list


def load_data(data_path, rww_attr, node_attr):
    print("Loading dataset.....")

    path = data_path
    label_path = "/hss01/A.hattori/all_graphs"

    if multivariate:
        files = list(glob.glob(path + '/*_campaign_fulldata.json'))
        files_news = list(glob.glob(path + '/news/*_fulldata.json'))
        files_finance = list(glob.glob(path + '/finance/*_fulldata.json'))
    else:
        files = list(glob.glob(path + '/*_fulldata.json'))

    if multivariate:
        with open(label_path + "/graph_labels_campaign.json", "r") as f:
            graph_labels = json.load(f)
    elif classify_news:
        with open(label_path + "/graph_labels_news.json", "r") as f:
            graph_labels = json.load(f)
    else:
        with open(label_path + "/graph_labels.json", "r") as f:
            graph_labels = json.load(f)

    exceptions = ['graph_labels', "Gomis_noncampaign_fulldata", "#Hıdırellez_noncampaign_fulldata",
                '35YaşŞartı_TorbaYasaya__2023-03-26_campaign_fulldata', 'Haluk_noncampaign_fulldata',
                '#ErdenTimurSezonu_noncampaign_fulldata', 'Gustavo_noncampaign_fulldata']
    data_list, label_list = process_data(files, graph_labels, exceptions, rww_attr, node_attr)

    return data_list, label_list


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

    train_data, test_data, train_labels, test_labels = train_test_split(data_list, label_list, stratify=label_list,
                                                                        test_size=0.20, shuffle=True,
                                                                        random_state=seed)
    random.shuffle(train_data)

    return train_data, test_data, None

def predict(model, test_data, args):
    model.eval()
    y_pred = []
    y_actual = []
    y_scores = []
    prediction_counter = {}
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
            for p in predictions.tolist():
                prediction_counter[p] = prediction_counter.get(p, 0) + 1
            y_pred += predictions.tolist()
            y_actual += labels.tolist()
            y_scores += pred[:, 1].tolist()
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    y_scores = np.array(y_scores)

    print(f"Labels predicted are: {prediction_counter}")
    return y_pred, y_actual, y_scores

def train_model(model, epochs, train_data, val_data, args):
    opt = optim.Adam(model.parameters(), lr=lr)

    train_loss_epochs = []
    val_loss_epochs = []

    # Mini-batch loader; shuffle=True reshuffles every epoch.
    loader = DataLoader(train_data, batch_size=int(args.batch_size), shuffle=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        counter = 0

        for batch in loader:
            batch = batch.to(device)
            if args.model == "GINE":
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                pred = model(batch.x, batch.edge_index)

            pooled_output = global_mean_pool(pred, batch.batch)

            # Raw logits: the loss functions (BCEWithLogitsLoss / CrossEntropyLoss)
            # apply their own activation internally, so no softmax here.
            pred = model.out(pooled_output)
            # Generate Labels
            if (multivariate):
                label = batch.y.to(device)
            else:
                # Per-graph one-hot targets for BCEWithLogitsLoss.
                label = F.one_hot(batch.y, num_classes=2).float().to(device)

            criterion.to(device)

            loss = criterion(pred, label)
            train_loss += loss.item()
            counter += 1
            loss.backward()
            opt.step()
            opt.zero_grad()
        train_loss /= counter
        # val_loss = val_model(model, val_data, args)
        train_loss_epochs.append(train_loss)
        # val_loss_epochs.append(val_loss)
    return model, train_loss_epochs, val_loss_epochs



def evaluate(y_pred, y_actual):
    micro_f1 = None
    macro_f1 = None
    precision = None
    recall = None
    accuracy = accuracy_score(y_actual, y_pred)
    if multivariate:
        precision = precision_score(y_actual, y_pred, average='weighted')
        recall = recall_score(y_actual, y_pred, average='weighted')
        micro_f1 = f1_score(y_actual, y_pred, average='micro')
        macro_f1 = f1_score(y_actual, y_pred, average='macro')
        return accuracy, precision, recall, micro_f1, macro_f1

    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)

    # macro_f1 = f1_score(y_actual, y_pred, average='macro')

    return accuracy, precision, recall, f1


def getReport(y_pred, y_actual):
    print(f"Prediction: {y_pred}")
    print(f"Actual: {y_actual}")
    y_actual[-1] = 2
    y_actual[-5] = 2
    cm = confusion_matrix(y_pred, y_actual)

    # Set the figure size and plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 12))  # Increase the figure size as needed
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["News", "Politics", "Entertainment",
                                                                       "Reform", "Common", "Cult", "Finance"])

    disp.plot(ax=ax)

    # Adjust label sizes
    ax.set_xlabel('True label', fontsize=10)
    ax.set_ylabel('Predicted label', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)  # Adjust tick label size

    # Save the confusion matrix as an image file
    plt.savefig(f'{model_name}_confusion_matrix.png')

    plt.show()


def save_graph_image(data, name, pred, actual, out_path):
    """Draw one graph's network diagram and annotate it with the true label,
    node count and edge count, then save to out_path."""
    num_nodes = data.num_nodes
    num_edges = int(data.edge_index.shape[1])

    G = to_networkx(data, to_undirected=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    # spring_layout is O(n^2) per iteration; fall back to the fast random layout
    # for large graphs so saving every test graph doesn't stall.
    if num_nodes <= 500:
        pos = nx.spring_layout(G, seed=1)
    else:
        pos = nx.random_layout(G, seed=1)
    nx.draw(G, pos, ax=ax, node_size=20, width=0.3, arrows=True,
            with_labels=False, node_color='#1f77b4', edge_color='#999999')
    ax.set_title(f"{name}\nactual={actual}  pred={pred}\n"
                 f"nodes={num_nodes}  edges={num_edges}", fontsize=10)
    ax.axis('off')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def log_and_visualize(test_data, y_pred, y_actual, run_dir):
    """Log per-file predictions vs. ground truth and save a visualization of
    each test graph into a folder named after its predicted label.

    Relies on predict() using shuffle=False so that y_pred[i] / y_actual[i]
    correspond to test_data[i].
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'predictions.log')

    total = len(test_data)
    with open(log_path, 'w') as logf:
        logf.write("file_name\tpredicted\tactual\tcorrect\tnum_nodes\tnum_edges\n")
        for i, data in enumerate(test_data):
            name = getattr(data, 'name', f'graph_{i}')
            pred = int(y_pred[i])
            actual = int(y_actual[i])
            num_nodes = data.num_nodes
            num_edges = int(data.edge_index.shape[1])
            correct = (pred == actual)

            logf.write(f"{name}\t{pred}\t{actual}\t{correct}\t{num_nodes}\t{num_edges}\n")

            # One folder per predicted label; index-prefixed, sanitized filename
            # keeps names unique and filesystem-safe (keeps unicode word chars).
            label_dir = os.path.join(run_dir, f'pred_{pred}')
            os.makedirs(label_dir, exist_ok=True)
            safe = re.sub(r'[^\w\-.#]', '_', name)
            out_path = os.path.join(label_dir, f"{i:04d}_{safe}.png")
            save_graph_image(data, name, pred, actual, out_path)

            print(f"\r  Saving visualizations {i + 1}/{total}", end='', flush=True)
    print(f"\n  Wrote predictions log and visualizations to {run_dir}")


if __name__ == '__main__':
    print("Inside Main")
    # small_dir ="/projects/academic/erdem/atulanan/twitter_analytics/new_networks/fulldata/descriptive_data/small_encoder_final"
    # All_dir ="/projects/academic/erdem/atulanan/twitter_analytics/new_networks/fulldata/descriptive_data/encoder_final"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN', help="Enter model name")
    parser.add_argument("--lr", default=1e-4, help="Enter learning rate")
    parser.add_argument('--hidden_dim', default=128, help="Enter hidden dimensions")
    parser.add_argument('--output_dim', default=2, help="Enter output dimensions")
    parser.add_argument('--data_type', default='small', help="Either small or All")
    parser.add_argument('--edge_attr', default=0, help="Either small or All")
    parser.add_argument('--multivariate', default=0,
                        help="Initialize if you are performing multivariate classification")
    parser.add_argument("--classify_noise", default=0, help="Classify the noise graphs")
    parser.add_argument("--classify_news", default=0, help="Classify the news graphs")
    parser.add_argument("--small_graphs_path", help="Mention path to small graphs")
    parser.add_argument("--all_graphs_path", help="Mention path to all graphs")
    parser.add_argument("--rww_attr", default="kcore", help="Mention what feature for rww")
    parser.add_argument("--node_attr", default="1", help="Mention whether node features should be used or not")
    parser.add_argument("--batch_size", default=32, help="Mini-batch size for training and inference")
    parser.add_argument("--analysis_out",
                        default="/home/A.hattori/ECMLPKDD25/results/20260715_時間幅0で埋め込み再計算した実験_テスト時のログを可視化",
                        help="Base directory for per-test prediction logs and graph visualizations")
    args = parser.parse_args()

    model_name = args.model  # model name
    hidden_channels = int(args.hidden_dim)  # hidden dim
    num_classes = int(args.output_dim)  # output dim
    multivariate = int(args.multivariate)
    classify_noise = int(args.classify_noise)
    classify_news = int(args.classify_news)
    small_dir = args.small_graphs_path
    all_dir = args.all_graphs_path
    lr = float(args.lr)
    rww_attr = args.rww_attr
    node_attr = int(args.node_attr)

    if args.data_type == 'small':
        data_path = small_dir
    else:
        data_path = all_dir

    epochs = 100

    data_list, label_list = load_data(data_path, rww_attr, node_attr)
    print("Dataset loading done  ", data_path, len(data_list))
    print(f"Number of node features: {num_node_features} and number of edge features :{num_edge_features}")

    all_results = []
    training_time = []
    # Per-setting subfolder under the fixed analysis base dir keeps the outputs of
    # different models (run back-to-back by the launcher .sh) from overwriting.
    analysis_base_dir = os.path.join(
        args.analysis_out,
        f"{model_name}_{rww_attr}_nodeattr{node_attr}_{args.data_type}_mv{multivariate}")
    for exp in range(0, 5):
        seed_everything(exp)

        # Re-split the data for every run so the reported variance reflects
        # split variability, not just weight initialisation.
        train_data, test_data, val_data = split_data(data_list, label_list, seed=exp)

        if not multivariate:
            criterion = BCEWithLogitsLoss()
        else:
            label_counts = dict()
            for i in range(len(train_data)):
                graph = train_data[i]
                label = graph.y.tolist()[0]

                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1

            label_counts = dict(sorted(label_counts.items()))
            label_counts = np.array(list(label_counts.values()))
            # Inverse-frequency class weights. The previous np.exp(-label_counts)
            # underflowed to ~0 for all but the rarest class.
            weights = label_counts.sum() / (len(label_counts) * label_counts)
            weights /= np.sum(weights)
            print(weights, label_counts)
            weights = torch.tensor(weights, dtype=torch.float32)
            criterion = CrossEntropyLoss(weight=weights)

        print(f"Length of training, testing datasets: {len(train_data)} {len(test_data)}")

        # num_node_features / num_edge_features are set during data loading, so the
        # conv layers must be rebuilt per run.
        conv_dictionary = {'GCN': (GCNConv(num_node_features, hidden_channels), GCNConv(hidden_channels, hidden_channels)),

                           'GAT': (GATConv(num_node_features, hidden_channels), GATConv(hidden_channels, hidden_channels)),

                           'SAGE': (
                           SAGEConv(num_node_features, hidden_channels), SAGEConv(hidden_channels, hidden_channels)),

                           'GIN': (GINConv(Sequential(Linear(num_node_features, hidden_channels), nn.LeakyReLU(0.1),
                                                      Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.1), ),
                                           train_eps=True),
                                   GINConv(Sequential(Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.1), ),
                                           train_eps=False)),

                           'GINE': (GINEConv(Sequential(Linear(num_node_features, hidden_channels), nn.LeakyReLU(0.2),
                                                        Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2), ),
                                             train_eps=True, edge_dim=num_edge_features),
                                    GINEConv(Sequential(Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2),
                                                        Linear(hidden_channels, hidden_channels), nn.LeakyReLU(0.2), ),
                                             train_eps=True, edge_dim=num_edge_features))
                           }

        if model_name == "GINE":
            conv1 = conv_dictionary[model_name][0]
            conv2 = conv_dictionary[model_name][1]
            args.edge_attr = 0
            model = GCN_edge(conv1, conv2, num_classes)
        else:
            conv1 = conv_dictionary[model_name][0]
            conv2 = conv_dictionary[model_name][1]
            args.edge_attr = 0

            model = GCN(conv1, conv2, num_classes)

        model.to(device)
        model, train_loss_epochs, val_loss_epochs = train_model(model, epochs, train_data, val_data, args)
        y_pred, y_actual, y_scores = predict(model, test_data, args)

        # Log per-file predictions vs. ground truth and save graph visualizations
        # grouped by predicted label (shuffle=False in predict keeps order aligned).
        run_dir = os.path.join(analysis_base_dir, f'exp{exp}')
        log_and_visualize(test_data, y_pred, y_actual, run_dir)

        if (exp == 4):
            print(f"Prediction: {y_pred}")
            print(f"Actual: {y_actual}")
            print(f"Y scores: {y_scores}")

        if multivariate:
            acc, prec, rec, micro_f1, macro_f1 = evaluate(y_pred, y_actual)
            all_results.append([acc, prec, rec, micro_f1, macro_f1])
        else:
            acc, prec, rec, f1 = evaluate(y_pred, y_actual)
            all_results.append([acc, prec, rec, f1])

    if multivariate:
        # print(all_results, np.mean(all_results, axis=0), np.std(all_results, axis=0))
        all_mean, all_std = np.round(np.mean(all_results, axis=0), 3), np.round(np.std(all_results, axis=0), 3)
        print(model_name, args.data_type)
        print(
            f"{all_mean[0]} ± {all_std[0]},{all_mean[1]} ± {all_std[1]},{all_mean[2]} ± {all_std[2]},{all_mean[3]} ± {all_std[3]},{all_mean[4]} ± {all_std[4]}")
        # print(f"Time taken: {int(all_mean[5]//60)}:{all_mean[5]%60} ± {int(all_std[5]//60)}:{all_std[5]%60}")
    else:
        # print(all_results, np.mean(all_results, axis=0), np.std(all_results, axis=0))
        all_mean, all_std = np.round(np.mean(all_results, axis=0), 3), np.round(np.std(all_results, axis=0), 3)
        print(model_name, args.data_type)
        print(
            f"{all_mean[0]} ± {all_std[0]},{all_mean[1]} ± {all_std[1]},{all_mean[2]} ± {all_std[2]},{all_mean[3]} ± {all_std[3]}")
        # print(f"Time taken: {int(all_mean[4] // 60)}:{all_mean[4] % 60} ± {int(all_std[4] // 60)}:{all_std[4] % 60}")

    # Persist the run configuration and metrics as JSON under results/.
    if multivariate:
        metric_names = ['accuracy', 'precision', 'recall', 'micro_f1', 'macro_f1']
    else:
        metric_names = ['accuracy', 'precision', 'recall', 'f1']

    os.makedirs('results', exist_ok=True)
    result_record = {
        'args': vars(args),
        'epochs': epochs,
        'num_node_features': num_node_features,
        'num_edge_features': num_edge_features,
        'metric_names': metric_names,
        'per_run': [[float(v) for v in run] for run in all_results],
        'mean': {name: float(all_mean[i]) for i, name in enumerate(metric_names)},
        'std': {name: float(all_std[i]) for i, name in enumerate(metric_names)},
    }
    out_path = os.path.join(
        'results',
        f"{model_name}_{rww_attr}_nodeattr{node_attr}_{args.data_type}_mv{multivariate}.json")
    with open(out_path, 'w') as f:
        json.dump(result_record, f, indent=2)
    print(f"Saved results to {out_path}")
