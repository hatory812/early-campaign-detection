import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv
from torch.nn import LazyLinear, Linear, Sequential, Dropout, LeakyReLU, Sigmoid
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,Sequential, BatchNorm1d as BN)
import torch.nn.functional as F


class GCN_edge(torch.nn.Module):
    def __init__(self, conv1, conv2, num_classes):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2

        self.w_kcore_network = Sequential(
            LazyLinear(out_features=128),  # First lazy linear layer
            Sigmoid(),  # Leaky ReLU activation
            LazyLinear(out_features=1),  # Second lazy linear layer
            Sigmoid()
        )

        self.out = Sequential(
            LazyLinear(out_features=128),  # First lazy linear layer
            Sigmoid(),
            LazyLinear(out_features=num_classes)  # Second lazy linear layer (raw logits)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        return x

class GCN_News(torch.nn.Module):
    def __init__(self, conv1, conv2, num_classes):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = Dropout(0.3)

        self.out = Sequential(
            LazyLinear(out_features=num_classes)  # Second lazy linear layer (raw logits)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.sigmoid(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        # x = F.leaky_relu(self.conv3(x, edge_index))
        return x

class GCN(torch.nn.Module):
    def __init__(self, conv1, conv2, num_classes, dropout=0):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.dropout = dropout

        self.dropout = Dropout(self.dropout)
        self.out = Sequential(
            LazyLinear(out_features=1024),  # First lazy linear layer
            Sigmoid(),  # Leaky ReLU activation
            LazyLinear(out_features=num_classes)  # Second lazy linear layer (raw logits)
        )

        self.w_ktruss_network = Sequential(
            LazyLinear(out_features=128),  # First lazy linear layer
            Sigmoid(),  # Leaky ReLU activation
            LazyLinear(out_features=1),  # Second lazy linear layer
            Sigmoid()
        )

        self.w_kcore_network = Sequential(
            LazyLinear(out_features=128),  # First lazy linear layer
            Sigmoid(),  # Leaky ReLU activation
            LazyLinear(out_features=1),  # Second lazy linear layer
            Sigmoid()
        )
        self.w_ktruss = torch.nn.Parameter(torch.tensor(1.0))
        self.w_kcore = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        #x = self.dropout(x)  # Apply dropout after first convolution
        x = F.leaky_relu(self.conv2(x, edge_index))
        #x = self.dropout(x)  # Apply dropout after second convolution
        return x

