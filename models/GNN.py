import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from .Dropedge import *

class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN, self).__init__()

        self.dropedge = Dropedge(args['dropedge']) if args['dropedge'] != 0 else None

        if args['model_type'] == 'gcn': conv_type = pyg_nn.GCNConv
        elif args['model_type'] == 'graphsage': conv_type = pyg_nn.SAGEConv
        elif args['model_type'] == 'gat': conv_type = pyg_nn.GATConv

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(conv_type(input_dim, args['hidden_dim']))
        self.bns.append(nn.BatchNorm1d(args['hidden_dim']))
        for l in range(args['num_layers']-2):
            self.convs.append(conv_type(args['num_heads'] * args['hidden_dim'], args['hidden_dim']))
            self.bns.append(nn.BatchNorm1d(args['hidden_dim']))
        self.convs.append(conv_type(args['num_heads'] * args['hidden_dim'], output_dim))

        self.dropout = args['dropout']
        self.num_layers = args['num_layers']

    def forward(self, x, adj_t):
        if self.training and self.dropedge is not None:
            adj_t = self.dropedge(adj_t)

        for i in range(self.num_layers-1):
            print("W shape: {}, x shape: {}".format(self.convs[i].weight.shape, x.shape))
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        x = F.log_softmax(x, dim=1)

        return x
