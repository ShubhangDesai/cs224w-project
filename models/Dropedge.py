import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_adj

# Applies dropedge with bernoulli function with conversions
class Dropedge(object):
    def __init__(self, dropedge_rate):
        self.dropedge_rate = dropedge_rate

    def apply_dropedge(self, adj_t):
        # num nodes:
        num_row_nodes = adj_t.t().size(dim=0)
        num_col_nodes = adj_t.t().size(dim=1)

        # Convert to edge index to use DropEdge PyG function
        row, col, edge_attr = adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)

        # Dropedge
        train_edge_index = dropout_adj(edge_index, edge_attr, p=self.dropedge_rate)
        train_adj = SparseTensor(row=train_edge_index[0][0], col=train_edge_index[0][1], sparse_sizes=(num_row_nodes,num_col_nodes))
        train_adj_t = train_adj.t()

        return train_adj_t

    def __call__(self, adj_t):
        return self.apply_dropedge(adj_t)
