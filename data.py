import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

def get_data(args):
    dataset = PygNodePropPredDataset(name=args['dataset_name'], transform=T.ToSparseTensor())
    evaluator = Evaluator(name=args['dataset_name'])

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    print(args['dataset_name'])
    if args['dataset_name'] == "ogbn-proteins":
        data.x = data.adj_t.mean(dim=1)

        # Pre-compute GCN normalization for adjacency matrix
        if args['model_type'] != 'gat' and args['model_type'] != 'graphsage':
            data.adj_t.set_value_(None)
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t

    split_idx = dataset.get_idx_split()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    for setname in ['train', 'valid', 'test']:
        split_idx[setname] = split_idx[setname].to(device)

    return data, dataset, split_idx, evaluator
