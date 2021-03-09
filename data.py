import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

def get_data(args):
    dataset = PygNodePropPredDataset(name=args['dataset_name'], transform=T.ToSparseTensor())
    evaluator = Evaluator(name=args['dataset_name'])

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    for setname in ['train', 'valid', 'test']:
        split_idx[setname] = split_idx[setname].to(device)

    return data, dataset, split_idx, evaluator
