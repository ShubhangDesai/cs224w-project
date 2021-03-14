from .IncepGCN import *
from .GNN import *

def get_model(dataset, args):
    if args['model_type'] in ['gcn', 'graphsage', 'gat']:
        model_type = GNN
    elif args['model_type'] == 'incepgcn':
        model_type = IncepGCN


    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    if args['dataset_name'] == 'ogbn-proteins':
    	in_channels = data.num_features
    	out_channels = 112 # 112 talks
    	print("in, out: ", in_channels, out_channels)

    model = model_type(in_channels, out_channels, args)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    return model
