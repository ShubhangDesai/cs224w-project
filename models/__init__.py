from .IncepGCN import *
from .GCN import *

def get_model(dataset, args):
    if args['model_type'] in ['graphsage', 'gat']:
        model_type = GCN
    elif args['model_type'] == 'incepgcn':
        model_type = IncepGCN

    model = model_type(dataset.num_node_features, dataset.num_classes, args)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    return model
