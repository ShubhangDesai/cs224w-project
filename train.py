from models import *
from data import *

import numpy as np
import scipy.sparse as sp
import argparse, copy

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)

    # Common Model Parameters
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--dropedge', default=0.05, type=float)

    # GCN Parameters
    parser.add_argument('--num_heads', default=1, type=int)

    # IncepGCN Parameters
    parser.add_argument('--num_branches', default=3, type=int)
    parser.add_argument('--self_loops', default=True, type=bool)

    # Training Parameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)

    return parser

def train(model, data, train_idx, optimizer, loss_fn, dropedge_rate):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]

    train_label = data.y.squeeze(1)[train_idx]
    loss = loss_fn(out, train_label)

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

if __name__ == '__main__':
    args = vars(get_parser().parse_args())

    data, dataset, split_idx, evaluator = get_data(args)
    model = get_model(dataset, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_fn = F.nll_loss

    best_valid_acc = 0
    for epoch in range(1, args['epochs'] + 1):
        loss = train(model, data, split_idx['train'], optimizer, loss_fn, args['dropedge'])
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
