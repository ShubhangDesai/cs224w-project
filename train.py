from models import *
from data import *

import numpy as np
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--runs', default=1, type=int)

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

    del data, train_idx, out
    torch.cuda.empty_cache()

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

    del data, split_idx, out, y_pred
    torch.cuda.empty_cache()

    return train_acc, valid_acc, test_acc

if __name__ == '__main__':
    args = vars(get_parser().parse_args())

    data, dataset, split_idx, evaluator = get_data(args)
    loss_fn = F.nll_loss

    best_train_accs, best_valid_accs, best_test_accs = [], [], []
    for run in range(1, args['runs'] + 1):
        print('Run ' + str(run))
        print('=' * 20)

        model = get_model(dataset, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        best_train_acc, best_valid_acc, best_test_acc = 0, 0, 0
        for epoch in range(1, args['epochs'] + 1):
            loss = train(model, data, split_idx['train'], optimizer, loss_fn, args['dropedge'])
            train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)

            if valid_acc > best_valid_acc:
                best_train_acc, best_valid_acc, best_test_acc = train_acc, valid_acc, test_acc

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

        best_train_accs.append(best_train_acc)
        best_valid_accs.append(best_valid_acc)
        best_test_accs.append(best_test_acc)

        del model
        print()

    print('Final Results')
    print('=' * 20)
    print(f'Best Train Acc: {100*np.mean(best_train_accs):0.2f}% ± {100*np.std(best_train_accs):0.2f}%')
    print(f'Best Valid Acc: {100*np.mean(best_valid_accs):0.2f}% ± {100*np.std(best_valid_accs):0.2f}%')
    print(f'Best Test Acc: {100*np.mean(best_test_accs):0.2f}% ± {100*np.std(best_test_accs):0.2f}%')
