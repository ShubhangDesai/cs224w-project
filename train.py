from models import *
from data import *

import numpy as np
import argparse
# from torch_geometric.nn.models import CorrectAndSmooth

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

    # Dropedge parameters
    parser.add_argument('--dropedge', default=0.05, type=float)

    # FLAG parameters
    parser.add_argument('--flag', action='store_true')
    parser.add_argument('--m', default=3, type=int) # Number of steps to remove perturbations
    parser.add_argument('--step_size', default=1e-3, type=float) # Number of steps to remove perturbations

    # C&S Parameters
    parser.add_argument('--cs', action='store_true')
    parser.add_argument('--cs_layers', default=50, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)

    # GCN Parameters
    parser.add_argument('--num_heads', default=1, type=int)

    # IncepGCN Parameters
    parser.add_argument('--num_branches', default=3, type=int)
    parser.add_argument('--self_loops', default=True, type=bool)

    # GAT Parameters: num_heads = 3
    parser.add_argument('--attn_dropout', default=0.05, type=float)

    # Training Parameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)

    return parser

def train(model, data, train_idx, optimizer, loss_fn, dropedge_rate, apply_flag, flag_steps, flag_step_size):
    model.train()
    optimizer.zero_grad()
    train_label = data.y.squeeze(1)[train_idx] # Get labels

    # Add FLAG: unbiased perturbation
    out = None
    loss = None
    if apply_flag: # Train with FLAG
        loss, out = flag_train(model, data, train_label, train_idx, loss_fn, dropedge_rate, flag_steps, flag_step_size)
    
    else:
        out = model(data.x, data.adj_t)[train_idx]
        loss = loss_fn(out, train_label)

    loss.backward()
    optimizer.step()

    del data, train_idx, out
    torch.cuda.empty_cache()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator, y_pred=None):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True) if y_pred is None else y_pred # Check if soft label for C&S

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

    del data, split_idx, out #, y_pred
    torch.cuda.empty_cache()

    return train_acc, valid_acc, test_acc, y_pred

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
            loss = train(model, data, split_idx['train'], optimizer, loss_fn, args['dropedge'], args['flag'], args['m'], args['step_size'])
            train_acc, valid_acc, test_acc, pred_soft = test(model, data, split_idx, evaluator) 

            # C&S
            # if args['cs']:
            #     print('Correct and smooth...')

                # # Compute updated adjacency matrices
                # adj_t = data.adj_t.to('cuda' if torch.cuda.is_available() else 'cpu')
                # deg = adj_t.sum(dim=1).to(torch.float)
                # deg_inv_sqrt = deg.pow_(-0.5)
                # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                # DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1) # D^(-1/2) A D^(-1/2)
                # DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t # D A

                # # Correct and Smooth
                # post_process = CorrectAndSmooth(num_correction_layers=args['cs_layers'], correction_alpha=args['alpha'],
                #                         num_smoothing_layers=args['cs_layers'], smoothing_alpha=args['alpha'])
                # pred_soft = post_process.correct(pred_soft, data.y[split_idx['train']], split_idx['train'], DAD)
                # pred_soft = post_process.smooth(pred_soft, data.y[split_idx['train']], split_idx['train'],DA)

                # # Compute final results
                # train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator, pred_soft) # Input new predictions

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

    print("Parameters: ", args)
    print('=' * 20)

    print('Final Results')
    print('=' * 20)
    print(f'Best Train Acc: {100*np.mean(best_train_accs):0.2f}% ± {100*np.std(best_train_accs):0.2f}%')
    print(f'Best Valid Acc: {100*np.mean(best_valid_accs):0.2f}% ± {100*np.std(best_valid_accs):0.2f}%')
    print(f'Best Test Acc: {100*np.mean(best_test_accs):0.2f}% ± {100*np.std(best_test_accs):0.2f}%')
