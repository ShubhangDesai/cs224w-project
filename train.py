from data import *
from losses import *
from models import *

import numpy as np
import argparse, functools

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--runs', default=1, type=int)

    # Common Model Parameters
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--dropedge', default=0, type=float)

    # GCN Parameters
    parser.add_argument('--num_heads', default=1, type=int)

    # IncepGCN Parameters
    parser.add_argument('--num_branches', default=3, type=int)
    parser.add_argument('--self_loops', default=True, type=bool)

    # Loss Parameters
    parser.add_argument('--coral_lambda', default=0, type=float)
    parser.add_argument('--num_loss_layers', default=1, type=int)

    # Training Parameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_flag_steps', default=0, type=int)
    parser.add_argument('--flag_step_size', default=1e-3, type=float)

    return parser

def flag_steps(model, data, split_idx, loss_fn, num_flag_steps, flag_step_size):
    perturb = torch.FloatTensor(data.x.shape).uniform_(-flag_step_size, flag_step_size) # Uniformation perturbation
    perturb = perturb.to('cuda' if torch.cuda.is_available() else 'cpu').requires_grad_()

    out, hiddens = model(data.x + perturb, data.adj_t)
    loss = loss_fn(out, data.y.squeeze(1), split_idx, hiddens) / num_flag_steps

    for _ in range(num_flag_steps): # Gradient ascent
        loss.backward()
        perturb.data = (perturb.detach() + flag_step_size * torch.sign(perturb.grad.detach())).data # Perturbation gradient ascent
        perturb.grad[:] = 0.

        out, hiddens = model(data.x + perturb, data.adj_t)
        loss = loss_fn(out, data.y.squeeze(1), split_idx, hiddens) / num_flag_steps

    del out
    torch.cuda.empty_cache()

    return loss

def train(model, data, split_idx, optimizer, loss_fn, num_flag_steps, flag_step_size):
    model.train()
    optimizer.zero_grad()

    if num_flag_steps == 0:
        out, hiddens = model(data.x, data.adj_t)
        loss = loss_fn(out, data.y.squeeze(1), split_idx, hiddens)
    else:
        loss = flag_steps(model, data, split_idx, loss_fn, num_flag_steps, flag_step_size)

    loss.backward()
    optimizer.step()

    del data, split_idx
    if num_flag_steps == 0: del out, hiddens
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
    loss_fn = get_loss(args)
    train_fn = functools.partial(train, num_flag_steps=args['num_flag_steps'], flag_step_size=args['flag_step_size'])

    best_train_accs, best_valid_accs, best_test_accs = [], [], []
    for run in range(1, args['runs'] + 1):
        print('Run ' + str(run))
        print('=' * 20)

        model = get_model(dataset, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        best_train_acc, best_valid_acc, best_test_acc = 0, 0, 0
        for epoch in range(1, args['epochs'] + 1):
            loss = train_fn(model, data, split_idx, optimizer, loss_fn)
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
