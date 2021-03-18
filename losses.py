import torch.nn as nn
import torch.nn.functional as F

class GraphCORAL(nn.Module):
    def __init__(self, lambd, num_loss_layers):
        super(GraphCORAL, self).__init__()

        self.lambd = lambd
        self.num_loss_layers = num_loss_layers

    def forward(self, out, label, split_idx, hiddens):
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        loss = F.nll_loss(out[train_idx], label[train_idx])

        for i, hidden in enumerate(hiddens[::-1]):
            train_feats = hidden[train_idx].mean(0)
            valid_feats = hidden[valid_idx].mean(0)
            test_feats = hidden[test_idx].mean(0)

            loss += self.lambd * (train_feats - valid_feats).norm()
            loss += self.lambd * (train_feats - test_feats).norm()
            
            if i == (self.num_loss_layers - 1): break

        return loss

def nll_loss(out, label, split_idx, hiddens):
    train_idx = split_idx['train']
    loss = F.nll_loss(out[train_idx], label[train_idx])

    return loss

def get_loss(args):
    if args['coral_lambda'] != 0:
        loss_fn = GraphCORAL(args['coral_lambda'], args['num_loss_layers'])
    else:
        loss_fn = nll_loss

    return loss_fn 
