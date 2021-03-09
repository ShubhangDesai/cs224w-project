import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .Dropedge import *

class GCNBlock(nn.Module):
  '''
  Block of layers containing GCN --> BatchNorm --> Dropout --> RELU
  Used as a unit for Inception architecture
  '''
  def __init__(self, in_dim, out_dim, self_loops, dropout=0.5, use_relu=True, use_dropout=True):
    super(GCNBlock, self).__init__()

    self.conv = GCNConv(in_dim, out_dim, add_self_loops=self_loops)
    self.bn = nn.BatchNorm1d(out_dim)
    self.dropout = dropout
    self.use_relu = use_relu
    self.use_dropout = use_dropout

  def forward(self, x, adj_t):
    out = x
    out = self.conv(out, adj_t)
    out = self.bn(out)

    if self.use_relu: out = F.relu(out)
    if self.use_dropout: out = F.dropout(out, self.dropout, self.training)

    return out


class IncepGCNBlock(nn.Module):
  '''
  Block of inception module branches
  Multiple branches of graph layers that are concatenated for the output
  '''
  def __init__(self, num_branches, in_dim, out_dim, self_loops, dropout):
    super(IncepGCNBlock, self).__init__()

    # Inception branch layers
    self.branches = nn.ModuleList()
    for i in range(num_branches): # Each inception branch
      branch = nn.ModuleList() # Each layer (GCN+BN+Dropout) within a branch
      for j in range(i + 1):
        curr_in_dim = in_dim if j == 0 else out_dim
        branch.append(GCNBlock(curr_in_dim, out_dim, self_loops, dropout=dropout))

      self.branches.append(branch)

    self.block_out_dim = in_dim + num_branches * out_dim # Save dimensions for later use

  def forward(self, input, adj_t):
    out = input
    for branch in self.branches: # Iterate over each Inception branch
      branch_out = input
      for gcn in branch: # Iterate over each GCN block of layers
        branch_out = gcn(branch_out, adj_t)

      out = torch.cat((out, branch_out), 1) # Concatenate: (block_out_dim)

    return out

  def get_block_out_dim(self):
    return self.block_out_dim


class IncepGCN(nn.Module):
  '''
  Inception architecture with GCN
  '''
  def __init__(self, in_dim, out_dim, args):
    super(IncepGCN, self).__init__()

    self.dropedge = Dropedge(args['dropedge']) if args['dropedge'] != 0 else None

    self.in_layer = GCNBlock(in_dim, args['hidden_dim'], args['self_loops'], dropout=args['dropout'])
    current_in_dim = args['hidden_dim']

    self.blocks = nn.ModuleList()
    for i in range(args['num_layers']):
      block = IncepGCNBlock(args['num_branches'], current_in_dim, args['hidden_dim'], args['self_loops'], args['dropout'])
      self.blocks.append(block)
      current_in_dim = block.get_block_out_dim()

    self.out_layer = GCNBlock(current_in_dim, out_dim, args['self_loops'], use_relu=False, use_dropout=False)

  def forward(self, x, adj_t):
    if self.training and self.dropedge is not None:
        adj_t = self.dropedge(adj_t)

    out = self.in_layer(x, adj_t)

    for block in self.blocks: # Hidden inception blocks
      out = block(out, adj_t)

    out = self.out_layer(out, adj_t)
    out = F.log_softmax(out, dim=-1)

    return out
