import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.graph_utils import *
import dgl.function as fn
import copy


EPS = 1e-10

class GCN_1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(GCN_1, self).__init__()
        self.weight_1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight_1 = self.reset_parameters(self.weight_1)
        if dropout:
            self.gc_drop = nn.Dropout(dropout)
        else:
            self.gc_drop = lambda x: x
        self.bc = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def forward(self, feat, adj, nlayer, sampler=False):

        if not sampler:
            z = self.gc_drop(torch.mm(feat.to(torch.float32), self.weight_1.to(torch.float32)))
            z = torch.mm(adj, z)
            for i in range(1, nlayer):
                z = torch.mm(adj, z)

            outputs = F.normalize(z, dim=1)

            return outputs
        else:
            z_ = self.gc_drop(torch.mm(feat, self.weight_1))
            z = torch.mm(adj, z_)
            for i in range(1, nlayer):
                z = torch.mm(adj, z)

            return z


class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, nlayers, sparse = False):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()

        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 1):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 1):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.sparse = sparse

    def forward(self, x, Adj):
        # Adj = F.dropout(Adj, p=self.dropout, training=self.training)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn_encoder_layers[-1](x, Adj)
        return x


class Attention_shared(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.1):
        super(Attention_shared, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att_l = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)

        self.att_h = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds_l, embeds_h):
        beta = []
        attn_l = self.attn_drop(self.att_l)
        attn_h = self.attn_drop(self.att_h)
        for embed in embeds_l:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_l.matmul(sp.t()))
        for embed in embeds_h:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_h.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        print("Beta (modality weights):", beta)
        z_fusion = 0

        embeds = embeds_l + embeds_h
        for i in range(len(embeds)):
            z_fusion += embeds[i]*beta[i]

        return F.normalize(z_fusion, dim=1, p=2)


def normalize_adj_dgl(rows, cols, values, num_nodes):

    degree = torch.zeros(num_nodes, device=values.device)
    degree.index_add_(0, rows, values)
    degree.index_add_(0, cols, values)

    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    values = values * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    return values


def normalize_laplacian_dgl(rows, cols, values, num_nodes):
    degree = torch.zeros(num_nodes, device=values.device)
    degree.index_add_(0, rows, values)
    degree.index_add_(0, cols, values)

    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    values = values * d_inv_sqrt[rows] * d_inv_sqrt[cols]

    laplacian_values = -values
    laplacian_rows = rows
    laplacian_cols = cols

    diag_indices = torch.arange(num_nodes, device=values.device)
    diag_values = torch.ones([num_nodes], device=values.device)

    laplacian_rows = torch.cat([laplacian_rows, diag_indices])
    laplacian_cols = torch.cat([laplacian_cols, diag_indices])
    laplacian_values = torch.cat([laplacian_values, diag_values])

    return laplacian_rows, laplacian_cols, laplacian_values



def to_dgl_adj(sim_matrix):
    rows, cols = sim_matrix.nonzero(as_tuple=True)
    values = sim_matrix[rows, cols]
    values = normalize_adj_dgl(rows, cols, values, sim_matrix.shape[0])
    g = dgl.graph((rows, cols), num_nodes=sim_matrix.shape[0], device='cuda')
    g.edata['w'] = values

    return g

def to_dgl_L(sim_matrix):
    rows, cols = sim_matrix.nonzero(as_tuple=True)
    values = sim_matrix[rows, cols]
    rows, cols, values = normalize_laplacian_dgl(rows, cols, values, sim_matrix.shape[0])
    g = dgl.graph((rows, cols), num_nodes=sim_matrix.shape[0], device='cuda')
    g.edata['w'] = values

    return g


class FusionRepresentation(nn.Module):
    def __init__(self):
        super(FusionRepresentation, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, z1, z2):
        a = torch.sigmoid(self.a)
        return (1 - a) * z1 + a * z2
