import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres

EPS = 1e-10

def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')

def cal_similarity_graph(embeddings):
    similarity_graph = torch.mm(embeddings, embeddings.t())
    return similarity_graph

def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

def normalize_adj_from_tensor(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EPS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EPS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EPS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def Laplace_h(sim_h):
    sim_h = normalize_adj_from_tensor(sim_h, mode='sym')
    L = (torch.eye(sim_h.shape[0], sim_h.shape[1]).to(sim_h.device) - sim_h).to(sim_h.device)
    return L


def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        non_diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, non_diag_index], adj.values()[non_diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def remove_self_loop_dense(tensor):
    assert tensor.shape[0] == tensor.shape[1]
    mask = 1 - torch.eye(tensor.shape[0], device=tensor.device, dtype=tensor.dtype)

    return tensor * mask

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new.coalesce()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return torch.sparse.FloatTensor(index, torch.ones(len(index[0])).to(adj.device), adj.shape).coalesce()


def generate_simple_graph(sim, k_l, k_h):
    sim_l = top_k(sim, k_l)
    sim_h = top_k(1 - sim, k_h)

    sim_l = symmetrize(sim_l)
    sim_h = symmetrize(sim_h)

    adj_l = adj_values_one(sim_l.to_sparse()).to_dense()
    adj_h = adj_values_one(sim_h.to_sparse()).to_dense()

    adj_l = normalize_adj_from_tensor(adj_l, mode='sym')
    L_h = Laplace_h(adj_h)

    return adj_l, L_h



def sim_product(f_list):
    adjs = []
    for i in range(len(f_list)-1):
        adj = f_list[i].to_dense()
        adj = normalize_adj_from_tensor(adj, 'row', sparse=False)

        adjs.append(adj)

    adjs_rw = torch.stack(adjs, dim=0).mean(dim=0)
    adj_norm = torch.norm(adjs_rw, dim=1, keepdim=True)
    zero_indices = torch.nonzero(adj_norm.flatten() == 0)
    adj_norm[zero_indices] += EPS
    adj_sim = torch.mm(adjs_rw, adjs_rw.t()) / torch.mm(adj_norm, adj_norm.t())

    return adj_sim


def symmetrize_sparse(sparse_matrix):

    indices = sparse_matrix._indices()
    values = sparse_matrix._values()
    sym_indices = torch.cat([indices, indices[[1, 0], :]], dim=1)
    sym_values = torch.cat([values, values], dim=0)
    return torch.sparse.FloatTensor(sym_indices, sym_values, sparse_matrix.size()).coalesce()


def sparse_tensor_to_dgl(sparse_matrix):
    sparse_matrix = sparse_matrix.coalesce()
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()
    rows, cols = indices[0], indices[1]

    graph = dgl.graph((rows, cols), num_nodes=sparse_matrix.size(0), device=sparse_matrix.device)
    graph.edata['w'] = values
    return graph

def sparse_identity_matrix(size, device):
    indices = torch.stack([torch.arange(size), torch.arange(size)], dim=0).to(device)
    values = torch.ones(size).to(device)
    return torch.sparse.FloatTensor(indices, values, torch.Size([size, size])).coalesce()

def generate_simple_scalable_graph(feat, sim_b, k_l, k_h, pseudo_label, fast_knn_b, anchor_indices):

    adj_l = generate_homophily_graph_fast_knn(feat, k_l, pseudo_label, fast_knn_b)
    # L_h = generate_heterophily_graph_with_anchor(sim_b, k_h, pseudo_label, anchor_indices)
    adj_l = sparse_tensor_to_dgl(adj_l)
    # L_h = sparse_tensor_to_dgl(L_h)
    return adj_l

    return adj_l, L_h

def generate_homophily_graph_fast_knn(feat, k, pseudo_label, b):
    pseudo_label = pseudo_label.to(feat.device)
    unique_labels = torch.unique(pseudo_label)
    all_rows = []
    all_cols = []

    for label in unique_labels:
        label_indices = (pseudo_label == label).nonzero().squeeze()
        label_feat = feat[label_indices]
        if k >= len(label_indices):
            rows, cols, values = knn_fast(label_feat, len(label_indices) - 1, b)
        else:
            rows, cols, values = knn_fast(label_feat, k, b)
        all_rows.append(label_indices[rows])
        all_cols.append(label_indices[cols])

    all_rows = torch.cat(all_rows).to(feat.device)
    all_cols = torch.cat(all_cols).to(feat.device)
    all_values = torch.ones_like(all_rows).to(feat.device)

    adj = torch.sparse_coo_tensor(torch.stack([all_rows, all_cols]), all_values, (feat.shape[0], feat.shape[0])).coalesce().to(feat.device)
    adj_l = normalize_adj_from_tensor(adj, mode='sym', sparse=True)
    return adj_l


def generate_heterophily_graph_with_anchor(sim_b, k_h, pseudo_label, anchor_indices):
    pseudo_label = pseudo_label.to(sim_b.device)

    sim_h = 1. - sim_b

    sim_h_filtered = top_k(sim_h, k_h) # sim_h_filtered

    non_zero_mask = sim_h_filtered > 0
    row_indices, col_indices = non_zero_mask.nonzero(as_tuple=True)
    anchor_col_indices = anchor_indices[col_indices]
    adj_h_indices = torch.stack([row_indices, anchor_col_indices], dim=0).to(sim_b.device)

    print('finish anchor')

    adj_h_values = torch.ones(len(adj_h_indices[0])).to(sim_b.device)
    adj_h = torch.sparse.FloatTensor(adj_h_indices, adj_h_values, torch.Size([sim_h_filtered.shape[0], sim_h_filtered.shape[0]])).coalesce().to(sim_b.device)
    adj_h = symmetrize_sparse(adj_h)
    adj_h = adj_values_one(adj_h)

    adj_h = normalize_adj_from_tensor(adj_h, mode='sym', sparse=True)
    I_sparse = sparse_identity_matrix(adj_h.shape[0], adj_h.device)
    L_h = I_sparse - adj_h

    return L_h




