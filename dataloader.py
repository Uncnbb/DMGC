import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops
import torch as th
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import torch.nn.functional as F
import json
from nc_dataset import NodeClassificationDataset
import os
from utils.graph_utils import *
import scipy.io as sio
import warnings
from scipy.sparse import csr_matrix
warnings.filterwarnings("ignore")


def APPNP(h_list, adjs_o, nlayer, alpha):
    f_list = []
    for i in range(len(adjs_o)):
        h_0 = h_list[i]
        z = h_list[i]

        adj = adjs_o[i]
        for i in range(nlayer):
            z = torch.sparse.mm(adj, z)
            z = (1 - alpha) * z + alpha * h_0
        #
        f_list.append(z)

    return f_list

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

def preprocess_features_1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def preprocess_features_dense(features):
    """Row-normalize feature matrix for dense np.array."""
    rowsum = np.sum(features, axis=1, keepdims=True)  # 按行求和
    r_inv = np.power(rowsum, -1)  # 求逆
    r_inv[np.isinf(r_inv)] = 0.  # 防止除以零的情况
    features = features * r_inv  # 进行行归一化
    return features

def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, diag_index], adj.values()[diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    rowsum = features.sum(dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1,1)
    features = features * r_inv
    return features

def load_douban():
    path = "data/douban/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_t = np.load(path + 'features_douban_928_1_movie.npz')
    feat_m = np.load(path + 'img_feature_douban_512_199.npz')['feature']

    mam = csr_matrix(np.load(path + 'doubanMAM.npy'))
    mdm = csr_matrix(np.load(path +'doubanMDM.npy'))
    adjs = [mam, mdm]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    label = th.FloatTensor(label)
    label = torch.argmax(label, dim=1)

    feat_t = th.FloatTensor(preprocess_features_dense(feat_t))
    feat_m = th.FloatTensor(preprocess_features_dense(feat_m))

    nb_classes = int(label.max() + 1)

    return feat_t, feat_m, label, nb_classes, adjs

def load_imdb():
    path = "data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_t = sp.load_npz('data/imdb/features_0.npz')
    feat_m = np.load('data/imdb/img_feature_20_6_25_199.npz')['feature']

    adj1 = csr_matrix(np.load(path + 'imdb_adj1.npy'))
    adj2 = csr_matrix(np.load(path + 'imdb_adj2.npy'))
    adjs = [adj1, adj2]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    label = th.FloatTensor(label)
    label = torch.argmax(label, dim=1)

    feat_t = th.FloatTensor(preprocess_features_1(feat_t))
    feat_m = th.FloatTensor(preprocess_features_dense(feat_m))

    nb_classes = int(label.max() + 1)

    return feat_t, feat_m, label, nb_classes, adjs

def load_acm_4019():

    path = "data/acm-4019/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    adjs = [pap, psp]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    label = th.FloatTensor(label)
    label = torch.argmax(label,dim=1)

    feat_p = th.FloatTensor(preprocess_features_1(feat_p))

    nb_classes = int(label.max() + 1)

    return feat_p, label, nb_classes, adjs

def load_dblp():
    path = "data/dblp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    return feat, label, nclasses, adjs
def load_yelp():
    path = "./data/yelp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = F.normalize(feat, p=2, dim=1)
    adjs = torch.load(path+'adj.pt')

    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    return feat, label, nclasses, adjs

def load_amazon():
    path = "data/amazon/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_t = np.load('data/amazon/feature_item_amazon.npz')['feature']
    feat_m = np.load('data/amazon/img_feature_20_1_25_199.npz')['feature']

    # noise_scale = 0.2
    # std = np.std(feat_t)
    # noise = np.random.normal(0, noise_scale * std, feat_m.shape)
    # feat_t = feat_t + noise
    #
    # std = np.std(feat_m)
    # noise = np.random.normal(0, noise_scale * std, feat_m.shape)
    # feat_m = feat_m + noise

    adj1 = sp.load_npz(path + 'ii_adj_matrix.npz')
    adj2 = sp.load_npz(path + 'iu_aat_matrix.npz')
    adjs = [adj1, adj2]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize_adj_from_tensor(adj, mode='sym') for adj in adjs]

    label = th.FloatTensor(label)
    label = torch.argmax(label, dim=1)

    feat_t = th.FloatTensor(preprocess_features_dense(feat_t))
    feat_m = th.FloatTensor(preprocess_features_dense(feat_m))

    nb_classes = int(label.max() + 1)

    return feat_t, feat_m, label, nb_classes, adjs


def load_ele_fashion():
    data_path = './data'  # replace this with the path where you save the datasets
    dataset_name = 'ele-fashion'
    feat_name = 't5vit'
    verbose = True
    device = 'cpu'  # use 'cuda' if GPU is available

    dataset = NodeClassificationDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        verbose=verbose,
        device=device
    )

    graph = dataset.graph

    feat = graph.ndata['feat']
    label = graph.ndata['label']

    # Reassign labels to remove label 5, because it is empty
    label = torch.where(label > 5, label - 1, label)
    nclasses = 11  # Total number of classes after removing label 5


    v_feat = feat[:, int(feat.shape[1] / 2):]
    t_feat = feat[:, :int(feat.shape[1] / 2)]
    graph = dgl.remove_self_loop(graph)
    graph, _ = to_bidirected_with_reverse_mapping(graph)
    adj = graph.adj(scipy_fmt='coo')
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([adj.row, adj.col]),
        values=torch.ones(adj.nnz),
        size=(graph.num_nodes(), graph.num_nodes())
    )
    adj = sparse_tensor_add_self_loop(adj)

    print(feat.shape,label.shape)
    print(label.unique())
    return [t_feat, v_feat], label, nclasses, [adj]


def load_data(dataset, AGGlayer, alpha):

    if dataset == 'amazon':
        feat_t, feat_m, label, nb_classes, adjs = load_amazon()
        feat_t = APPNP([feat_t for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        feat_m = APPNP([feat_m for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        f = [feat_m, feat_t]
        feat_dim = f[0][0].shape[1]
        modals = len(f)
        return f, modals, label, nb_classes, feat_dim, adjs

    elif dataset == 'imdb':
        feat_t, feat_m, label, nb_classes, adjs = load_imdb()
        feat_t = APPNP([feat_t for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        feat_m = APPNP([feat_m for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        f = [feat_m, feat_t]
        feat_dim = f[0][0].shape[1]
        modals = len(f)
        return f, modals, label, nb_classes, feat_dim, adjs

    elif dataset == 'dblp':
        feat_o, label, nclasses, adjs = load_dblp()
        feats = APPNP([feat_o for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        views = len(adjs)
        feats = [F.normalize(feats[i]) for i in range(len(adjs))]
        feat_dim = feats[0].shape[1]
        return feats, views, label, nclasses, feat_dim, [feat_o], adjs

    elif dataset == 'yelp':
        feat_o, label, nclasses, adjs = load_yelp()
        feats = APPNP([feat_o for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        views = len(adjs)
        feats = [F.normalize(feats[i]) for i in range(len(adjs))]
        feat_dim = feats[0].shape[1]
        return feats, views, label, nclasses, feat_dim, [feat_o], adjs

    elif dataset == 'acm-4019':
        feat_o, label, nclasses, adjs = load_acm_4019()
        feats = APPNP([feat_o for _ in range(len(adjs))], adjs, AGGlayer, alpha)
        views = len(adjs)
        feats = [F.normalize(feats[i]) for i in range(len(adjs))]
        feat_dim = feats[0].shape[1]
        return feats, views, label, nclasses, feat_dim, [feat_o], adjs

    elif dataset == 'ele-fashion':
        feat_list, label, nclasses, adjs = load_ele_fashion()
        feats = APPNP(feat_list, [adjs[0] for _ in range(len(feat_list))], AGGlayer, alpha)
        feats = [F.normalize(feats[i]) for i in range(len(feats))]
        views = len(feat_list)
        feat_dim = feats[0].shape[1]
        return feats, views, label, nclasses, feat_dim, feat_list