import torch
import time
from utils.metrics import *
from model_1 import *
from utils.graph_utils import *
from sklearn.cluster import KMeans
from params import *
import random
from dataloader import *
import datetime
from Encoder_1 import *
import torch.nn.functional as F
import pdb    
import scipy.io as scio
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score
import warnings
import dgl.sparse as dglsp

# args = set_params()
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed = 0


def train():
    # hyperparameters for ele-fashion
    hidden_dim = 256
    embed_dim = 256
    nlayer = 1
    tau = 1.
    beta = 0.1

    dropout = 0.5
    weight_decay = 0.000
    lr = 0.002
    epoches = 20

    AGGlayer = 1
    alpha = 0.8
    contrast_batch_size = 3000

    r = 0.5

    # Encoder
    k_l = 15
    k_h = 2
    fast_knn_b = 1500
    num_anchor_per_class = 700
    sparse = True

    dataset = 'ele-fashion'
    f_list, views, labels, nb_classes, feat_dim, feat_o = load_data(dataset, AGGlayer, alpha)

    print('Dataset:', dataset)
    print('Number of classes:', nb_classes)
    print('Number of features:', feat_dim)
    print('Number of views:', views)

    if torch.cuda.is_available():
        print('Using CUDA')
        f_list = [f_list[i].cuda() for i in range(len(f_list))]

    model = Model(views, feat_dim, hidden_dim, embed_dim, nb_classes, tau, beta, dropout, nlayer, sparse,
                  contrast_batch_size, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        labels.cuda()

    normalized_f_list = [F.normalize(f, p=2, dim=1) for f in f_list]
    f = r*normalized_f_list[0] + (1-r)*normalized_f_list[0]
    estimator = KMeans(n_clusters=nb_classes, random_state=0).fit(f.cpu().numpy())
    pseudo_label = torch.tensor(estimator.labels_, device=device)
   
    anchor_indices = []
    for i in range(nb_classes):
        class_indices = torch.where(pseudo_label == i)[0]
        class_features = f[class_indices]
        center = torch.tensor(estimator.cluster_centers_[i], device=device)
        distances = torch.norm(class_features - center, dim=1)
        _, nearest_indices = torch.topk(distances, num_anchor_per_class, largest=False)
        anchor_indices.append(class_indices[nearest_indices])

    anchor_indices = torch.cat(anchor_indices)

    sim_b = torch.mm(f, f[anchor_indices].t())
    row_min = sim_b.min(dim=1, keepdim=True)[0]
    row_max = sim_b.max(dim=1, keepdim=True)[0]

    sim_b = (sim_b - row_min) / (row_max - row_min + EPS) + EPS

    adj_l = generate_simple_scalable_graph(f, sim_b, k_l, k_h, pseudo_label, fast_knn_b, anchor_indices)

    Ls = []
    for f in normalized_f_list:
        sim_b = torch.mm(f, f[anchor_indices].t())
        row_min = sim_b.min(dim=1, keepdim=True)[0]
        row_max = sim_b.max(dim=1, keepdim=True)[0]
        sim_b = (sim_b - row_min) / (row_max - row_min + EPS) + EPS
        L = generate_heterophily_graph_with_anchor(sim_b, k_h, pseudo_label, anchor_indices)
        L_h = sparse_tensor_to_dgl(L)
        Ls.append(L_h)

    adj_l = adj_l.adj(scipy_fmt="coo")
    indices1 = torch.tensor([adj_l.row, adj_l.col])
    values1 = torch.tensor(adj_l.data, dtype=torch.float32)
    adj_l = torch.sparse_coo_tensor(indices1, values1, adj_l.shape).to(f_list[0].device)
    L_hs = []
    for L in Ls:
        L_h = L.adj(scipy_fmt="coo")
        indices2 = torch.tensor([L_h.row, L_h.col])
        values2 = torch.tensor(L_h.data, dtype=torch.float32)
        L = torch.sparse_coo_tensor(indices2, values2, L_h.shape).to(f_list[0].device)
        L_hs.append(L)

    
    cnt_wait = 0
    best = 1e9
    best_epoch = 0
    patience = 50

    start_time = time.time()
    fh = open("result_" + dataset + "_NMI&ARI.txt", "a") 
    fh.write(
        'r=%f, seed = %d, AGGlayer = %d, alpha = %f, hidden_dim = %d, embed_dim = %d, tau = %f, dropout = %f, nlayer = %d, lr = %f, weight_decay = %f, epoches = %d'
        % (r, seed, AGGlayer, alpha, hidden_dim, embed_dim, tau, dropout, nlayer, lr, weight_decay, epoches))
    fh.write('\r\n')
    fh.write('Encoder params: k_l = %d, k_h = %d,sparse = %s' % (k_l, k_h, sparse))
    fh.write('\r\n')
    fh.flush()
    fh.close()
    for epoch in range(epoches):
        model.train()
        loss, _ = model(f_list, adj_l, L_hs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch)
        print('Total loss: ', loss.item())

        if (epoch + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"Time elapsed after {epoch + 1} epochs: {elapsed_time:.2f} secondes")



        if best > loss.item():
            best = loss.item()
            cnt_wait = 0
            best_epoch = epoch
            torch.save(model.state_dict(), './checkpoint/' + dataset + '/best_' + str(seed) + '.pth')
        else:
            cnt_wait += 1
        if cnt_wait >= patience:
            break

    model.load_state_dict(torch.load('./checkpoint/' + dataset + '/best_' + str(seed) + '.pth'))

    model.cuda()
    epoch = best_epoch
    print("---------------------------------------------------")
    model.eval()
    embeds = model.get_embeds(f_list, adj_l, L_hs).cpu().numpy()

    estimator = KMeans(n_clusters=nb_classes)
    ACC_list = []
    F1_list = []
    NMI_list = []
    ARI_list = []
    for _ in range(10):
        estimator.fit(embeds)
        y_pred = estimator.predict(embeds)
        cm = clustering_metrics(labels.cpu().numpy(), y_pred)
        ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()

        ACC_list.append(ac)
        F1_list.append(f1)
        NMI_list.append(nm)
        ARI_list.append(ari)
    acc = sum(ACC_list) / len(ACC_list)
    f1 = sum(F1_list) / len(F1_list)
    ari = sum(ARI_list) / len(ARI_list)
    nmi = sum(NMI_list) / len(NMI_list)

    print('\t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc * 100, 2),
                                                                                         np.round(f1 * 100, 2),
                                                                                         np.round(nmi * 100, 2),
                                                                                         np.round(ari * 100, 2)))
    fh = open("result_" + dataset + "_NMI&ARI.txt", "a")
    fh.write(
        'ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1, nmi, ari))
    fh.write('\r\n')
    fh.write(
        'seed = %d, AGGlayer = %d, alpha = %f, hidden_dim = %d, embed_dim = %d, tau = %f, dropout = %f, nlayer = %d, lr = %f, weight_decay = %f, epoches = %d'
        % (seed, AGGlayer, alpha, hidden_dim, embed_dim, tau, dropout, nlayer, lr, weight_decay, epoches))
    fh.write('\r\n')
    fh.write('Encoder params: k_l = %d, k_h = %d,sparse = %s' % (k_l, k_h, sparse))
    fh.write('\r\n')
    fh.write('---------------------------------------------------------------------------------------------------')
    fh.write('\r\n')
    fh.flush()
    fh.close()


if __name__ == '__main__':
    set_seed(seed)
    train()




