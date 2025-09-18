from utils.metrics import *
from model_mm import *
from utils.graph_utils import *
from sklearn.cluster import KMeans
from params import *
import random
from dataloader import *
import datetime
from Encoder_1 import *
import torch.nn.functional as F
import warnings

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

seed = 123


def train():
    # hyperparameters for multimodal multi-relational graphs referring to params.py
    hidden_dim = 128
    embed_dim = 128
    nlayer = 1
    tau = 1.
    beta = 0.1

    dropout = 0.5
    weight_decay = 0.
    lr = 0.003
    epoches = 160

    AGGlayer = 1
    alpha = 1
    r = 1

    # Encoder
    k_l = 10
    k_h = 6
    sparse = False

    dataset = 'amazon'
    f_list, modals, labels, nb_classes, feat_dim, adjs = load_data(dataset, AGGlayer, alpha)
    print(labels.shape)

    print('Dataset:', dataset)
    print('Number of classes:', nb_classes)
    print('Number of features:', feat_dim)
    print('Number of modals:', modals)

    model = Model(modals, feat_dim, hidden_dim, embed_dim, nb_classes, tau, beta, dropout, nlayer, sparse, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        f_list = [[tensor.cuda() for tensor in sublist] for sublist in f_list]
        model.cuda()
        labels.cuda()



    cnt_wait = 0
    best = 1e9
    best_epoch = 0
    patience = 50



    L_hs = []
    fs = []
    for f_mm in f_list:
        f = torch.cat(f_mm, dim=1)
        f = F.normalize(f, p=2, dim=1)
        fs.append(f)
    f = r*fs[0] + (1-r)*fs[1]
    sim_f = cal_similarity_graph(f)
    row_min = sim_f.min(dim=1, keepdim=True)[0]
    row_max = sim_f.max(dim=1, keepdim=True)[0]
    sim = (sim_f - row_min) / (row_max - row_min + EPS) + EPS
    adj_l, _ = generate_simple_graph(sim, k_l, k_h)  # self-loop
    for f in fs:
        sim_f = cal_similarity_graph(f)
        row_min = sim_f.min(dim=1, keepdim=True)[0]
        row_max = sim_f.max(dim=1, keepdim=True)[0]
        sim = (sim_f - row_min) / (row_max - row_min + EPS) + EPS
        _, L_h = generate_simple_graph(sim, k_l, k_h)
        L_hs.append(L_h)



    for epoch in range(epoches):
        model.train()
        loss = model(f_list, adj_l, L_hs)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Epoch:", epoch)
        print('Total loss: ', loss.item())


        if (epoch+1) % 10 == 0:
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

            print('z_t: \t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc * 100, 2),
                                                                                                 np.round(f1 * 100, 2),
                                                                                                 np.round(nmi * 100, 2),
                                                                                                 np.round(ari * 100, 2)))


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
    fh = open("result_" + dataset + ".txt", "a")
    fh.write(
        'ACC=%f, F1=%f, NMI=%f, ARI=%f\n' % (acc, f1, nmi, ari))
    fh.write('\r\n')
    fh.write('seed = %d, AGGlayer = %d, alpha = %f, hidden_dim = %d, embed_dim = %d, tau = %f, dropout = %f, nlayer = %d, lr = %f, weight_decay = %f, epoches = %d'
             % (seed, AGGlayer, alpha, hidden_dim, embed_dim, tau, dropout, nlayer, lr, weight_decay, epoches))
    fh.write('\r\n')
    fh.write('Encoder params: k_l = %d, k_h = %d' % (k_l, k_h))
    fh.write('\r\n')
    fh.write('---------------------------------------------------------------------------------------------------')
    fh.write('\r\n')
    fh.flush()
    fh.close()

if __name__ == '__main__':

    set_seed(seed)
    train()




