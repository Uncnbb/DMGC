import torch
from Encoder_1 import *
import torch.nn as nn
import numpy as np
from loss import *
import scipy.sparse as sp
from utils.graph_utils import *
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, modals, feats_dim, hidden_dim, embed_dim, num_clusters, tau, beta, dropout, nlayer, sparse,
                 device):
        super(Model, self).__init__()
        self.modals = modals
        self.feats_dim = feats_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_clusters = num_clusters
        self.nlayer = nlayer
        self.tau = tau
        self.beta = beta
        self.encoder = GraphEncoder(hidden_dim, hidden_dim, dropout, self.nlayer, sparse=sparse)
        self.att = Attention_shared(hidden_dim)
        self.att_t = Attention_shared(hidden_dim)
        self.att_m = Attention_shared(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, feats_dim)
        )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        self.contrast_l = Contrast(hidden_dim, hidden_dim, self.tau)
        self.contrast_h = Contrast(hidden_dim, hidden_dim, self.tau)
        self.contrast = Contrast(hidden_dim, hidden_dim, self.tau)
        self.device = device
        self.fusion_1 = FusionRepresentation()
        self.fusion_2 = FusionRepresentation()

        self.multi_view_projections = nn.ModuleList([nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(int(feats_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
        ) for _ in range(modals)])

        self.multi_view_projections_t = nn.ModuleList([nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(int(feats_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
        ) for _ in range(modals)])

        self.multi_view_projections_m = nn.ModuleList([nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(int(feats_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
        ) for _ in range(modals)])



    def forward(self, f_list, adj_l, L_hs):

        lamb = 0 # 0, 0.001, 0.1, 1, 10
        miu = 0

        t_list = f_list[0]
        m_list = f_list[1]


        zt_list = []
        zm_list = []

        for i in range(len(t_list)):
            zt_list.append(self.multi_view_projections_t[i](t_list[i]))
            zm_list.append(self.multi_view_projections_m[i](m_list[i]))

        z_list = [zt_list, zm_list]

        pos = torch.eye(len(t_list[0])).to_sparse().to(self.device)

        z_lps = []
        z_hps = []

        for i in range(self.modals):
            z_l_sublist = []
            z_h_sublist = []

            for j in range(len(z_list[0])):
                z_l = self.encoder(z_list[i][j], adj_l)
                z_h = self.encoder(z_list[i][j], L_hs[i])

                z_l_sublist.append(z_l)
                z_h_sublist.append(z_h)

            z_lps.append(z_l_sublist)
            z_hps.append(z_h_sublist)

        zt_mean_l = torch.stack(z_lps[0], dim=0).mean(dim=0)
        zm_mean_l = torch.stack(z_lps[1], dim=0).mean(dim=0)

        zt_mean_h = torch.stack(z_hps[0], dim=0).mean(dim=0)
        zm_mean_h = torch.stack(z_hps[1], dim=0).mean(dim=0)

        z_t = self.fusion_1(zt_mean_l, zt_mean_h)
        z_m = self.fusion_2(zm_mean_l, zm_mean_h)

        z = self.att([z_t], [z_m])

        loss_rec = 0
        loss_l = 0
        loss_c = 0
        loss_con = 0
        loss_h = 0

        for i in range(self.modals):
            z_cat = torch.cat([torch.stack(z_lps[i], dim=0), torch.stack(z_hps[i], dim=0)], dim=-1)  # 拼接
            fea_rec = self.decoder(z_cat)

            loss_rec += sce_loss(fea_rec, torch.stack(f_list[i], dim=0))

        loss_rec = loss_rec / (len(f_list[0])*2)


        z_lps = [zt_mean_l, zm_mean_l]
        z_hps = [zt_mean_h, zm_mean_h]
        for i in range(self.modals):
            z_lp = z_lps[i]
            z_hp = z_hps[i]

            loss_l += self.contrast_l(z_lp, z_t, pos) + self.contrast_l(z_lp, z_m, pos)
            loss_h += self.contrast_h(z_hp, z_t, pos) + self.contrast_h(z_hp, z_m, pos)

        loss_con += self.contrast(z_m, z_t, pos)

        loss_clu, pseudo_label = KL_clustering(self.device, z, self.num_clusters, z_lps, z_hps)

        loss = loss_rec + loss_clu + lamb * (loss_l + loss_h) + miu * loss_con

        return loss

    def get_embeds(self, f_list, adj_l, L_hs):

        t_list = f_list[0]
        m_list = f_list[1]

        zt_list = []
        zm_list = []

        for i in range(len(t_list)):
            zt_list.append(self.multi_view_projections_t[i](t_list[i]))
            zm_list.append(self.multi_view_projections_m[i](m_list[i]))

        z_list = [zt_list, zm_list]

        z_lps = []
        z_hps = []

        for i in range(self.modals):
            z_l_sublist = []
            z_h_sublist = []

            for j in range(len(z_list[0])):
                z_l = self.encoder(z_list[i][j], adj_l)
                z_h = self.encoder(z_list[i][j], L_hs[i])

                z_l_sublist.append(z_l)
                z_h_sublist.append(z_h)

            z_lps.append(z_l_sublist)
            z_hps.append(z_h_sublist)

        zt_mean_l = torch.stack(z_lps[0], dim=0).mean(dim=0)
        zm_mean_l = torch.stack(z_lps[1], dim=0).mean(dim=0)

        zt_mean_h = torch.stack(z_hps[0], dim=0).mean(dim=0)
        zm_mean_h = torch.stack(z_hps[1], dim=0).mean(dim=0)

        z_t = self.fusion_1(zt_mean_l, zt_mean_h)
        z_m = self.fusion_2(zm_mean_l, zm_mean_h)

        z = self.att([z_t], [z_m])

        return z.detach()




