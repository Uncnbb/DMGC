import torch
from Encoder_1 import *
import torch.nn as nn
import numpy as np
from loss import *
import scipy.sparse as sp
from utils.graph_utils import *
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, views, feats_dim, hidden_dim, embed_dim, num_clusters, tau, beta, dropout, nlayer, sparse, device):
        super(Model, self).__init__()
        self.views = views
        self.feats_dim = feats_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_clusters = num_clusters
        self.nlayer = nlayer
        self.tau = tau
        self.beta = beta
        self.encoder = GraphEncoder(hidden_dim, hidden_dim, dropout, self.nlayer, sparse=sparse)
        self.att = Attention_shared(hidden_dim)
        self.decoder = nn.Sequential(
                                     nn.Linear(2*hidden_dim, embed_dim),
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

        self.multi_view_projections = nn.ModuleList([nn.Sequential(
        nn.Linear(int(feats_dim), hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        )for _ in range(views)])
        self.fusion_1 = FusionRepresentation()
        self.fusion_2 = FusionRepresentation()



    def forward(self, f_list, adj_l, L_h):


        z_list = []

        for i in range(self.views):

            z_list.append((self.multi_view_projections[i](f_list[i])))

        pos = torch.eye(len(f_list[0])).to_sparse().to(self.device)

        z_lps = []
        z_hps = []

        for i in range(self.views):# self.views-1,

            z_l = self.encoder(z_list[i], adj_l)
            z_h = self.encoder(z_list[i], L_h)
            z_lps.append(z_l)
            z_hps.append(z_h)

        z_mean_l = torch.stack(z_lps, dim=0).mean(dim=0)
        z_mean_h = torch.stack(z_hps, dim=0).mean(dim=0)

        z = self.att([z_mean_l], [z_mean_h])


        loss_rec = 0
        loss_l = 0


        for i in range(self.views):

            fea_rec = self.decoder(torch.cat((z_lps[i],z_hps[i]),dim=-1))
            loss_rec += sce_loss(fea_rec, f_list[i])

        loss_rec = loss_rec / len(f_list)

        for i in range(self.views):

            z_lp = z_lps[i]
            z_hp = z_hps[i]

            loss_l += self.contrast_l(z_lp, z, pos) + self.contrast_h(z_hp, z, pos)

        loss_clu, pseudo_label = KL_clustering(self.device, z, self.num_clusters, z_lps, z_hps)


        loss = loss_rec + loss_clu + loss_l
        return loss, pseudo_label


    def get_embeds(self, f_list, adj_l, L_h):

        z_list = []

        for i in range(self.views):

            z_list.append((self.multi_view_projections[i](f_list[i])))

        z_lps = []
        z_hps = []

        for i in range(self.views):

            z_l = self.encoder(z_list[i], adj_l)
            z_h = self.encoder(z_list[i], L_h)
            z_lps.append(z_l)
            z_hps.append(z_h)

        z_mean_l = torch.stack(z_lps, dim=0).mean(dim=0)
        z_mean_h = torch.stack(z_hps, dim=0).mean(dim=0)

        z = self.att([z_mean_l], [z_mean_h])

        return z.detach()