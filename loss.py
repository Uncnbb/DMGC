import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from kmeans_pytorch import kmeans
from itertools import combinations
from math import comb

EPS = 1e-15


class Contrast(nn.Module):
    def __init__(self, hidden_dim, project_dim, tau):
        super(Contrast, self).__init__()
        self.tau = tau
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
        for model in self.proj_2:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_1, z_2, pos):
        z_proj_1 = self.proj_1(z_1)
        z_proj_2 = self.proj_2(z_2)
        matrix_1 = self.sim(z_proj_1, z_proj_2)
        matrix_2 = matrix_1.t()


        matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EPS)
        lori_1 = -torch.log(matrix_1.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EPS)
        lori_2 = -torch.log(matrix_2.mul(pos.to_dense()).sum(dim=-1)).mean()


        return (lori_1 + lori_2) / 2


class Contrast_mi:
    def __init__(self, tau):
        super(Contrast_mi, self).__init__()
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def compute_loss(self, z_1, z_2, pos):

        matrix_1 = self.sim(z_1, z_2)
        matrix_2 = matrix_1.t()

        matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EPS)
        lori_1 = -torch.log(matrix_1.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EPS)
        lori_2 = -torch.log(matrix_2.mul(pos.to_dense()).sum(dim=-1)).mean()

        return (lori_1 + lori_2) / 2


def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()

    return 10 * loss



def target_distribution(q) :
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def soft_assignment(device, embeddings, n_clusters, cluster_index, alpha=1):

    cluster_centers = [embeddings[cluster_index_].mean(dim=0) for cluster_index_ in cluster_index]
    cluster_centers = torch.stack(cluster_centers, dim=0)

    cluster_layer = Parameter(torch.Tensor(n_clusters, 64))

    cluster_layer.data = torch.tensor(cluster_centers).to(device)
    q = 1.0 / (1.0 + torch.sum(torch.pow(embeddings.unsqueeze(1) - cluster_layer, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return q


def KL_clustering(device, z, num_classes, z_lps, z_hps):

    z = F.normalize(z, dim=1, p=2)
    z_lps = [F.normalize(m, dim=1, p=2) for m in z_lps]
    z_hps = [F.normalize(m, dim=1, p=2) for m in z_hps]
    pseudo_label, _ = kmeans(X=z, num_clusters=num_classes, distance='euclidean', device=device)
    cluster_index = [torch.nonzero(pseudo_label == j).flatten() for j in range(int(pseudo_label.max()))]
    q = soft_assignment(device, z, num_classes, cluster_index, alpha=1)
    p = target_distribution(q)
    clu_loss = F.kl_div(q.log(), p, reduction='mean')



    return clu_loss, pseudo_label

def off_diagonal(x):

    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def BarlowTwins_multi(device, z_list, Batch_size, n_z):
    bn = nn.BatchNorm1d(n_z, affine=False).to(device)
    Bl = torch.tensor([0]).to(device).float()
    num_view = 2
    V = range(num_view)

    if num_view != 0:
        for combn in combinations(V, 2):
            c = bn(z_list[combn[0]]).t().mm(bn(z_list[combn[1]]))
            c.div_(Batch_size)

            lambd = 0.0051
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            Bl += on_diag + lambd * off_diag

        Bl = Bl / comb(num_view, 2)

    return Bl / 10



