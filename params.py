import argparse
import torch
import sys
dataset = ''

def acm_4019_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm-4019")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # The parameters of learning process
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epoches', type=int, default=40)

    # data
    parser.add_argument('--AGGlayer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)

    # GSL
    parser.add_argument('--k_l', type=int, default=10)
    parser.add_argument('--k_h', type=int, default=4)
    parser.add_argument('--sparse', default=False)

    args, _ = parser.parse_known_args()
    return args

def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0) # 0

    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # The parameters of learning process
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epoches', type=int, default=70)

    # data
    parser.add_argument('--AGGlayer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)

    # GSL
    parser.add_argument('--k_l', type=int, default=15)
    parser.add_argument('--k_h', type=int, default=5)
    parser.add_argument('--sparse', default=False)

    args, _ = parser.parse_known_args()
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # The parameters of learning process
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoches', type=int, default=80)

    # data
    parser.add_argument('--AGGlayer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)

    # GSL
    parser.add_argument('--k_l', type=int, default=10)
    parser.add_argument('--k_h', type=int, default=3)
    parser.add_argument('--sparse', default=False)

    args, _ = parser.parse_known_args()
    return args

def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--nlayer', type=int, default=5)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # The parameters of learning process
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epoches', type=int, default=90)

    # data
    parser.add_argument('--AGGlayer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--r', type=float, default=0.5)

    # GSL
    parser.add_argument('--k_l', type=int, default=15)
    parser.add_argument('--k_h', type=int, default=2)
    parser.add_argument('--sparse', default=False)

    args, _ = parser.parse_known_args()
    return args

def amazon_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="amazon")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # The parameters of learning process
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epoches', type=int, default=160)

    # data
    parser.add_argument('--AGGlayer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--r', type=float, default=1)

    # GSL
    parser.add_argument('--k_l', type=int, default=10)
    parser.add_argument('--k_h', type=int, default=6)
    parser.add_argument('--sparse', default=False)

    args, _ = parser.parse_known_args()
    return args

