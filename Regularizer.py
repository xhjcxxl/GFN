# -*- coding: utf-8 -*-
# @Time    : 2020-03-10 09:32
# @Author  : dai yong
# @File    : Regularizer.py


import torch


class regu_loss(object):
    def __init__(self, edges, edges_weights, edges_mappings, alpha = 0.001, beta = 0.001, gamma = 0.1):

        self.edges = edges
        self.edges_weights = edges_weights
        self.edges_mappings = edges_mappings
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def cal_regu_loss(self, ori_edge_embeddings, adapted_edge_embeddings):
        if self.edges in [0, 1, 2, 3, 4, 6]:
            re_loss = self.alpha * torch.norm(adapted_edge_embeddings) \
                      + self.beta * torch.norm(ori_edge_embeddings - adapted_edge_embeddings)
            return re_loss * self.gamma
        else:
            print('Regularization error')
