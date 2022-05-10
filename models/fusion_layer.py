# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : fusion layer.py
# @explanation : Fusion Layer Network


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Conv import Conv1d_no_relu
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from models.Conv import MemoryFusion


class TransEncoder(nn.Module):
    def __init__(self, in_dim, dataset, proj=0):
        super(TransEncoder, self).__init__()
        self.proj = proj
        self.max_len = {'r8': 100, 'r52': 100, 'oh': 160, 'mr': 50, 'sst5': 50, '20ng': 300}
        self.LayerNorm = nn.LayerNorm(in_dim)
        self.Dropout = nn.Dropout(p=0.3)
        self.position_embeddings = nn.Embedding(self.max_len[dataset], in_dim)
        self.layer = TransformerEncoderLayer(d_model=in_dim, nhead=2)
        self.layer.linear1 = nn.Linear(in_dim, 512)
        self.layer.linear2 = nn.Linear(512, in_dim)
        self.encoder = TransformerEncoder(self.layer, num_layers=2)
        self.llproj = nn.Linear(in_dim * 4, in_dim)

    def forward(self, doc_features):
        doc_features = self.LayerNorm(doc_features)
        doc_features = self.Dropout(doc_features)
        encoded = self.encoder(doc_features)
        # adopt different feature statistic
        if self.proj == 0:
            h_g_mean = torch.mean(encoded, dim=1)
            h_g_max = torch.max(encoded, dim=1)[0]
            doc_features = h_g_max + h_g_mean
        # projected to the corresponding dimension
        elif self.proj == 1:
            encoded = encoded.view(encoded.shape[0], -1)
            doc_features = self.llproj(encoded)
        return doc_features


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Fusion(nn.Module):
    def __init__(self, input_d, out_d):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(input_d, out_d, bias=True)

    def forward(self, pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features):
        doc_features = torch.cat((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 1)
        logits = self.linear(doc_features)
        return logits


class CombineFusion(nn.Module):
    def __init__(self, args, pmi_model, cos_model, euc_model, co_model, in_dim, out_dim):
        super(CombineFusion, self).__init__()
        self.fusion_style = args.fusion_style
        self.adopt_dropout = args.adopt_dropout
        self.pmi_model = pmi_model
        self.cos_model = cos_model
        self.euc_model = euc_model
        self.co_model = co_model
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.init_params()
        if self.fusion_style == 4:
            self.conv = Conv1d_no_relu(4, 12, [1])
        elif self.fusion_style == 5:
            self.ll = nn.Linear(in_dim * 3, in_dim)
        elif self.fusion_style == 6:
            self.transEncoder = TransEncoder(in_dim, args.dataset)
        elif self.fusion_style == 7:
            self.memnet = MemoryFusion(in_channel1=in_dim, in_channel2=in_dim, output_channel1=args.output_channel,
                                       kernel_size1=args.kernel_size, kernel_size2=args.kernel_size)

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, content, adj):
        pmi_doc_features = self.pmi_model(content, adj)
        cos_doc_features = self.cos_model(content, adj)
        euc_doc_features = self.euc_model(content, adj)
        co_doc_features = self.co_model(content, adj)

        if self.fusion_style == 0:
            doc_features = torch.cat((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 1)
        elif self.fusion_style == 1:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 0)
            doc_features = torch.mean(doc_features, dim=0)

        elif self.fusion_style == 2:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 0)
            doc_features = torch.max(doc_features, dim=0)[0]

        elif self.fusion_style == 3:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 0)
            doc_features = torch.sum(doc_features, dim=0)
        elif self.fusion_style == 4:
            doc_features = torch.stack([pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features], dim=1)
            fused_feature = self.conv(doc_features)
            fused_feature = fused_feature[0]
            fused_feature = torch.transpose(fused_feature, 1, 2)
            doc_features = F.max_pool1d(fused_feature, fused_feature.shape[2]).squeeze()
        elif self.fusion_style == 5:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 0)
            mean_features = torch.mean(doc_features, dim=0)
            max_features = torch.max(doc_features, dim=0)[0]
            sum_features = torch.sum(doc_features, dim=0)
            doc_features = torch.cat((mean_features, max_features, sum_features), dim=1)
            doc_features = self.ll(doc_features)
        elif self.fusion_style == 6:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 1)
            doc_features = self.transEncoder(doc_features)
        elif self.fusion_style == 7:
            doc_features = torch.stack((pmi_doc_features, cos_doc_features, euc_doc_features, co_doc_features), 1)
            doc_features = self.memnet(doc_features)
        if self.adopt_dropout:
            doc_features = self.dropout(doc_features)
        logits = self.linear(doc_features)
        return logits
