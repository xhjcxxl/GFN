# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : Conv.py
# @explanation : Convolution Network


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Conv1d_no_relu(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d_no_relu, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs, bias=False)
            for fs in filter_sizes
        ])
        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        return [conv(x) for conv in self.convs]


class MemoryLayer(nn.Module):
    def __init__(self, in_channels: int = 8, out_channels: int = 2, kernel_size: int = 3):
        super(MemoryLayer, self).__init__()
        self.mems = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in kernel_size * [1]])
        self.fuse = nn.Linear(kernel_size, 1)
        self.outpro = nn.Linear(in_channels, in_channels)

    def forward(self, query):
        assignments = [mem(query) for mem in self.mems]
        assignments = torch.stack([ass for ass in assignments], dim=3)
        fused_assignments = self.fuse(assignments)
        fused_assignments = fused_assignments.squeeze(dim=-1)
        # here we need to be attention the softmax dimmension
        fused_assignments = torch.softmax(fused_assignments, dim=-2)
        fused_logits = torch.matmul(fused_assignments.permute(0, 2, 1), query)
        output_features = self.outpro(fused_logits)
        return output_features


class MemoryFusion(nn.Module):
    def __init__(self, in_channel1: int = 8, in_channel2: int = 8, output_channel1: int = 2, output_channel2: int = 1,
                 kernel_size1: int = 2, kernel_size2: int = 2):
        super(MemoryFusion, self).__init__()
        self.layer1 = MemoryLayer(in_channel1, output_channel1, kernel_size1)
        self.layer2 = MemoryLayer(in_channel2, output_channel2, kernel_size2)
        self.layernorm = nn.LayerNorm(in_channel1)

    def forward(self, query):
        query1 = self.layer1(query)
        query2 = self.layer2(query1)
        out_features = self.layernorm(query2.squeeze())
        return out_features


if __name__ == '__main__':
    a = torch.randn(64, 8, 4)
    query = a.permute(0, 2, 1)
    MM = MemoryFusion()
    MM(query)
