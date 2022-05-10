# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : lnfer_model.py
# @explanation : linear model


import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.dropout = nn.Dropout(p=0.5)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
