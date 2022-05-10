# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : WarmUpLr.py
# @explanation : Warm Up Lr


import torch.nn as nn


class WarmUpGradually(nn.Module):
    def __init__(self, init_lr, training_step, warm_up_steps):
        super(WarmUpGradually, self).__init__()

        self.init_lr = init_lr
        self.traing_step = training_step
        self.warm_up_steps = warm_up_steps

    def forward(self, init_lr = None):
        learning_rate = 0.0
        if self.warm_up_steps and self.traing_step < self.warm_up_steps:
            warmup_percent_done = self.traing_step / self.warm_up_steps
            warmup_learning_rate = self.init_lr * warmup_percent_done  # gradual warmup_lr
            learning_rate = warmup_learning_rate
        else:
            learning_rate = learning_rate ** 1.0001
        return learning_rate
