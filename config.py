# -*- coding: utf-8 -*-
# @Time    : 2020-05-16 16:38
# @Author  : dai yong
# @File    : config.py


class DefaultConfig(object):
    mask_ngram = 0
    dir_base = ''
    stored_models = dir_base + 'stored_models/'
    ngram = 3
    name = 'temp_model'
    dropout = 0.5
    dataset = 'r8'
    edges = 0
    nrep = 3  # how many times the code run
    trainable_edges = 1
    set_edge_1 = 1
    global_edge = 1  # decide whether the graph is global or local
    global_add_local = 1  # decide whether the global graph need to add the ngram information
    adapte_edge = 1  # decide whether the graph weights need to be modified and add the regularizer loss
    add_regu_loss = 0
    early_stop_epoch = 10
    weight_decay = 1e-4  # the weight decay of the optimizer of Adam
    max_epoch = 100
    batch_size = 64
    adopt_SGD = 0
    SGD_lr = 1e-4
    SGD_momentum = 0
    adopt_adamW = 1  # decide whether to adopt the AdamW optimizer
    two_layer = 0  # decide whether to adopt two layers of graph convolution
    readout = 0  # decide what readout layer to be selected
    reduce = 0  # decide what reduce function to be selected
    # decide whether the code is under the baseline mode
    baseline = 0
    lstm_encoder = 0
    adopt_gat = 0  # decide whether to adopt the gat in the case that set_egde_1 is 1
    label_num = {'r8': 8, 'r52': 52, 'oh': 23, 'mr': 2, 'sst5': 5, '20ng': 20}
    rand = 7
    alpha = 0.1
    beta = 0.1
    gamma = 0.005
    adopt_nni = 0
    del_model = 0
    out_channels = 2  # out_channel is used to decide how many kernel to be used in the file 'learning_and_infer' by using cnn model
    pooler = 1  # pooler is used in file 'learning_and_infer' by using cnn model
    ensemble_model = 2  # ensemble_model is used to decide whether the linear model or cnn model is adopted (1:linear;2:cnn)
    adopt_scheduler = 0
    warmup_steps = 200
