# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : nni_train_ada_regu.py
# @explanation : build gfn and training


import torch
from data_helper import del_duplicate_pkl
from model_ada import Model
from data_helper import DataHelper
from models.WarmUpLr import WarmUpGradually
import matplotlib.pyplot as plt
import time

start = time.time()
import numpy as np
import tqdm
import random
import argparse
import time, datetime
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from pmi import EdgecalHelper
from pmi import Mask_ngram
from Regularizer import regu_loss

NUM_ITER_EVAL = 150
import nni

EARLY_STOP_EPOCH = 10
from config import DefaultConfig  # add config


def edges_mapping(vocab_len, content, ngram):
    count = 1
    mapping = np.zeros(shape=(vocab_len, vocab_len), dtype=np.int32)
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i - ngram), min(len(doc), i + ngram + 1)):
                dst = doc[dst_id]

                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                    count += 1

    for word in range(vocab_len):
        mapping[word, word] = count
        count += 1

    return count, mapping


def get_time_dif(start_time):
    """get used time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset):
    """
    validation dataset for trained model
    :param model: trained model
    :param dataset: validation dataset
    """
    data_helper = DataHelper(dataset, mode='dev')
    total_pred = 0
    correct = 0
    iter = 0
    for content, label, _ in data_helper.batch_iter(batch_size=64, num_epoch=1):
        iter += 1
        model.eval()

        logits = model(content)
        pred = torch.argmax(logits, dim=1)

        correct_pred = torch.sum(pred == label)

        correct += correct_pred
        total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    return torch.div(correct, total_pred)


def test(model_name, dataset, pkl_dir):
    """
    test dataset for trained model
    :param dataset: test dataset
    :param pkl_dir: saved model
    """
    model = torch.load(pkl_dir)
    data_helper = DataHelper(dataset, mode='test')
    total_pred = 0
    correct = 0
    iter = 0
    for content, label, _ in data_helper.batch_iter(batch_size=64, num_epoch=1):
        iter += 1
        model.eval()

        logits = model(content)
        pred = torch.argmax(logits, dim=1)
        correct_pred = torch.sum(pred == label)
        correct += correct_pred
        total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    return torch.div(correct, total_pred).to('cpu')


def train(config, pkl_dir, bar, is_cuda=True):
    """
    trian gfn model
    :param config: model config parameters
    :param pkl_dir: dataset data
    :param bar: to show bar information
    :param is_cuda: use GPU
    """
    edge_helper = EdgecalHelper(config.dir_base)
    hidden_size_node = 300
    data_helper = DataHelper(config.dataset, mode='train')
    print('the vocab_file is:', data_helper.vocab_file)

    node_embeddings, node_hidden = edge_helper.init_node_embeddings(config.dataset, hidden_size_node,
                                                                    vocab=data_helper.vocab)
    edges_type = {0: 'ngram', 1: 'pmi', 2: 'cosine', 3: 'euclidean', 4: 'co-occurrence', 5: 'ngram', 6: 'syntactic'}
    threshold = {0: 20, 1: 20, 2: 0.5, 3: 1.1, 4: 1.1, 5: 20, 6: 20}
    if not os.path.exists(config.stored_models + 'model_description'):
        print('create the model desription file')
    if not os.path.exists(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_edges_weights_{threshold[config.edges]}_{config.ngram}.npy'):
        if config.edges == 0:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_PMI(config.ngram, config.dataset,
                                                                                      config.global_add_local)
        if config.edges == 1:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_PMI_ngram(config.ngram,
                                                                                            config.dataset,
                                                                                            config.global_add_local)
        elif config.edges == 2:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_cosine_similarity(
                config.global_add_local, config.dataset, data_helper.vocab, node_hidden, threshold[config.edges],
                config.ngram)
        elif config.edges == 3:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_euclidean_distance(
                config.global_add_local, config.dataset, data_helper.vocab, node_hidden, threshold[config.edges],
                config.ngram)
        elif config.edges == 4:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_co_occurrence(config.ngram,
                                                                                                config.dataset,
                                                                                                config.global_add_local)
        elif config.edges == 5:
            print('ngram')
        elif config.edges == 6:
            edges_weights, edges_mappings, count, global_matrix = edge_helper.cal_syntactic(config.ngram, data_helper)
        else:
            print('edge error')
        np.save(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_edges_weights_{threshold[config.edges]}_{config.ngram}.npy',
            edges_weights.numpy())
        np.save(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_edges_mappings_{threshold[config.edges]}_{config.ngram}.npy',
            edges_mappings)
        np.save(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_count_{threshold[config.edges]}_{config.ngram}.npy',
            np.array(count))
        np.save(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_global_matrix_{threshold[config.edges]}_{config.ngram}.npy',
            global_matrix)
    else:
        edges_weights = torch.from_numpy(
            np.load(
                config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_edges_weights_{threshold[config.edges]}_{config.ngram}.npy'))
        edges_mappings = np.load(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_edges_mappings_{threshold[config.edges]}_{config.ngram}.npy')
        count = int(np.load(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_count_{threshold[config.edges]}_{config.ngram}.npy'))
        global_matrix = np.load(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_global_matrix_{threshold[config.edges]}_{config.ngram}.npy')

    if config.mask_ngram and not os.path.exists(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_mask_ngram_{threshold[config.edges]}.npy'):
        Mask = Mask_ngram(data_helper, config.ngram)
        edges_mappings = edges_mappings * Mask
        np.save(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_mask_ngram_{threshold[config.edges]}.npy',
            edges_mappings)
    elif config.mask_ngram:
        edges_mappings = np.load(
            config.dir_base + f'distance_matrix/{config.dataset}_{edges_type[config.edges]}_mask_ngram_{threshold[config.edges]}.npy')
    else:
        print('no mask ngram')

    if config.set_edge_1 == 1:
        edges_weights.fill_(1)

    ori_edge_embedding = edges_weights.cuda()
    seq_edge_w = edge_helper.init_edge_embeddings(config.edges, edges_weights, config.trainable_edges)

    model = Model(config, global_matrix, ori_edge_embedding, seq_edge_w, node_hidden, hidden_size_node,
                  class_num=len(data_helper.labels_str),
                  vocab=data_helper.vocab, edges_matrix=edges_mappings, edges_num=count, pmi=edges_weights,
                  cuda=is_cuda)
    print(model)
    if is_cuda:
        print('cuda')
        model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    if not config.adopt_scheduler:
        if config.adopt_adamW:
            optim = torch.optim.AdamW(model.parameters(), lr=0.001)
            print('adopted adamW')
        else:
            optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)
            print('adopted adam')
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_acc = 0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total = 0
    lr_li = []
    if config.adopt_scheduler:
        scheduler = WarmUpGradually(init_lr=0.01, training_step=iter + 1,
                                    warm_up_steps=int(len(data_helper.content) / config.batch_size * 0.5))

    for content, label, epoch in data_helper.batch_iter(config.batch_size, num_epoch=200):
        if config.adopt_scheduler:
            if epoch < 2:
                optim.param_groups[0]['lr'] = scheduler()
                lr_li.append(scheduler())
            else:
                optim.param_groups[0]['lr'] = 0.001
        if config.adopt_SGD and epoch == 5:
            optim = torch.optim.SGD(model.parameters(), lr=config.SGD_lr, momentum=config.SGD_momentum)
        improved = ''
        model.train()  # training
        logits = model(content)
        loss = loss_func(logits, label)
        if config.add_regu_loss:
            re_loss_class = regu_loss(config.edges, edges_weights, edges_mappings, config.alpha, config.beta,
                                      config.gamma)
            r_loss = re_loss_class.cal_regu_loss(config.add_regu_loss, model.sub_embedding_dist,
                                                 model.sub_edge_embedding, model.sub_adapted_egdes)
            comb_loss = loss + r_loss

        pred = torch.argmax(logits, dim=1)
        correct = torch.sum(pred == label)
        total_correct += correct
        total += len(label)
        if config.add_regu_loss:
            total_loss += comb_loss.item()
        else:
            total_loss += loss.item()
        optim.zero_grad()

        if config.add_regu_loss:
            comb_loss.backward()
        else:
            loss.backward()
        optim.step()

        iter += 1
        if bar:
            pbar.update()
        if epoch == 0 and iter == 1:
            torch.save(model, pkl_dir)
        elif epoch >= config.max_epoch or epoch - last_best_epoch >= config.early_stop_epoch:
            return config.name
        elif iter % NUM_ITER_EVAL == 0:
            if bar:
                pbar.close()

            with torch.no_grad():
                val_acc = dev(model, dataset=config.dataset)
            if val_acc > best_acc:
                best_acc = val_acc
                last_best_epoch = epoch
                improved = '*'

                torch.save(model, pkl_dir)
            msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Acc: {6:>7.2%}' \
                  + 'Val Acc: {2:>7.2%}, Time: {3}{4}' \

            print(msg.format(epoch, iter, val_acc, get_time_dif(start_time), improved, total_loss / NUM_ITER_EVAL,
                             float(total_correct) / float(total)))

            total_loss = 0.0
            total_correct = 0
            total = 0
            if bar:
                pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    if config.adopt_scheduler:
        plt.plot(lr_li)
        plt.show()
    return config.name


def word_eval(pkl_dir):
    print('load model from file.')
    data_helper = DataHelper('r8')
    edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, 1)
    model = torch.load(pkl_dir)

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()
    core_word = 'billion'
    core_index = data_helper.vocab.index(core_word)

    results = {}
    for i in range(len(data_helper.vocab)):
        word = data_helper.vocab[i]
        n_word = edges_matrix[i, core_index]
        if n_word != 0:
            results[word] = edges_weights[n_word][0]

    sort_results = sorted(results.items(), key=lambda d: d[1])
    print(sort_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_base', required=False, type=str, default='/hdd/daiyong/gfn/',
                        help='the path of temporary files')
    parser.add_argument('--ngram', required=False, type=int, default=3, help='ngram number')
    parser.add_argument('--name', required=False, type=str, default='temp_model', help='project name')
    parser.add_argument('--bar', required=False, type=int, default=0, help='show bar')
    parser.add_argument('--dropout', required=False, type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--rand', required=False, type=int, default=7, help='rand_seed')  # random seed
    parser.add_argument('--nrep', required=False, type=int, default=1, help='run n times')

    parser.add_argument('--edges', required=False, type=int, default=0, help='trainable edges')
    parser.add_argument('--trainable_edges', required=False, type=int, default=0, help='the edge is trainable or not')
    parser.add_argument('--set_edge_1', required=False, type=int, default=1, help='set all edge weight to 1')
    parser.add_argument('--global_edge', required=False, type=int, default=1,
                        help='0:just neighbors; 1: neighbors and globals')
    parser.add_argument('--global_add_local', required=False, type=int, default=1,
                        help='add edge according to the global information')
    parser.add_argument('--adapte_edge', required=False, type=int, default=1,
                        help='adapt the edge automatically')

    parser.add_argument('--add_regu_loss', required=False, type=int, default=0,
                        help='add the regularizer loss, 0 for no, 1 for regu_1, 2 for regu_2')
    parser.add_argument('--early_stop_epoch', required=False, type=int, default=10, help='early stop epoch')
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-4, help='weght decay')
    parser.add_argument('--max_epoch', required=False, type=int, default=100, help='batch_size')
    parser.add_argument('--batch_size', required=False, type=int, default=32, help='batch_size')
    parser.add_argument('--adopt_SGD', required=False, type=int, default=1, help='after some epochs adopt SGD')
    parser.add_argument('--SGD_lr', required=False, type=float, default=0.001, help='SGD learning rate')
    parser.add_argument('--SGD_momentum', required=False, type=float, default=0.0, help='SGD_momentum')
    parser.add_argument('--mask_ngram', required=False, type=int, default=0,
                        help='mask out the items not belong to ngram')
    parser.add_argument('--alpha', required=False, type=float, default=0.1, help='control the part of regularizer loss')
    parser.add_argument('--beta', required=False, type=float, default=0.1, help='control the part of regularizer loss')
    parser.add_argument('--gamma', required=False, type=float, default=0.005,
                        help='control the part of regularizer loss')
    parser.add_argument('--adopt_nni', required=False, type=int, default=0, help='adopt nni or not')
    parser.add_argument('--adopt_adamW', required=False, type=int, default=1, help='adopt adamW')
    parser.add_argument('--adopt_nni_paras', required=False, type=int, default=1, help='adopt adamW')
    parser.add_argument('--two_layer', required=False, type=int, default=0, help='adopt two layer or not')
    parser.add_argument('--readout', required=False, type=int, default=0,
                        help='0:sum; 1:mean; 2:max; 3:rcru; 4:lstm; 5:cnn')
    parser.add_argument('--reduce', required=False, type=int, default=0, help='0:max; 1:sum; 2:mean;')
    parser.add_argument('--baseline', required=False, type=int, default=0, help='adopt the baseline or our model')
    parser.add_argument('--lstm_encoder', required=False, type=int, default=0,
                        help='adopt the lstm as the encoder for further processing')
    parser.add_argument('--adopt_gat', required=False, type=int, default=0,
                        help='adopt GAT as the message processing module')
    parser.add_argument('--del_model', required=False, type=int, default=1,
                        help='store the model for infer with the best validate accuracy')

    args = parser.parse_args()

    if args.adapte_edge:
        print('Adopted adapted edge')
    if args.add_regu_loss:
        print('Adopted regu loss')

    if args.adopt_nni and args.baseline:
        params = nni.get_next_parameter()  # add nni parameter
        args.max_epoch = params['max_epoch']  # set epoch
        args.ngram = params['ngram']
        args.weight_decay = params['weight_decay']
        args.dropout = params['dropout']
        args.batch_size = params['batch_size']
        args.name = str(np.random.random(1).item())

    elif args.adopt_nni and args.adopt_nni_paras:
        print('Adopted nni')
        params = nni.get_next_parameter()
        args.max_epoch = params['max_epoch']
        args.ngram = params['ngram']
        args.weight_decay = params['weight_decay']
        args.dropout = params['dropout']
        args.batch_size = params['batch_size']
        args.alpha = params['alpha']
        args.beta = params['beta']
        args.gamma = params['gamma']
        args.trainable_edges = params['trainable_edges']
        args.name = str(np.random.random(1).item())
    else:
        args.name = str(np.random.random(1).item())
        print('Not adopted nni')
    if args.gamma < 0:
        args.gamma == 0.01

    print('readout:', args.readout)
    print('ngram: %d' % args.ngram)
    print('dataset: %s' % args.dataset)
    print('edges: %s' % args.edges)

    SEED = args.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.bar == 1:
        bar = True
    else:
        bar = False

    # parameter read and setting
    if args.dataset == 'sst5':
        args.ngram = 5
    elif args.dataset == 'oh':
        args.ngram = 7
    elif args.dataset == '20ng':
        args.ngram = 7

    config = DefaultConfig()
    config.name = args.name
    config.global_add_local = args.global_add_local
    config.reduce = args.reduce
    config.readout = args.readout
    config.two_layer = args.two_layer
    config.adopt_adamW = args.adopt_adamW
    config.dir_base = args.dir_base
    config.max_epoch = args.max_epoch
    config.dir_base = args.dir_base
    config.alpha = args.alpha
    config.beta = args.beta
    config.gamma = args.gamma
    config.SGD_momentum = args.SGD_momentum
    config.SGD_lr = args.SGD_lr
    config.adopt_SGD = args.adopt_SGD
    config.batch_size = args.batch_size
    config.weight_decay = args.weight_decay
    config.early_stop_epoch = args.early_stop_epoch
    config.adapte_edge = args.adapte_edge
    config.add_regu_loss = args.add_regu_loss
    config.global_edge = args.global_edge
    config.trainable_edges = args.trainable_edges
    config.set_edge_1 = args.set_edge_1
    config.ngram = args.ngram
    config.dropout = args.dropout
    config.edges = args.edges
    config.dataset = args.dataset
    config.mask_ngram = args.mask_ngram
    config.lstm_encoder = args.lstm_encoder
    config.adopt_gat = args.adopt_gat
    config.del_model = args.del_model  # end of the parameter reading

    pkl_dir = 'pkl/' + args.name + '.pkl'  # add data pkl
    # use this dic to store models with 3 top accuracy
    if not os.path.exists(config.stored_models + 'top_3_acc.npy'):
        top_3_acc = {'r8_1': [0.1, 0.11, 0.12], 'r8_2': [0.1, 0.11, 0.12], 'r8_3': [0.1, 0.11, 0.12],
                     'r8_4': [0.1, 0.11, 0.12], \
                     'r52_1': [0.1, 0.11, 0.12], 'r52_2': [0.1, 0.11, 0.12], 'r52_3': [0.1, 0.11, 0.12],
                     'r52_4': [0.1, 0.11, 0.12], \
                     'oh_1': [0.1, 0.11, 0.12], 'oh_2': [0.1, 0.11, 0.12], 'oh_3': [0.1, 0.11, 0.12],
                     'oh_4': [0.1, 0.11, 0.12], \
                     'mr_1': [0.1, 0.11, 0.12], 'mr_2': [0.1, 0.11, 0.12], 'mr_3': [0.1, 0.11, 0.12],
                     'mr_4': [0.1, 0.11, 0.12], \
                     'sst5_1': [0.1, 0.11, 0.12], 'sst5_2': [0.1, 0.11, 0.12], 'sst5_3': [0.1, 0.11, 0.12],
                     'sst5_4': [0.1, 0.11, 0.12], \
                     '20ng_1': [0.1, 0.11, 0.12], '20ng_2': [0.1, 0.11, 0.12], '20ng_3': [0.1, 0.11, 0.12],
                     '20ng_4': [0.1, 0.11, 0.12]}
    else:
        top_3_acc = np.load(config.stored_models + 'top_3_acc.npy').item()

    for i in range(args.nrep):
        model = train(config, pkl_dir, bar, is_cuda=True)  # train and get model
        with torch.no_grad():
            test_acc_temp = test(model, args.dataset, pkl_dir).numpy()  # predict
        print(test_acc_temp)
        if args.adopt_nni:
            nni.report_final_result(test_acc_temp.item())
        logging.info('nni logs')
        logging.info('\n')
        logging.info(f'ngram--{args.ngram}')
        logging.info(f'epoch--{args.max_epoch}')
        logging.info(f'weight_decay--{args.weight_decay}')
        logging.info(f'dropout--{args.dropout}')
        logging.info(f'batchsize--{args.batch_size}')
        logging.info(f'edges--{args.edges}')
        logging.info(f'add_regu_loss--{args.add_regu_loss}')
        logging.info(f'test_accuracy--{test_acc_temp.item()}')

        logging.info(f'alpha--{args.alpha}')
        logging.info(f'beta--{args.beta}')
        logging.info(f'gamma--{args.gamma}')
        logging.info(f'trainable_edges--{args.trainable_edges}')

        if i == 0:
            test_acc = test_acc_temp
        else:
            test_acc = np.append(test_acc, test_acc_temp)
        print(test_acc)

        if i == (args.nrep - 1):
            test_acc_total = np.append(test_acc.mean(), test_acc.var())
            logging.info(f'the average accuracy is -- {test_acc.mean()}')
            logging.info(f'the variance is -- {test_acc.var()}')
            print(f'the standard deviation is -- {np.std(test_acc, ddof=1)}')
            print(f'the average accuracy is: {test_acc.mean()}')
            print(f'the variance is: {test_acc.var()}')
            print('the training time is:')
            end = time.time()
            print(str(end - start))
            # the code below is to decide whether to delete the model according to the accuracy
            print(pkl_dir)
            if config.edges in [1, 2, 3, 4]:
                store_dir = config.stored_models + config.dataset + '_' + str(config.edges) + '_' + str(
                    test_acc_temp) + '.pkl'
                os.system(f'mv {pkl_dir} {store_dir}')
                del_duplicate_pkl()
                print('Have delete the pkl out of the top 5 range')

            else:
                print('the edge is 0 or 6, just to test the accuracy')
