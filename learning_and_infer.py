# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : learning_and_infer.py
# @explanation : inference model


import torch
from data_helper import DataHelper
import numpy as np
import nni
import random
import argparse
import datetime
import os
import torch.nn.functional as F
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from sklearn.metrics import f1_score

from models import Infer_model
from models.Conv import Conv1d_no_relu
from models.Conv import MemoryFusion

NUM_ITER_EVAL = 150
EARLY_STOP_EPOCH = 10
# this file is modified to adapt edges accompying with the model_ada.py (this file is modified to obtain the right gradients)
from config import DefaultConfig
import time
start = time.time()


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
    """get using time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def train(args, in_channels, pmi_model, cos_model, euc_model, co_model, config, sgd_lr, sgd_mom):
    data_helper = DataHelper(config.dataset, mode='train')
    for model in [pmi_model, cos_model, euc_model, co_model]:
        for _, paras in model.named_parameters():
            paras.requires_grad = False
    # ensemble model
    if config.ensemble_model == 1:
        infer_model = Infer_model.Linear(in_channels * config.label_num[config.dataset],
                                         config.label_num[config.dataset])
    elif config.ensemble_model == 2:
        out_channels = config.out_channels
        convs = Conv1d_no_relu(in_channels, out_channels, [1])
        print(convs)
    elif config.ensemble_model == 3:
        infer_model = MemoryFusion(config.label_num[config.dataset], config.label_num[config.dataset],
                                   output_channel1=args.output_channel, kernel_size1=args.kernel_size,
                                   kernel_size2=args.kernel_size)
        for _, paras in infer_model.named_parameters():
            paras.requires_grad = True
    loss_func = torch.nn.CrossEntropyLoss()

    if config.ensemble_model == 1:
        if config.adopt_SGD == 2:
            optim = torch.optim.AdamW(infer_model.parameters(), lr=sgd_lr)
            print('adopt the adamw')
        elif config.adopt_SGD == 1:
            optim = torch.optim.SGD(infer_model.parameters(), lr=sgd_lr, momentum=sgd_mom)
            print('adopt the sgd optimer')
        else:
            print('the optimer is error')
        infer_model.cuda()
    elif config.ensemble_model == 2:
        if config.adopt_SGD == 2:
            optim = torch.optim.AdamW(convs.parameters(), lr=sgd_lr)
            print('adopt the adamw')
        elif config.adopt_SGD == 1:
            optim = torch.optim.SGD(convs.parameters(), lr=sgd_lr, momentum=sgd_mom)
            print('adopt the sgd optimer')
        else:
            print('the optimer is error')
        convs.cuda()
    elif config.ensemble_model == 3:
        if config.adopt_SGD == 2:
            optim = torch.optim.AdamW(infer_model.parameters(), lr=sgd_lr)
            print('adopt the adamw')
        elif config.adopt_SGD == 1:
            optim = torch.optim.SGD(infer_model.parameters(), lr=sgd_lr, momentum=sgd_mom)
            print('adopt the sgd optimer')
        else:
            print('the optimer is error')
        infer_model.cuda()
    iter = 0
    start_time = time.time()
    best_acc = 0.0
    last_best_iter = 0
    total_loss = 0.0
    total_correct = 0
    total = 0
    pmi_model.eval()
    cos_model.eval()
    euc_model.eval()
    co_model.eval()
    if config.ensemble_model == 1:
        infer_model.train()
    elif config.ensemble_model == 2:
        convs.train()
    for content, label, epoch in data_helper.batch_iter(config.batch_size, num_epoch=200):
        iter += 1
        with torch.no_grad():
            pmi_logits = pmi_model(content)
            cos_logits = cos_model(content)
            euc_logits = euc_model(content)
            co_logits = cos_model(content)

            pmi_logits = pmi_logits.detach()
            cos_logits = cos_logits.detach()
            euc_logits = euc_logits.detach()
            co_logits = co_logits.detach()
            if config.ensemble_model == 1:
                input_logits = torch.cat((pmi_logits, cos_logits, euc_logits, co_logits), 1)
            elif config.ensemble_model == 2:
                if in_channels == 4:
                    input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)
                elif in_channels == 2:
                    input_logits = torch.stack([co_logits, euc_logits], dim=1)
                elif in_channels == 3:
                    input_logits = torch.stack([cos_logits, euc_logits, co_logits], dim=1)
            elif config.ensemble_model == 3:
                input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)
        if config.ensemble_model == 2:  # aggregation
            logits = convs(input_logits)
            logits = logits[0]
            logits = torch.transpose(logits, 1, 2)
            if config.pooler == 1:
                logits = F.max_pool1d(logits, logits.shape[2]).squeeze()
            elif config.pooler == 2:
                logits = F.avg_pool1d(logits, logits.shape[2]).squeeze()
            elif config.pooler == 3:
                print('sum pooler')
        elif config.ensemble_model == 1:
            logits = infer_model(input_logits)
        elif config.ensemble_model == 3:
            logits = infer_model(input_logits)
        loss = loss_func(logits, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        pred = torch.argmax(logits, dim=1)
        correct = torch.sum(pred == label)
        total_correct += correct
        total += len(label)
        total_loss += loss.detach().item()
        if iter % 10 == 0:
            with torch.no_grad():
                val_acc = dev(in_channels, pmi_model, cos_model, euc_model, cos_model,
                              convs if config.ensemble_model == 2 else infer_model, config)
            if val_acc > best_acc:
                best_acc = val_acc
                last_best_iter = iter
                improved = '*'

                msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Acc: {6:>7.2%}' \
                      + 'Val Acc: {2:>7.2%}, Time: {3}{4}' \

                print(msg.format(epoch, iter, val_acc, get_time_dif(start_time), improved, total_loss / NUM_ITER_EVAL,
                                 float(total_correct) / float(total)))

                total_loss = 0.0
                total_correct = 0
                total = 0

                torch.save(convs if config.ensemble_model == 2 else infer_model,
                           config_pmi.stored_models + config_pmi.dataset + '_infer.pkl')
        elif epoch >= config.max_epoch or iter - last_best_iter >= 50:
            print(f'The {epoch}th epoch and {iter}th iteration Converged!')
            return convs if config.ensemble_model == 2 else infer_model


def dev(in_channels, pmi_model, cos_model, euc_model, co_model, infer_model, config):
    batch_size = config.batch_size
    dataset = config.dataset
    data_helper = DataHelper(dataset, mode='dev')

    total_pred = 0
    correct = 0
    iter = 0
    for content, label, _ in data_helper.batch_iter(batch_size, num_epoch=1):
        iter += 1
        pmi_model.eval()
        cos_model.eval()
        euc_model.eval()
        co_model.eval()
        infer_model.eval()

        pmi_logits = pmi_model(content)
        cos_logits = cos_model(content)
        euc_logits = euc_model(content)
        co_logits = co_model(content)
        if config.ensemble_model == 1:
            input_logits = torch.cat((pmi_logits, cos_logits, euc_logits, co_logits), 1)
        elif config.ensemble_model == 2:
            if in_channels == 4:
                input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)
            elif in_channels == 2:
                input_logits = torch.stack([co_logits, euc_logits], dim=1)
            elif in_channels == 3:
                input_logits = torch.stack([cos_logits, euc_logits, co_logits], dim=1)
        elif config.ensemble_model == 3:
            input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)

        logits = infer_model(input_logits)
        if config.ensemble_model == 2:
            logits = logits[0]
            logits = torch.transpose(logits, 1, 2)
            if config.pooler == 1:
                logits = F.max_pool1d(logits, logits.shape[2]).squeeze()
            elif config.pooler == 2:
                logits = F.avg_pool1d(logits, logits.shape[2]).squeeze()
        pred = torch.argmax(logits, dim=1)
        correct_pred = torch.sum(pred == label)
        correct += correct_pred
        total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    return torch.div(correct, total_pred)


def test(metric_type, in_channels, pmi_model, cos_model, euc_model, co_model, dataset, config_pmi):
    infer_model = torch.load(config_pmi.stored_models + config_pmi.dataset + '_infer.pkl')
    data_helper = DataHelper(dataset, mode='test')
    total_pred = 0
    correct = 0
    iter = 0
    pred_list = []
    label_list = []
    all_content = []
    for content, label, _ in data_helper.batch_iter(config_pmi.batch_size, num_epoch=1):
        pmi_model.eval()
        cos_model.eval()
        euc_model.eval()
        co_model.eval()
        infer_model.eval()

        pmi_logits = pmi_model(content)
        cos_logits = cos_model(content)
        euc_logits = euc_model(content)
        co_logits = co_model(content)
        if config_pmi.ensemble_model == 1:
            input_logits = torch.cat((pmi_logits, cos_logits, euc_logits, co_logits), 1)
        elif config_pmi.ensemble_model == 2:
            if in_channels == 4:
                input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)
            elif in_channels == 2:
                input_logits = torch.stack([co_logits, euc_logits], dim=1)
            elif in_channels == 3:
                input_logits = torch.stack([cos_logits, euc_logits, co_logits], dim=1)
        elif config_pmi.ensemble_model == 3:
            input_logits = torch.stack([pmi_logits, cos_logits, euc_logits, co_logits], dim=1)

        logits = infer_model(input_logits)
        if config_pmi.ensemble_model == 2:
            logits = logits[0]
            logits = torch.transpose(logits, 1, 2)
            if config_pmi.pooler == 1:
                logits = F.max_pool1d(logits, logits.shape[2]).squeeze()
            elif config_pmi.pooler == 2:
                logits = F.avg_pool1d(logits, logits.shape[2]).squeeze()
        pred = torch.argmax(logits, dim=1)
        if metric_type in [1, 2, 3]:
            if iter == 0:
                pred_list = pred.cpu().numpy().tolist()
                label_list = label.cpu().numpy().tolist()

            else:
                pred_list.extend(pred.cpu().numpy().tolist())
                label_list.extend(label.cpu().numpy().tolist())
            all_content.extend(content.cpu().numpy().tolist())

        correct_pred = torch.sum(pred == label)
        correct += correct_pred
        total_pred += len(content)

        iter += 1
    if metric_type == 0:
        total_pred = float(total_pred)
        correct = correct.float()
        acc = torch.div(correct, total_pred).to('cpu').numpy()
        return [acc]
    elif metric_type == 1:
        f1_micro = f1_score(label_list, pred_list, average='micro')
        return [f1_micro]
    elif metric_type == 2:
        f1_macro = f1_score(label_list, pred_list, average='macro')
        return [f1_macro]
    elif metric_type == 3:
        f1_micro = f1_score(label_list, pred_list, average='micro')
        f1_macro = f1_score(label_list, pred_list, average='macro')
        return [[f1_micro], [f1_macro]]
    else:
        print('metric type error')
    df = pd.DataFrame([all_content, label_list, pred_list], columns=["content", "label", "pred"])
    df["new_content"] = df["content"].apply(" ".join(list(lambda x: data_helper.dtoword(x))))


def load_model(file_dir, file_pkl, is_cuda=True):
    if os.path.exists(file_dir + file_pkl):
        print('load model from file.')
        model = torch.load(file_dir + file_pkl)
    else:
        print('the pkl is not exist')
    if is_cuda:
        print('cuda')
        model.cuda()
    return model


def read_model(dir, dataset, top_k=1):
    files_dic = {}
    pmi_file_name = dataset + '_1'
    cos_file_name = dataset + '_2'
    euc_file_name = dataset + '_3'
    co_file_name = dataset + '_4'

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list = file.split('_')
            if len(file_list) == 3 and file_list[0] == dataset:
                file_name = file_list[0] + '_' + file_list[1]
                acc = file_list[2].split('.')[0] + '.' + file_list[2].split('.')[1]
                if file_name in files_dic:
                    files_dic[file_name].append(acc)
                else:
                    files_dic[file_name] = []
                    files_dic[file_name].append(acc)
    pmi_list = sorted(files_dic[pmi_file_name], reverse=1)
    cos_list = sorted(files_dic[cos_file_name], reverse=1)
    euc_list = sorted(files_dic[euc_file_name], reverse=1)
    co_list = sorted(files_dic[co_file_name], reverse=1)

    pmi_pkl_dir = pmi_file_name + '_' + pmi_list[top_k - 1] + '.pkl'
    cos_pkl_dir = cos_file_name + '_' + cos_list[top_k - 1] + '.pkl'
    euc_pkl_dir = euc_file_name + '_' + euc_list[top_k - 1] + '.pkl'
    co_pkl_dir = co_file_name + '_' + co_list[top_k - 1] + '.pkl'
    return pmi_pkl_dir, cos_pkl_dir, euc_pkl_dir, co_pkl_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', required=False, type=int, default=1, help='train or test mode')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--top_k', required=False, type=int, default=1, help='employ the model with top_k accuracy')
    parser.add_argument('--in_channels', required=False, type=int, default=2,
                        help='how many channels we want to ensemble')
    parser.add_argument('--out_channels', required=False, type=int, default=2, help='how many channels we adopt')
    parser.add_argument('--pooler', required=False, type=int, default=1, help='1:max_pool; 2:mean_pool')
    parser.add_argument('--adopt_SGD', required=False, type=int, default=2, help='1:sgd; 2:adamw')
    parser.add_argument('--ensemble_model', required=False, type=int, default=2, help='1:linear; 2:cnn;3:memory')
    parser.add_argument('--metric_type', required=False, type=int, default=0,
                        help='0:accuracy; 1:f1-micro;2:f1-macro;3.f1-micro & f1-macro')
    parser.add_argument('--adopt_nni', required=False, type=int, default=0, help='adopt nni or not')
    parser.add_argument('--output_channel', required=False, type=int, default=2,
                        help='the output channel of memory fusion methods')
    parser.add_argument('--kernel_size', required=False, type=int, default=2,
                        help='how many kernel adopted by the memory fusion methods')

    args = parser.parse_args()
    config_pmi = DefaultConfig()
    config_pmi.ensemble_model = args.ensemble_model
    config_pmi.adopt_SGD = args.adopt_SGD
    config_pmi.pooler = args.pooler
    config_pmi.out_channels = args.out_channels
    config_pmi.dataset = args.dataset

    config_cos = DefaultConfig()
    config_euc = DefaultConfig()
    config_co = DefaultConfig()

    SEED = config_pmi.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.top_k == 4:
        top_k_list = [1, 2, 3]
    elif args.top_k == 5:
        top_k_list = [1, 2, 3, 4, 5]
    elif args.top_k == 1:
        top_k_list = [1]
    elif args.top_k == 2:
        top_k_list = [2]
    elif args.top_k == 3:
        top_k_list = [3]
    else:
        print('the top_k value is error')

    if args.adopt_nni == 1:
        params = nni.get_next_parameter()
        args.top_k = params['top_k']
        args.ensemble_model = params['ensemble_model']
        args.output_channel = params['output_channel']
        args.kernel_size = params['kernel_size']
    for top_k in top_k_list:
        pmi_pkl_dir, cos_pkl_dir, euc_pkl_dir, co_pkl_dir = read_model(config_pmi.stored_models, config_pmi.dataset,
                                                                       top_k)
        print(pmi_pkl_dir, cos_pkl_dir, euc_pkl_dir, co_pkl_dir)
        print(f'The dataset is {config_pmi.dataset}')
        print(f'The output channels is {config_pmi.out_channels}')
        print(f'the pooler is {config_pmi.pooler}')
        print(f'the top_k is {top_k}')

        pmi_model = load_model(config_pmi.stored_models, file_pkl=pmi_pkl_dir, is_cuda=True)
        cos_model = load_model(config_cos.stored_models, file_pkl=cos_pkl_dir, is_cuda=True)
        euc_model = load_model(config_euc.stored_models, file_pkl=euc_pkl_dir, is_cuda=True)
        co_model = load_model(config_co.stored_models, file_pkl=co_pkl_dir, is_cuda=True)

        in_channels = args.in_channels
        if config_pmi.adopt_SGD == 1:
            if config_pmi.ensemble_model == 1:
                lr_list = [0.2]
                mom_list = [0.9]
            elif config_pmi.ensemble_model == 2:
                lr_list = [0.1]
                mom_list = [0.9]
            elif config_pmi.ensemble_model == 3:
                lr_list = [0.01]
                mom_list = [0.9]
        elif config_pmi.adopt_SGD == 2:
            if config_pmi.ensemble_model == 1:
                lr_list = [0.2]
                mom_list = [0.9]
            elif config_pmi.ensemble_model == 2:
                lr_list = [0.05]
                mom_list = [0.9]
            elif config_pmi.ensemble_model == 3:
                lr_list = [0.05]
                mom_list = [0.9]
        for sgd_lr in lr_list:
            for sgd_mom in mom_list:
                print(f'the momentum is {sgd_mom}')
                print(f'the learning rate is {sgd_lr}')
                if not args.train_mode:  # predict model
                    with torch.no_grad():
                        test_acc = test(args.metric_type, in_channels, pmi_model, cos_model, euc_model, co_model,
                                        config_pmi.dataset, config_pmi).numpy()
                    print(test_acc)
                else:  # train model
                    for i in range(3) if args.top_k in [1, 2, 3] else range(1):
                        convs = train(args, in_channels, pmi_model, cos_model, euc_model, co_model, config_pmi, sgd_lr,
                                      sgd_mom)
                        with torch.no_grad():
                            test_acc = test(args.metric_type, in_channels, pmi_model, cos_model, euc_model, co_model,
                                            config_pmi.dataset, config_pmi)
                        print(test_acc)
                        if config_pmi.ensemble_model == 2:
                            for paras in convs.parameters():
                                print(paras)
                        if i == 0:
                            test_acc_list = test_acc
                        else:
                            if len(test_acc) == 1:
                                test_acc_list = np.append(test_acc_list, test_acc)
                            else:
                                test_acc_list[0].extend(test_acc[0])
                                test_acc_list[1].extend(test_acc[1])
                        end = time.time()
                        print('the time is:')
                        print(str(end - start))

                    if args.top_k in [1, 2, 3]:
                        if args.metric_type in [0, 1, 2]:
                            print('the mean of the accuracy is:')
                            print(np.mean(test_acc_list))
                            if args.adopt_nni == 1:
                                nni.report_final_result(np.mean(test_acc_list))
                            print(test_acc_list)
                            print('the std of the test accuracy is')
                            print(np.std(test_acc_list, ddof=1))
                            print('the std is:')
                            print(np.sqrt(
                                ((test_acc_list - np.mean(test_acc_list)) ** 2).sum() / (len(test_acc_list) - 1)))
                            end = time.time()
                            print(str(end - start))
                        elif args.metric_type == 3:
                            f1_micro = test_acc_list[0]
                            f1_macro = test_acc_list[1]
                            print('the mean of the f1_micro is:')
                            print(np.mean(f1_micro))
                            print(f1_micro)
                            print('the std of the test accuracy is')
                            print(np.std(f1_micro, ddof=1))
                            print('the std is:')
                            print(np.sqrt(
                                ((f1_micro - np.mean(f1_micro)) ** 2).sum() / (len(f1_micro) - 1)))

                            print('the mean of the f1_ma    cro is:')
                            print(np.mean(f1_macro))
                            print(f1_macro)
                            print('the std of the test accuracy is')
                            print(np.std(f1_macro, ddof=1))
                            print('the std is:')
                            print(np.sqrt(
                                ((f1_macro - np.mean(f1_macro)) ** 2).sum() / (len(f1_macro) - 1)))
