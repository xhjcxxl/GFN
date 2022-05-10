# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : data_helper.py
# @explanation : data process


import os
import torch
import csv
import numpy as np
import spacy
import pickle
from nltk import Tree


class DataHelper(object):
    def __init__(self, dataset, mode='train', vocab=None):
        allowed_data = ['r8', '20ng', 'r52', 'mr', 'oh', 'sst5']
        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset

        if self.dataset in ['r8', 'r52', 'oh']:
            self.vocab_file = 'vocab-5.txt'
        elif self.dataset in ['sst5', 'mr']:
            self.vocab_file = 'vocab-1.txt'
        elif self.dataset == '20ng':
            self.vocab_file = 'vocab-10.txt'
        else:
            print('the vocab_file is error')

        self.mode = mode
        self.base = os.path.join('data', self.dataset)
        self.current_set = os.path.join(self.base, '%s-%s-stemmed.txt' % (self.dataset, self.mode))
        with open(os.path.join(self.base, 'label.txt')) as f:
            labels = f.read()
        self.labels_str = labels.split('\n')
        content, label = self.get_content()
        self.label = self.label_to_onehot(label)
        if vocab is None:
            self.vocab = []
            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))
        self.dtoword = {v: k for k, v in self.d.items()}
        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

    def label_to_onehot(self, label_str):
        return [self.labels_str.index(l) for l in label_str]

    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]
        if self.dataset == '20ng' or 'r52':
            cleaned = []
            for i, pair in enumerate(content):
                if len(pair) < 2:
                    pass
                else:
                    cleaned.append(pair)
        else:
            cleaned = content

        label, content = zip(*cleaned)
        return content, label

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']
        return result

    def id2word(self, id):
        try:
            result = self.dtoword[id]
        except KeyError:
            result = " "
        return result

    def get_vocab(self):
        with open(os.path.join(self.base, self.vocab_file)) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def build_vocab(self, content, min_count=10):
        vocab = []
        for c in content:
            words = c.split(' ')
            for word in words:
                if word not in vocab:
                    vocab.append(word)
        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1
        results = []
        for word in freq.keys():
            if freq[word] < min_count:
                continue
            else:
                results.append(word)

        results.insert(0, 'UNK')
        with open(os.path.join(self.base, 'vocab-5.txt'), 'w') as f:
            f.write('\n'.join(results))
        self.vocab = results

    def count_word_freq(self, content):
        freq = dict(zip(self.vocab, [0 for i in range(len(self.vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        with open(os.path.join(self.base, 'freq.csv'), 'w') as f:
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]
                yield content, torch.tensor(label).cuda(), i

    def build_syntactic_tree(self):
        nlp = spacy.load('en_core_web_sm')
        docs, _ = self.get_content()
        docs2graph = []
        if self.mode == 'train':
            fout = open(f'./distance_matrix/{self.dataset}_synctactic_tree.graph', 'wb')
        elif self.mode in ['dev', 'test']:
            fout = open(f'./distance_matrix/{self.dataset}_{self.mode}_synctactic_tree.graph', 'wb')
        for text in docs:
            document = nlp(text)
            seq_len = len(text.split())
            matrix = np.zeros((seq_len, seq_len)).astype('float32')

            for token in document:
                if token.i < seq_len:
                    matrix[token.i][token.i] = 1
                    for child in token.children:
                        if child.i < seq_len:
                            matrix[token.i][child.i] = 1
                            matrix[child.i][token.i] = 1
            docs2graph.append(matrix)
        pickle.dump(docs2graph, fout)
        fout.close()

    def test_dependency_tree(self,
                             str='The quick brown fox jumps over the lazy dog， the lazy dog scared. And then the lazy dog ran away'):
        en_nlp = spacy.load('en')
        doc = en_nlp(str)

        def to_nltk_tree(node):
            if node.n_lefts + node.n_rights > 0:
                return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
            else:
                return node.orth_
        [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


def del_duplicate_pkl(del_dir='./stored_models/', file_num=5):
    files_dic = {}
    for root, dirs, files in os.walk(del_dir):
        for file in files:
            file_list = file.split('_')
            if len(file_list) == 3 and file_list[0] in ['r8', 'r52', 'oh', 'mr', 'sst5', '20ng']:
                file_name = file_list[0] + '_' + file_list[1]
                acc = file_list[2].split('.')[0] + '.' + file_list[2].split('.')[1]
                if file_name in files_dic:
                    files_dic[file_name].append(acc)
                else:
                    files_dic[file_name] = []
                    files_dic[file_name].append(acc)
        for file_name in files_dic:
            acc_list = files_dic[file_name]
            acc_list = sorted(acc_list, reverse=1)
            for acc in acc_list[file_num:]:
                del_file = del_dir + file_name + '_' + acc + '.pkl'
                os.system(f'rm {del_file}')


if __name__ == '__main__':
    data_helper = DataHelper(dataset='r8', mode='test')
    docs2graph = data_helper.build_syntactic_tree()
    data_helper.test_dependency_tree(
        'crphilli hound dazixca ingr com ron phillips subject next mormon jew nntp posting host hound reply crphilli hound dazixca ingr com organization intergraph electronics mountain view ca distribution usa line article c n dyj world std com rjk world std com robert j kolker writes thank remembering matzada matzada insane act sanctification g name extreme denial tyranny possible day officer tzahal isreal defense force take oath fortress lo tepol shaynit matzadah matzadah fall anymore recent archaeological inspection site present pretty compelling evidence mass suicide masada never occured evidence compelling tha tzahal long hold secret ceremony fortress ron phillips crphilli hound dazixca ingr com senior customer engineer intergraph electronics east evelyn avenue voice mountain view ca fax')
    del_duplicate_pkl()
