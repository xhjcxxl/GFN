# -*- coding: utf-8 -*-
# @Time        : 2020-05-22 11:38
# @Author      : dai yong
# @File        : model_ada.py
# @explanation : model file with adapte


import dgl
import torch
import torch.nn.functional as F
import numpy as np
import word2vec
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from models.Conv import Conv1d
from dgl.nn.pytorch import GATConv
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder


def gcn_msg(edge):
    return {'m': edge.src['h'], 'w': edge.data['w']}


def gcn_reduce(node):
    w = node.mailbox['w']
    new_hidden = torch.mul(w, node.mailbox['m'])
    new_hidden, _ = torch.max(new_hidden, 1)
    return {'h': new_hidden}


class Model(torch.nn.Module):
    def __init__(self,
                 config,
                 global_matrix,
                 ori_edge_embedding,
                 seq_edge_w,
                 node_hidden,
                 hidden_size_node,
                 class_num,
                 vocab,
                 edges_num,
                 edges_matrix,
                 max_length=350,
                 pmi=None,
                 cuda=True
                 ):
        super(Model, self).__init__()
        self.adopt_gat = config.adopt_gat
        self.global_add_local = config.global_add_local
        self.global_matrix = global_matrix
        self.dataset = config.dataset
        self.reduce = config.reduce
        self.readout = config.readout
        self.two_layer = config.two_layer
        self.is_cuda = cuda
        self.vocab = vocab
        self.add_regu_loss = config.add_regu_loss
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        self.edges_num = edges_num
        print(f'edge number is {edges_num}')
        self.node_hidden = node_hidden
        self.node_hidden.weight.requires_grad = True
        self.ori_edge_ebeding = ori_edge_embedding
        self.seq_edge_w = seq_edge_w
        self.edges = config.edges
        self.global_edge = config.global_edge
        self.adapte_edge = config.adapte_edge

        if self.adapte_edge:
            self.edge_modifiers = nn.Parameter(torch.rand(edges_num, 1), requires_grad=True)
            self.relu = nn.ReLU()
            # use adated_edges and original_embeddings to calculate the regu loss
            self.sub_adapted_egdes = None
            self.sub_edge_embedding = None
            self.sub_node_embedding = None
            self.sub_embedding_dist = None

        self.hidden_size_node = hidden_size_node
        self.len_vocab = len(vocab)
        self.ngram = config.ngram
        self.d = dict(zip(self.vocab, range(len(self.vocab))))
        self.max_length = max_length
        self.edges_matrix = edges_matrix
        self.dropout = torch.nn.Dropout(p=config.dropout)  # config.dropout
        self.activation = torch.nn.ReLU()
        self.lstm_encoder = config.lstm_encoder

        if self.adopt_gat:
            self.num_layers = 1
            self.gat_layers = nn.ModuleList()
            self.heads = [6, 1]
            self.feat_drop = 0.6
            self.attn_drop = 0.6
            self.negative_slope = 0.2
            self.residual = False
            self.activation = nn.LeakyReLU(self.negative_slope)
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                self.hidden_size_node, int(self.hidden_size_node / self.heads[0]), self.heads[0],
                self.feat_drop, self.attn_drop, self.negative_slope, False, self.activation))  # 图注意力网络
            # hidden layers
            for l in range(1, self.num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    self.hidden_size_node * self.heads[l - 1], int(self.hidden_size_node / self.heads[0]),
                    self.heads[l],
                    self.feat_drop, self.attn_drop, self.negative_slope, self.residual, self.activation))

        if self.lstm_encoder:
            self.encoder_linear = torch.nn.Linear(self.hidden_size_node, self.hidden_size_node, bias=True)
            self.lstm = nn.LSTM(self.hidden_size_node, int(self.hidden_size_node / 2), 1, bidirectional=True, dropout=0)

        if self.two_layer:
            self.L1_linear = torch.nn.Linear(self.hidden_size_node, self.hidden_size_node)

        if self.readout == 3:
            self.Linear = torch.nn.Linear(self.hidden_size_node * 2, class_num, bias=True)
            self.lstm = nn.LSTM(self.hidden_size_node, self.hidden_size_node, 2, bidirectional=True, dropout=0.5)
            self.W2 = nn.Linear(2 * self.hidden_size_node + self.hidden_size_node, self.hidden_size_node * 2)

            self.rnn_dropout = nn.Dropout(0.5)
        elif self.readout == 4:
            self.Linear = torch.nn.Linear(self.hidden_size_node, class_num, bias=True)
            self.lstm = nn.LSTM(self.hidden_size_node, self.hidden_size_node, 2, bidirectional=True, dropout=0.5)
            self.W2 = nn.Linear(2 * self.hidden_size_node + self.hidden_size_node, self.hidden_size_node)

            self.rnn_dropout = nn.Dropout(0.5)
        elif self.readout == 5:
            cnn_hiden_size = 50
            self.Linear = torch.nn.Linear(cnn_hiden_size * 5, class_num, bias=True)
            self.convs = Conv1d(self.hidden_size_node, cnn_hiden_size, [1, 2, 3, 4, 5])
            self.batchnorm = nn.BatchNorm1d(cnn_hiden_size * 5)
            self.cnn_relu = nn.ReLU()
        elif self.readout == 6:
            cnn_hiden_size = 60
            self.convs = Conv1d(cnn_hiden_size * 10, cnn_hiden_size, [1, 2, 3, 4, 5])
            self.convs_embeded = Conv1d(cnn_hiden_size * 5, cnn_hiden_size, [1, 2, 3, 4, 5])
            self.Linear = torch.nn.Linear(self.hidden_size_node * 2, class_num, bias=True)
            self.lstm = nn.LSTM(self.hidden_size_node, self.hidden_size_node, 2, bidirectional=True, dropout=0.5)
            self.W2 = nn.Linear(2 * self.hidden_size_node + self.hidden_size_node, self.hidden_size_node * 2)
            self.rnn_dropout = nn.Dropout(0.5)
        elif self.readout == 7:
            self.Linear = torch.nn.Linear(self.hidden_size_node, class_num, bias=True)
            self.Linear_w = torch.nn.Linear(self.hidden_size_node, self.hidden_size_node, bias=True)
            self.Linear_f = torch.nn.Linear(self.hidden_size_node, self.hidden_size_node, bias=True)
        elif self.readout == 8:
            self.max_len = {'r8': 100, 'r52': 100, 'oh': 160, 'mr': 50, 'sst5': 50, '20ng': 300}
            self.LayerNorm = nn.LayerNorm(self.hidden_size_node)
            self.Dropout = nn.Dropout(p=0.3)
            self.position_embeddings = nn.Embedding(self.max_len[self.dataset], self.hidden_size_node)
            self.layer = TransformerEncoderLayer(d_model=self.hidden_size_node, nhead=3)
            self.layer.linear1 = nn.Linear(self.hidden_size_node, 512)
            self.layer.linear2 = nn.Linear(512, self.hidden_size_node)
            self.encoder = TransformerEncoder(self.layer, num_layers=2)
            self.Linear = nn.Linear(self.hidden_size_node, class_num, bias=True)

        else:
            self.Linear = torch.nn.Linear(self.hidden_size_node, class_num, bias=True)

    def trans(self, graph_list):
        batch_tensors = []
        len_list = []
        tensors_list = None
        for i, graph in enumerate(graph_list):
            weighted_message = graph.ndata['h']
            batch_tensors.append(weighted_message)
            tensor_len = len(weighted_message)
            len_list.append(tensor_len)
            if tensor_len < self.max_len[self.dataset]:
                weighted_message = torch.cat((weighted_message, torch.zeros(self.max_len[self.dataset] - tensor_len,
                                                                            self.hidden_size_node).cuda()))
            else:
                weighted_message = weighted_message[0:self.max_len[self.dataset], :]
            weighted_message = torch.unsqueeze(weighted_message, dim=0)
            if i == 0:
                tensors_list = weighted_message
            else:
                tensors_list = torch.cat((tensors_list, weighted_message), dim=0)
        device = tensors_list.device
        position_ids = torch.arange(self.max_len[self.dataset], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

        input_features = tensors_list
        input_features = self.LayerNorm(input_features)
        input_features = self.Dropout(input_features)
        encoded = self.encoder(input_features)
        h_g_mean = torch.mean(encoded, dim=1)
        h_g_max = torch.max(encoded, dim=1)[0]
        h_g = h_g_max + h_g_mean
        return h_g

    def att(self, graph_list):
        h_g_grouped = None
        for i, graph in enumerate(graph_list):
            h_v = graph.ndata['h']
            att = torch.sigmoid(self.Linear_w(h_v))
            emb = torch.tanh(self.Linear_f(h_v))
            h_v_ = att.mul(emb)
            h_g_mean = torch.mean(h_v_, dim=0)
            h_g_max = torch.max(h_v_, dim=0)[0]
            h_g = h_g_max + h_g_mean
            h_g = torch.unsqueeze(h_g, dim=0)
            if i == 0:
                h_g_grouped = h_g
            else:
                h_g_grouped = torch.cat((h_g_grouped, h_g), dim=0)
        return h_g_grouped

    def rnncnn_readout(self, graph_list):
        max_len = {'r8': 100, 'r52': 100, 'oh': 160, 'mr': 50, 'sst5': 50, '20ng': 300}
        batch_tensors = []
        len_list = []
        for graph in graph_list:
            weighted_message = graph.ndata['h']
            batch_tensors.append(weighted_message)
            len_list.append(len(weighted_message))
        local_max = np.max(len_list)
        weighted_message = rnn_utils.pad_sequence(batch_tensors, batch_first=True)
        embedded = self.rnn_dropout(weighted_message)
        paded_embedded = embedded.permute(0, 2, 1)
        if local_max < max_len[self.dataset]:
            paded_embedded = F.pad(paded_embedded, (0, max_len[self.dataset] - local_max))
        else:
            paded_embedded = paded_embedded[:, :, 0:max_len[self.dataset]]
        conved_embeded = self.convs_embeded(paded_embedded)
        pooled_embedded = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                           for conv in conved_embeded]
        cat_embedded = torch.cat(pooled_embedded, dim=1)

        weighted_message_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_list, batch_first=True,
                                                                          enforce_sorted=False)
        outputs_packed, _ = self.lstm(weighted_message_packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs_packed)
        outputs = outputs.permute(1, 2, 0)
        if local_max < max_len[self.dataset]:
            outputs = F.pad(outputs, (0, max_len[self.dataset] - local_max))
        else:
            outputs = outputs[0:max_len[self.dataset], :]

        conved = self.convs(outputs)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = torch.cat(pooled, dim=1)
        y3 = torch.cat((cat, cat_embedded), dim=1)
        return y3

    def cnn_readout(self, graph_list):
        max_len = {'r8': 100, 'r52': 100, 'oh': 160, 'mr': 50, 'sst5': 50, '20ng': 300}
        batch_tensors = []
        len_list = []
        tensors_list = None
        for i, graph in enumerate(graph_list):
            weighted_message = graph.ndata['h']
            batch_tensors.append(weighted_message)
            tensor_len = len(weighted_message)
            len_list.append(tensor_len)
            if tensor_len < max_len[self.dataset]:
                weighted_message = torch.cat(
                    (weighted_message, torch.zeros(max_len[self.dataset] - tensor_len, self.hidden_size_node).cuda()))
            else:
                weighted_message = weighted_message[0:max_len[self.dataset], :]
            weighted_message = torch.unsqueeze(weighted_message, dim=0)
            if i == 0:
                tensors_list = weighted_message
            else:
                tensors_list = torch.cat((tensors_list, weighted_message), dim=0)

        tensors_list = tensors_list.permute(0, 2, 1)
        conved = self.convs(tensors_list)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = torch.cat(pooled, dim=1)
        return cat

    def lstm_readout(self, graph_list):
        batch_tensors = []
        len_list = []
        for graph in graph_list:
            weighted_message = graph.ndata['h']
            batch_tensors.append(weighted_message)
            len_list.append(len(weighted_message))
        weighted_message = rnn_utils.pad_sequence(batch_tensors, batch_first=True)
        embedded = self.rnn_dropout(weighted_message)
        weighted_message_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_list, batch_first=True,
                                                                          enforce_sorted=False)
        outputs_packed, (h, c) = self.lstm(weighted_message_packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs_packed)
        y3 = torch.sum(h, 0)
        return y3

    def rcnn_readout(self, graph_list):
        batch_tensors = []
        len_list = []
        for graph in graph_list:
            weighted_message = graph.ndata['h']
            batch_tensors.append(weighted_message)
            len_list.append(len(weighted_message))

        weighted_message = rnn_utils.pad_sequence(batch_tensors, batch_first=True)
        embedded = self.rnn_dropout(weighted_message)
        weighted_message_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_list, batch_first=True,
                                                                          enforce_sorted=False)
        outputs_packed, _ = self.lstm(weighted_message_packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs_packed)
        x = torch.cat((outputs.permute(1, 0, 2), embedded), 2)
        y2 = torch.tanh(self.W2(x))
        y2 = y2.permute(0, 2, 1)
        y3 = torch.sum(y2, dim=2)
        return y3

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def load_word2vec(self, word2vec_file):
        model = word2vec.load(word2vec_file)
        embedding_matrix = []

        for word in self.vocab:
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                embedding_matrix.append(model['the'])
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix

    def add_all_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []
        local_vocab = list(set(doc_ids))

        for i, src_word_old in enumerate(local_vocab):
            src = old_to_new[src_word_old]
            for dst_word_old in local_vocab[i:]:
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def cal_sub_embedding_dist(self, edges_pair):
        if self.edges == 3:
            edges_pair_tensor = torch.tensor(edges_pair).cuda()
            edges_pair_embedding = self.node_hidden(edges_pair_tensor)
            edges_pair_temp = torch.sub(edges_pair_embedding[:, 0, :], edges_pair_embedding[:, 1, :]) + 1e-20
            edges_pair_temp = torch.pow(edges_pair_temp, 2)
            edges_pair_temp = torch.sum(edges_pair_temp, 1)
            edges_pair_weight = torch.sqrt(edges_pair_temp)
        elif self.edges == 2:
            edges_pair_tensor = torch.tensor(edges_pair).cuda()
            edges_pair_embedding = self.node_hidden(edges_pair_tensor)
            edges_pair_temp = torch.cosine_similarity(edges_pair_embedding[:, 0, :], edges_pair_embedding[:, 1, :])
            edges_pair_temp = (1 - edges_pair_temp) / 2
            tmp = torch.randn(edges_pair_temp.shape).cuda()
            tmp.fill_(1e-8)
            edges_pair_weight = torch.where(edges_pair_temp == 0, edges_pair_temp, tmp)
        else:
            print('This type of edges do not need the |x1-x2| regularizer')
        return edges_pair_weight

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):  # 添加文本的边
        edges = []
        edges_pair = []
        old_edge_id = []
        if self.global_edge:
            for index, src_word_old in enumerate(doc_ids):
                src = old_to_new[src_word_old]
                for j in range(len(doc_ids)):
                    dst_word_old = doc_ids[j]
                    dst = old_to_new[dst_word_old]
                    if self.global_matrix[src_word_old, dst_word_old] != 0:  # 全局边
                        edges.append([src, dst])
                        old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
                        edges_pair.append([src_word_old, dst_word_old])
                if self.global_add_local:
                    for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                        dst_word_old = doc_ids[i]
                        dst = old_to_new[dst_word_old]
                        if self.global_matrix[src_word_old, dst_word_old] == 0:  # 局部边
                            if self.edges_matrix[src_word_old, dst_word_old] != 0:
                                # - first connect the new sub_graph
                                edges.append([src, dst])
                                # - then get the hidden from parent_graph
                                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
                                edges_pair.append([src_word_old, dst_word_old])
                edges.append([src, src])
                old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])
                edges_pair.append([src_word_old, src_word_old])

            return edges, old_edge_id, edges_pair
        else:
            if self.edges == 0:
                for index, src_word_old in enumerate(doc_ids):
                    src = old_to_new[src_word_old]
                    for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                        dst_word_old = doc_ids[i]
                        dst = old_to_new[dst_word_old]
                        if self.edges_matrix[src_word_old, dst_word_old] != 0:
                            # - first connect the new sub_graph
                            edges.append([src, dst])
                            # - then get the hidden from parent_graph
                            old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
                            edges_pair.append([src_word_old, dst_word_old])
                    # self circle
                    edges.append([src, src])
                    old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])
                    edges_pair.append([src_word_old, src_word_old])

                return edges, old_edge_id, edges_pair
            elif self.edges == 6:
                edges = []
                edges_pair = []
                old_edge_id = []
                for src_index, src_word_old in enumerate(doc_ids):
                    for dst_index, dst_word_old in enumerate(doc_ids):
                        src = old_to_new[src_word_old]
                        dst = old_to_new[dst_word_old]
                        try:
                            if adj_doc[src_index, dst_index] == 1 and self.edges_matrix[
                                src_word_old, dst_word_old] != 0:
                                edges.append([src, dst])
                                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
                                edges_pair.append([src_word_old, dst_word_old])
                        except Exception:
                            print(len(doc_ids))
                            print(len(adj_doc))
                    if self.global_add_local:
                        for i in range(max(0, src_index - self.ngram), min(src_index + self.ngram + 1, len(doc_ids))):
                            if self.global_matrix[src_word_old, dst_word_old] == 0:
                                dst_word_old = doc_ids[i]
                                dst = old_to_new[dst_word_old]
                                if self.edges_matrix[src_word_old, dst_word_old] != 0:
                                    # - first connect the new sub_graph
                                    edges.append([src, dst])
                                    # - then get the hidden from parent_graph
                                    old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])
                                    edges_pair.append([src_word_old, dst_word_old])

                return edges, old_edge_id, edges_pair

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]
        local_vocab = set(doc_ids)
        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph()
        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)

        if self.lstm_encoder:
            len_list = []
            len_list.append(len(local_node_hidden))
            lstm_embeddings = torch.unsqueeze(local_node_hidden, 0)
            outputs, (_, _) = self.lstm(lstm_embeddings)

            outputs = torch.squeeze(outputs)
            if len(outputs.size()) == 1:
                outputs = torch.unsqueeze(outputs, 0)
            self.sub_node_embedding = outputs
            sub_graph = sub_graph.to(
                torch.device(local_node_hidden.device.type + ':' + str(local_node_hidden.device.index)))

            sub_graph.ndata['h'] = outputs
        else:
            self.sub_node_embedding = local_node_hidden
            sub_graph = sub_graph.to(
                torch.device(local_node_hidden.device.type + ':' + str(local_node_hidden.device.index)))
            sub_graph.ndata['h'] = local_node_hidden

        seq_edges, seq_old_edges_id, edges_pair = self.add_seq_edges(doc_ids, old_to_new)  # 添加边

        edges, old_edge_id = [], []
        old_edge_id.extend(seq_old_edges_id)
        edges.extend(seq_edges)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)
        if edges:
            srcs, dsts = zip(*edges)
            sub_graph.add_edges(srcs, dsts)
        try:
            if self.adapte_edge:
                self.sub_edge_embedding = self.ori_edge_ebeding[old_edge_id, :]
                self.sub_adapted_egdes = self.relu(
                    self.edge_modifiers[old_edge_id, :].mul(self.seq_edge_w(old_edge_id)))
                if self.add_regu_loss == 2:
                    self.sub_embedding_dist = self.cal_sub_embedding_dist(edges_pair)
                seq_edges_w = self.sub_adapted_egdes
            else:
                seq_edges_w = self.seq_edge_w(old_edge_id)

        except RuntimeError:
            print(old_edge_id)
        sub_graph.edata['w'] = seq_edges_w

        return sub_graph

    def forward(self, doc_ids, adj=None, is_20ng=None):
        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]
        batch_graph = dgl.batch(sub_graphs)

        if self.adopt_gat:
            for l in range(self.num_layers):
                h = self.gat_layers[l](batch_graph, batch_graph.ndata['h']).flatten(1)
            batch_graph.ndata['h'] = h

        else:
            reduce_func_dic = {0: dgl.function.max('weighted_message', 'h'),
                               1: dgl.function.mean('weighted_message', 'h'),
                               2: dgl.function.sum('weighted_message', 'h')}

            batch_graph.update_all(
                message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
                reduce_func=reduce_func_dic[self.reduce]
            )
            if self.two_layer:
                h = batch_graph.ndata['h']
                h = self.L1_linear(h)
                h = F.tanh(h)
                batch_graph.ndata['h'] = h
                batch_graph.update_all(
                    message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
                    reduce_func=reduce_func_dic[self.reduce]
                )
        if self.readout == 0:  # sub-graph aggregation
            h1 = dgl.sum_nodes(batch_graph, feat='h')
        elif self.readout == 1:  # mean aggregation
            h1 = dgl.mean_nodes(batch_graph, feat='h')
        elif self.readout == 2:  # max aggregation
            h1 = dgl.max_nodes(batch_graph, feat='h')
        elif self.readout == 3:  # rcnn aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.rcnn_readout(graph_list)
        elif self.readout == 4:  # lstm aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.lstm_readout(graph_list)
        elif self.readout == 5:  # cnn aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.cnn_readout(graph_list)
        elif self.readout == 6:  # rnncnn aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.rnncnn_readout(graph_list)
        elif self.readout == 7:  # attention aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.att(graph_list)
        elif self.readout == 8:  # trans aggregation
            graph_list = dgl.unbatch(batch_graph)
            h1 = self.trans(graph_list)
        else:
            print('readout error')

        if self.readout in [0, 1, 2]:
            drop1 = self.dropout(h1)
            act1 = self.activation(drop1)
        elif self.readout in [3, 4]:
            drop1 = self.dropout(h1)
            act1 = self.activation(drop1)
        elif self.readout == 5:
            drop1 = self.dropout(h1)
            act1 = drop1
        elif self.readout == 6:
            drop1 = self.dropout(h1)
            act1 = drop1
        elif self.readout == 7:
            drop1 = self.dropout(h1)
            act1 = drop1
        elif self.readout == 8:
            drop1 = self.dropout(h1)
            act1 = drop1
        l = self.Linear(act1)
        return l
