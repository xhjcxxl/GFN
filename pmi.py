# -*- coding: utf-8 -*-
# @Time    : 2020-03-10 09:32
# @Author  : dai yong
# @File    : pmi.py


from data_helper import DataHelper
import numpy as np
import torch
import word2vec
import os
import spacy


class EdgecalHelper(object):
    """
    get edges and nodes in graph
    """
    def __init__(self, dir_base='./'):
        self.dir_base = dir_base

    def cal_PMI(self, dataset: str, window_size=20):
        """
        :param dataset: the sentences need to build graph
        :param window_size: the windows size to build matrix
        :return:
        """
        helper = DataHelper(dataset=dataset, mode="train")
        content, _ = helper.get_content()
        pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        word_count = np.zeros(len(helper.vocab), dtype=int)
        global_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)

        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy'):
            for sentence in content:
                sentence = sentence.split(' ')
                for i, word in enumerate(sentence):
                    try:
                        word_count[helper.d[word]] += 1
                    except KeyError:
                        continue
                    start_index = max(0, i - window_size)
                    end_index = min(len(sentence), i + window_size)
                    for j in range(start_index, end_index):
                        if i == j:
                            continue
                        else:
                            target_word = sentence[j]
                            try:
                                pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                            except KeyError:
                                continue

            total_count = np.sum(word_count)
            word_count = word_count / total_count
            pair_count_matrix = pair_count_matrix / total_count

            pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)  # 计算PMI
            for i in range(len(helper.vocab)):
                for j in range(len(helper.vocab)):
                    pmi_matrix[i, j] = np.log(
                        pair_count_matrix[i, j] / (word_count[i] * word_count[j])
                    )
            pmi_matrix = np.nan_to_num(pmi_matrix)
            pmi_matrix = np.maximum(pmi_matrix, 0.0)
            np.save(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy', pmi_matrix)
        else:
            pmi_matrix = np.load(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy')

        edges_weights = [0.0]
        count = 1
        edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        for i in range(len(helper.vocab)):
            for j in range(len(helper.vocab)):
                if pmi_matrix[i, j] != 0:
                    global_matrix[i, j] += 1
                    edges_weights.append(pmi_matrix[i, j])
                    edges_mappings[i, j] = count
                    count += 1

        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)
        return edges_weights, edges_mappings, count, global_matrix

    def cal_PMI_ngram(self, ngram, dataset: str, window_size=20):
        helper = DataHelper(dataset=dataset, mode="train")
        content, _ = helper.get_content()
        pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        word_count = np.zeros(len(helper.vocab), dtype=int)
        global_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)

        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy'):
            for sentence in content:
                sentence = sentence.split(' ')
                for i, word in enumerate(sentence):
                    try:
                        word_count[helper.d[word]] += 1
                    except KeyError:
                        continue
                    start_index = max(0, i - window_size)
                    end_index = min(len(sentence), i + window_size)
                    for j in range(start_index, end_index):
                        if i == j:
                            continue
                        else:
                            target_word = sentence[j]
                            try:
                                pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                            except KeyError:
                                continue

            total_count = np.sum(word_count)
            word_count = word_count / total_count
            pair_count_matrix = pair_count_matrix / total_count

            pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
            for i in range(len(helper.vocab)):
                for j in range(len(helper.vocab)):
                    pmi_matrix[i, j] = np.log(
                        pair_count_matrix[i, j] / (word_count[i] * word_count[j])
                    )

            pmi_matrix = np.nan_to_num(pmi_matrix)
            pmi_matrix = np.maximum(pmi_matrix, 0.0)
            np.save(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy', pmi_matrix)
        else:
            pmi_matrix = np.load(self.dir_base + f'distance_matrix/{dataset}_pmi_distance_whole.npy')
        edges_weights = [0.0]
        count = 1
        edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        for i in range(len(helper.vocab)):
            for j in range(len(helper.vocab)):
                if pmi_matrix[i, j] >= 8:
                    global_matrix[i, j] += 1
                    edges_weights.append(pmi_matrix[i, j])
                    edges_mappings[i, j] = count
                    count += 1
        print(f'there are {count} globle nodes')
        for doc_ids in helper.content:
            for index, src_word in enumerate(doc_ids):
                for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                    dst_word = doc_ids[i]
                    if edges_mappings[src_word, dst_word] == 0:
                        edges_weights.append(pmi_matrix[src_word, dst_word].item())
                        edges_mappings[src_word, dst_word] = count
                        count += 1
        edges_weights = np.array(edges_weights)

        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)

        return edges_weights, edges_mappings, count, global_matrix

    def cal_co_occurrence(self, ngram, dataset: str, window_size=20):
        helper = DataHelper(dataset, mode="train")
        content, _ = helper.get_content()
        pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        word_count = np.zeros(len(helper.vocab), dtype=int)
        global_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)

        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_co_distance_whole.npy'):
            for sentence in content:
                sentence = sentence.split(' ')
                for i, word in enumerate(sentence):
                    try:
                        word_count[helper.d[word]] += 1
                    except KeyError:
                        continue
                    start_index = max(0, i - window_size)
                    end_index = min(len(sentence), i + window_size)
                    for j in range(start_index, end_index):
                        if i == j:
                            continue
                        else:
                            target_word = sentence[j]
                            try:
                                pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                            except KeyError:
                                continue

            total_count = np.sum(word_count)
            word_count = word_count / total_count
            pair_count_matrix = pair_count_matrix / total_count

            pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
            for i in range(len(helper.vocab)):
                for j in range(len(helper.vocab)):
                    pmi_matrix[i, j] = pair_count_matrix[i, j] / word_count[i]
            pmi_matrix = np.nan_to_num(pmi_matrix)

            pmi_matrix = np.maximum(pmi_matrix, 0.0)
            np.save(self.dir_base + f'distance_matrix/{dataset}_co_distance_whole.npy', pmi_matrix)
        else:
            pmi_matrix = np.load(self.dir_base + f'distance_matrix/{dataset}_co_distance_whole.npy')

        edges_weights = [0.0]
        count = 1
        edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
        for i in range(len(helper.vocab)):
            for j in range(len(helper.vocab)):
                if pmi_matrix[i, j] >= 0.2:
                    global_matrix[i, j] += 1
                    edges_weights.append(pmi_matrix[i, j])
                    edges_mappings[i, j] = count
                    count += 1
        print(f'there are {count} global items')
        for doc_ids in helper.content:
            for index, src_word in enumerate(doc_ids):
                for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                    dst_word = doc_ids[i]
                    if edges_mappings[src_word, dst_word] == 0:
                        edges_weights.append(pmi_matrix[src_word, dst_word].item())
                        edges_mappings[src_word, dst_word] = count
                        count += 1

        edges_weights = np.array(edges_weights)

        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)

        return edges_weights, edges_mappings, count, global_matrix

    def cal_cosine_similarity(self, dataset, vocab, node_hidden, threshold, ngram):
        cos_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        global_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        vocab_size = len(vocab)
        edges_weights = [0.0]
        count = 1
        local_count = 1
        edges_mappings = np.zeros((len(vocab), len(vocab)), dtype=int)
        helper = DataHelper(dataset=dataset, mode="train")

        '''store the whole cosine matrix to accelerate the running speed'''
        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_cosine_distance_whole.npy'):
            for i in range(vocab_size):
                for j in range(vocab_size):
                    vector_i = node_hidden(torch.LongTensor([i])).detach().numpy()
                    vector_j = node_hidden(torch.LongTensor([j])).detach().numpy()
                    vector_i = np.squeeze(vector_i)
                    vector_j = np.squeeze(vector_j)
                    similarity = (float(np.dot(vector_i, vector_j)) /
                                  (np.linalg.norm(vector_i) * np.linalg.norm(vector_j)))
                    cos_matrix[i, j] = similarity
            np.save(self.dir_base + f'distance_matrix/{dataset}_cosine_distance_whole.npy', cos_matrix)
        else:
            cos_matrix = np.load(self.dir_base + f'distance_matrix/{dataset}_cosine_distance_whole.npy')
        k = int(len(cos_matrix) * 0.005)
        if k != 0:
            cos_matrix = torch.from_numpy(cos_matrix)
            top20_edge = torch.topk(cos_matrix, k)
            top20_edge_values = top20_edge[0]
            top20_edge_indices = top20_edge[1]

        if k != 0:
            for i in range(top20_edge_values.shape[0]):
                for j in range(top20_edge_values.shape[1]):
                    if top20_edge_values[i, j] != 0 and top20_edge_values[i, j] < 1:
                        edges_weights.append(top20_edge_values[i, j].item())
                        edges_mappings[i, top20_edge_indices[i, j].item()] = count
                        global_matrix[i, top20_edge_indices[i, j].item()] += 1
                        count += 1
            print(f'the top cosine include {count} items')

            for doc_ids in helper.content:
                for index, src_word in enumerate(doc_ids):
                    for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                        dst_word = doc_ids[i]
                        if edges_mappings[src_word, dst_word] == 0:
                            edges_weights.append(cos_matrix[src_word, dst_word].item())
                            edges_mappings[src_word, dst_word] = count
                            count += 1
                            local_count += 1
        else:
            for i in range(vocab_size):
                for j in range(vocab_size):
                    similarity = cos_matrix[i, j]
                    if similarity > threshold:
                        edges_weights.append(cos_matrix[i, j])
                        edges_mappings[i, j] = count
                        count += 1
                    else:
                        cos_matrix[i, j] = 0

        edges_weights = np.array(edges_weights)

        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)

        return edges_weights, edges_mappings, count, global_matrix
    # calculate distance
    def cal_euclidean_distance(self, dataset, vocab, node_hidden, threshold, ngram):
        helper = DataHelper(dataset=dataset, mode="train")
        cos_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        global_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        vocab_size = len(vocab)
        edges_weights = [0.0]
        count = 1
        local_count = 1
        edges_mappings = np.zeros((len(vocab), len(vocab)), dtype=int)
        '''store the whole euclidean matrix to accelerate the running speed'''
        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_euclidean_distance_whole.npy'):
            for i in range(vocab_size):
                for j in range(vocab_size):
                    vector_i = node_hidden(torch.LongTensor([i])).detach().numpy()
                    vector_j = node_hidden(torch.LongTensor([j])).detach().numpy()
                    vector_i = np.squeeze(vector_i)
                    vector_j = np.squeeze(vector_j)
                    similarity = np.linalg.norm(vector_i - vector_j)
                    cos_matrix[i, j] = similarity
            np.save(self.dir_base + f'distance_matrix/{dataset}_euclidean_distance_whole.npy', cos_matrix)
        else:
            cos_matrix = np.load(self.dir_base + f'distance_matrix/{dataset}_euclidean_distance_whole.npy')

        k = int(len(cos_matrix) * 0.005)
        if k != 0:
            cos_matrix = torch.from_numpy(cos_matrix)
            sorted_matrix = cos_matrix.sort(dim=1)
            top20_edge_values = sorted_matrix[0]
            top20_edge_indices = sorted_matrix[1]

        if k != 0:
            for i in range(top20_edge_values.shape[0]):
                for j in range(k):
                    # if the value = 0, then the distance is zero
                    if top20_edge_values[i, j] != 0:
                        edges_weights.append(top20_edge_values[i, j].item())
                        edges_mappings[i, top20_edge_indices[i, j].item()] = count
                        global_matrix[i, top20_edge_indices[i, j].item()] += 1
                        count += 1
            print(f'the global items is {count}')
            for doc_ids in helper.content:
                for index, src_word in enumerate(doc_ids):
                    for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                        dst_word = doc_ids[i]
                        if edges_mappings[src_word, dst_word] == 0:
                            edges_weights.append(cos_matrix[src_word, dst_word].item())
                            edges_mappings[src_word, dst_word] = count
                            count += 1
                            local_count += 1
        else:
            for i in range(vocab_size):
                for j in range(vocab_size):
                    similarity = cos_matrix[i, j]
                    if similarity < threshold:
                        edges_weights.append(cos_matrix[i, j])
                        edges_mappings[i, j] = count
                        count += 1
                    else:
                        cos_matrix[i, j] = -1

        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)

        return edges_weights, edges_mappings, count, global_matrix

    def cal_syntactic(self, ngram, helper):
        global_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
        edges_weights = [0.0]
        count = 1
        edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)

        nlp = spacy.load('en_core_web_sm')
        docs, _ = helper.get_content()
        docs_tokens = helper.content
        for text, tokens in zip(docs, docs_tokens):
            document = nlp(text)
            seq_len = len(text.split())
            for token in document:
                if token.i < seq_len:
                    src_word = tokens[token.i]
                    if edges_mappings[src_word][src_word] == 0:
                        edges_weights.append(1)
                        edges_mappings[src_word][src_word] = count
                        global_matrix[src_word, src_word] += 1
                        count += 1
                    for child in token.children:
                        if child.i < seq_len:
                            dst_word = tokens[child.i]
                            if edges_mappings[src_word, dst_word] == 0:
                                edges_weights.append(1)
                                edges_mappings[src_word, dst_word] = count
                                global_matrix[src_word, src_word] += 1
                                count += 1
                                edges_weights.append(1)
                                edges_mappings[dst_word, src_word] = count
                                global_matrix[src_word, src_word] += 1
                                count += 1
        for doc_ids in helper.content:
            for index, src_word in enumerate(doc_ids):
                for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                    dst_word = doc_ids[i]
                    if edges_mappings[src_word, dst_word] == 0:
                        edges_weights.append(1)
                        edges_mappings[src_word, dst_word] = count
                        count += 1

        edges_weights = np.array(edges_weights)
        edges_weights = edges_weights.reshape(-1, 1)
        edges_weights = torch.Tensor(edges_weights)
        return edges_weights, edges_mappings, count, global_matrix

    def init_node_embeddings(self, dataset, hidden_size_node, vocab):
        node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)
        if not os.path.exists(self.dir_base + f'distance_matrix/{dataset}_node_embeddings.npy'):
            node_embeddings = load_word2vec(vocab, 'glove.6B.300d.txt')
            node_hidden.weight.data.copy_(torch.tensor(node_embeddings))
            np.save(self.dir_base + f'distance_matrix/{dataset}_node_embeddings.npy', node_embeddings)
        else:
            node_embeddings = np.load(self.dir_base + f'distance_matrix/{dataset}_node_embeddings.npy')
            node_hidden.weight.data.copy_(torch.tensor(node_embeddings))
        node_hidden.weight.requires_grad = True

        return node_embeddings, node_hidden

    def init_edge_embeddings(self, edges, edges_weights, trainable_edges):
        if trainable_edges:
            trainable_edges = False
        else:
            trainable_edges = True
        if edges == 0:
            seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(len(edges_weights), 1), freeze=trainable_edges)
        elif edges in [1, 2, 3, 4, 5, 6]:
            seq_edge_w = torch.nn.Embedding.from_pretrained(edges_weights, freeze=trainable_edges)
        else:
            print('the edge type is error')
        return seq_edge_w


def load_word2vec(vocab, word2vec_file):
    """
    load word2vec matrix
    """
    model = word2vec.load(word2vec_file)
    embedding_matrix = []
    cnt = 0
    for word in vocab:
        try:
            embedding_matrix.append(model[word])
        except KeyError:
            embedding_matrix.append(model['the'])
            cnt += 1
    embedding_matrix = np.array(embedding_matrix)
    print('the embedding finding accuracy is:', (len(vocab) - cnt) / len(vocab))
    return embedding_matrix


def store_node_embeddings(training_embedding_dir, node_embedding, vocab):
    file_dir = training_embedding_dir
    embedding_size = len(vocab)
    first_row = str(len(vocab)) + ' ' + '200' + '\n'
    node_embedding_narray = node_embedding(torch.LongTensor(list(range(embedding_size))).cuda()).cpu().detach().numpy()
    word_embedding_list = []
    np.savetxt(file_dir + 'narray', node_embedding_narray)
    with open(file_dir + 'narray', 'r') as f:
        vocab_embedding = f.readlines()
        for id, word in enumerate(vocab):
            word_embedding = word + ' ' + vocab_embedding[id]
            word_embedding_list.append(word_embedding)
    i = 0
    with open(file_dir, 'w') as f:
        if i == 0:
            f.write(first_row)
            i = 1
        for item in word_embedding_list:
            f.write("%s" % item)


def Mask_ngram(helper, ngram):
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for doc_ids in helper.content:
        for index, src_word in enumerate(doc_ids):
            for i in range(max(0, index - ngram), min(index + ngram + 1, len(doc_ids))):
                dst_word = doc_ids[i]
                edges_mappings[src_word, dst_word] = 1
    return edges_mappings


if __name__ == "__main__":
    print('test')
    helper = EdgecalHelper()
    helper.cal_PMI('r8')
