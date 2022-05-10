# -*- coding: utf-8 -*-
# @Time    : 2020-04-25 20:19
# @Author  : dai yong
# @File    : transformer_20ng.py

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

file = 'oh'
num_labels_dic = {'oh': 23, 'r8': 8, 'r52': 52, 'mr': 2, 'sst5': 5, '20ng': 20}
with open(f'./data/{file}/label.txt') as f:
    labels = f.readlines()
label_dic = {}
for i, label in enumerate(labels):
    label = label.strip()
    label_dic[label] = i

label_list = label_dic
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
train_list = []
test_list = []
dev_list = []
with open(f'./data/{file}/{file}-train-stemmed.txt') as f:
    train_file = f.readlines()
    for doc in train_file:
        temp = doc.split('\t')
        temp[0], temp[1] = temp[1], int(label_list[temp[0]])

        train_list.append(temp)

with open(f'./data/{file}/{file}-test-stemmed.txt') as f:
    test_file = f.readlines()
    for doc in test_file:
        temp = doc.split('\t')
        temp[0], temp[1] = temp[1], int(label_list[temp[0]])
        test_list.append(temp)

with open(f'./data/{file}/{file}-dev-stemmed.txt') as f:
    dev_file = f.readlines()
    for doc in test_file:
        temp = doc.split('\t')
        temp[0], temp[1] = temp[1], int(label_list[temp[0]])
        dev_list.append(temp)

i = 0
result_list = []
for i in range(5):
    train_df = pd.DataFrame(train_list)
    eval_df = pd.DataFrame(test_list)

    model_args = ClassificationArgs()
    model_args.early_stopping_metric = 'eval_loss'
    model_args.use_early_stopping = True
    model_args.n_gpu = 1
    model_args.num_train_epochs = 10
    model_args.fp16 = False
    model_args.train_batch_size = 16
    model_args.use_multiprocessing = True
    model_args.overwrite_output_dir = True
    model_args.learning_rate = 2e-5
    model_args.output_dir = './transformer_output/'
    model_args.max_seq_length = 300
    model_args.reprocess_input_data = True

    # Create a ClassificationModel
    model = ClassificationModel('bert', 'bert-base-cased', num_labels=num_labels_dic[file], args=model_args)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    result_list.append(result['mcc'])
    i += 1

result_list = np.array(result_list)
print(result_list.mean())
print(np.std(result_list, ddof=1))
