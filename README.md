# Graph Fusion Network for Text Classification（GFN）

The source code of our paper(Graph Fusion Network for Text Classification) presents in KBS 2022. https://doi.org/10.1016/j.knosys.2021.107659

## dataset process

we can use process all datasets:

`python pre_processing.py`
`python pmi.py`

## train GFN model

we used four datasets to train GFN model: mr r8 r52 oh

we can use the command to train GFN model, all parameters can be set。

Example for train MR dataset: mr edges: 4

`python nni_train_ada_regu.py --dataset mr --edges 4 --global_add_local 1 --global_edge 1 --adopt_nni 0 --adopt_nni_paras 0 --max_epoch 8  --adapte_edge 1`

Example for train R8 dataset: r8 edges: 4

`python nni_train_ada_regu.py --dataset r8 --edges 4 --global_add_local 1 --global_edge 1 --adopt_nni 0 --adopt_nni_paras 0 --max_epoch 8  --adapte_edge 1`

Example for train R52 dataset: r52 edges: 4

`python nni_train_ada_regu.py --dataset r52 --edges 4 --global_add_local 1 --global_edge 1 --adopt_nni 0 --adopt_nni_paras 0 --max_epoch 8  --adapte_edge 1`

Example for train OH dataset: oh edges: 4

`python nni_train_ada_regu.py --dataset oh --edges 4 --global_add_local 1 --global_edge 1 --adopt_nni 0 --adopt_nni_paras 0 --max_epoch 8  --adapte_edge 1`

## inference GFN model

Example for inference MR dataset:

`python learning_and_infer.py --dataset mr --ensemble_model 2 --out_channels 3 --pooler 2  --top_k 1`

Example for inference R8 dataset:

`python learning_and_infer.py --dataset r8 --ensemble_model 2 --out_channels 3 --pooler 2  --top_k 1`

Example for inference R52 dataset:

`python learning_and_infer.py --dataset r52 --ensemble_model 2 --out_channels 3 --pooler 2  --top_k 1`

Example for inference OH dataset:

`python learning_and_infer.py --dataset oh --ensemble_model 2 --out_channels 3 --pooler 2  --top_k 1`

## file description

- config.py : the model configure parameters
- pre_processing.py : the data preprocessing
- data_helper.py : the data process
- pmi.py : build PMI data in passages and generate distance matrix
- nni_train_ada_regu.py : train GFN model with adapted edges and regularizer
- learning_and_infer.py : inference GFN model in datasets
- model_ada.py : the GFN model with adapted
- transformer_20ng.py : test 20ng dataset in BERT model
- transformer_mr.py : test mr dataset in BERT model
- transformer_oh.py : test oh dataset in BERT model
- transformer_test.py : test test dataset in BERT model


## other note

Note: You need to put a pre-trained Glove model file named `glove.6B.300d.txt` in the root directory.
You get the file from https://nlp.stanford.edu/projects/glove/.

## Reference

If you find this project useful, please use the following format to cite the paper:

    @article{dai2022graph,
      title={Graph Fusion Network for Text Classification},
      author={Dai, Yong and Shou, Linjun and Gong, Ming and Xia, Xiaolin and Kang, Zhao and Xu, Zenglin and Jiang, Daxin},
      journal={Knowledge-Based Systems},
      volume={236},
      pages={107659},
      year={2022},
      publisher={Elsevier}
    }

## Contact information

For help or issues, please submit a GitHub issue.
