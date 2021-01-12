# Molecule Optimization via Fragment-based Generative Models


This is the implementation of our Modof model https://arxiv.org/pdf/2012.04231.pdf



# Requirements


* python==3.x
* scikit-learn==0.22.1

* networkx==2.4

* pytorch==1.5.1

* rdkit==2020.03.5

* scipy==1.4.1


# Data preprocessing
## Provided Processed Dataset
data/logp06/tensors-\*.pkl contains the processed data used in our paper. The raw dataset of this processed dataset is from https://github.com/wengong-jin/iclr19-graph2graph.
Each data point in these file has three elements:
```
(<MolTree object for molecule x>, <MolTree object for molecule y>, <edit path between molecule x and y>)
```

preprocess/ contains the code we used to get this processed dataset.
To get this processed dataset, you can run
```
python ./preprocess/preprocess.py
```
The processed dataset will be saved in directory "data/logp06".

You can also use this code to process a new dataset with molecule pairs by running
```
python ./preprocess/preprocess.py --train <train_path> --vocab <vocab_path>
```
The processed dataset will be saved in the same directory with <train_path>


# Training


Train *Modof* model with command below:

```
python ./model/train.py --depthT 3 --depthG 5 --hidden_size 64 --latent_size 8 --add_ds --beta 0.1 --step_beta 0.05 --max_beta 0.5 --warmup 2000 --beta_anneal_iter 500
```

The model will be saved at result/model.iter-*.pt





