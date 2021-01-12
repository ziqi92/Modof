# Molecule Optimization via Fragment-based Generative Models


This is the implementation of our Modof model https://arxiv.org/pdf/2012.04231.pdf

## Requirements

Operating systems: Red Hat Enterprise Linux (RHEL) 7.7


* python==3.6.12

* scikit-learn==0.22.1

* networkx==2.4

* pytorch==1.5.1

* rdkit==2020.03.5

* scipy==1.4.1

  

## Installation guide

Download the code and dataset with the command:

```
git clone https://github.com/ziqi92/Modof.git
```

The download can take several minutes.



## Data processing

### Provided Processed Dataset

data/logp06/tensors-\*.pkl contains the processed data used in our paper. The raw dataset of this processed dataset is from https://github.com/wengong-jin/iclr19-graph2graph.

Each data point in these processed data files has three elements:

```
(<MolTree object for molecule x>, <MolTree object for molecule y>, <edit path between molecule x and y>)
# Check the file `model/mol_tree.py` for the class MolTree
```

<code>preprocess/</code> contains the code we used to preprocess the dataset.

To get the processed dataset we used, you can run

```
python ./preprocess/preprocess.py
```
The processed data will be saved in the directory <code>data/logp06</code>.



You can also use our code to process a new dataset (i.e., the dataset must be molecule pairs) by running the following command.

```
python ./preprocess/preprocess.py --train <train_path> --vocab <vocab_path>
```
The processed dataset will be saved in the same directory with <train_path>

## Training


Running example

```
python ./model/train.py --depthT 3 --depthG 5 --hidden_size 64 --latent_size 8 --add_ds --beta 0.1 --step_beta 0.05 --max_beta 0.5 --warmup 2000 --beta_anneal_iter 500 --save_iter 3000 --print_iter 20 --train ./data/logp06/
```

<kbd>save_iter</kbd> defines how often the model would be saved. In the above example, the model will be saved every 3,000 steps. The model will be saved at result/model.iter-*.pt

<kbd>print_iter</kbd> defines how often the intermediate result would be displayed (e.g., the accuracy of each prediction, the loss of each function).

Use the command <code>python ./model/train.py -h</code> to check the meaning of other parameters.

It can take no more than **4 hours** to train a *modof* model for 6,000 steps.  



## Test

To test a trained model, you can run the file <code>./model/optimize.py</code> with following command:

```
python ./model/optimize.py --test ./data/logp06/test.txt --vocab ./data/logp06/vocab.txt --model <model data path> -d <test result path> --hidden_size 64 --latent_size 8 --depthT 3 --depthG 5 --iternum 5 -st 0 -si 800 --num 20 -s 0.6
```

<code>-s similarity_threshold</code> controls the similarity threshold of generated molecules.
<code>-num value</code> controls the number of latent embedding samples for each molecule at each iteration
<code>-iternum value</code> controls the number of iterations.

The outputs include:
* <code>*-iter[0-num].txt</code> include the optimization results of each input molecule with all the latent embedding samples.
* <code>*-iter[num]_results.txt</code> include the optimization results of each input molecule at all [num] iterations.
* <code>*-smiles.txt</code> include the best optimized molecules among all iterations, the property scores of these optimized molecules and the similarities of these optimized molecules with input molecules.


