# Molecule Optimization via Fragment-based Generative Models

This is the implementation of our Modof model: https://arxiv.org/abs/2012.04231. This paper has already been accepted by the journal "Nature Machine Intelligence". 

**Note**: This repository has been moved to https://github.com/ninglab/Modof. Please check this link for the most recent updates we have.



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

### 1.   Use provided processed dataset

If you want to use our provided processed dataset, please check the directories below: <code>./data/logp06/</code> : the dataset of pairs of molecules with 0.6 similarity and different on penalized logp property.

<code>./data/drd2_25</code> :  the dataset of pairs of molecules with 0.6 similarity and different on DRD2 property. The property difference between each pair of molecules is greater than 0.25.

<code>./data/qed_1</code> : the dataset of pairs of molecules with 0.6 similarity and different on QED property. The property difference between each pair of molecules is greater than 0.1. The QED properties of these molecules are greater than 0.7.

<code>./data/drd2_25_qed6</code> : the dataset of pairs of molecules with 0.6 similarity and different on QED property and DRD2 property. The property differences on DRD2 between each pair of molecules are greater than 0.25 (i.e., DRD2$(Y)-$DRD2$(X)\geq 0.25$ ). The QED property of each pair of molecules should satisfy $QED(X)<0.6\leq QED(Y)$.  



In each directory, you will see the following files:

1)   multiple zipped <code>tensors-\*.pkl</code> files. These binary files contain the processed data including pairs of molecules and their edit paths. The data in these <code>\*.pkl</code> files should be used for model training. All the <code>tensors-\*.pkl</code> files will be read into *Modof* as training data. If you are using your own training data rather than the provided one, you can generate such <code>tensors-\*.pkl</code> using the data processing tools as will be described below.  

Note: Due to the limit of file size, we only provide part of the processed file here. To use the whole training dataset, please use the provided data preprocessing script to preprocess the dataset. Please decompress the zipped file before using them to train the model.

2)   <code>train_pairs.txt</code> file in logp06 dataset. This file contains all pairs of molecules used in [Jin’s paper](https://arxiv.org/pdf/1812.01070.pdf). This file is identical to train_pairs.txt file in  (https://github.com/wengong-jin/iclr19-graph2graph/tree/master/data/logp06). Please note that the molecule pairs contained in <code>tensors-\*.pkl</code> files are a subset of all the molecule pairs in train_pairs.txt. 

* File format:  each line in <code>train_pairs.txt</code> has two SMILE strings, separated by an empty space. The first SMILE string represents the molecule with worse properties, and the second SMILE string represent the molecule with better properties. 

3)  <code>one_ds_pairs.txt</code> file. This file contains the pairs of molecules used in Modof.

- File format: each line in <code>one_ds_pairs.txt</code> has two SMILE strings, separated by an empty space.

4)   <code>test.txt</code>. This file contains the SMILE strings of single molecules that are used as the testing molecules in XXX’s paper. These molecules are also the testing molecules used in our Modof. 

* File format: each line in <code>test.txt</code> is a SMILE string of a testing molecule. 

5)   <code>vocab.txt</code>. This file contains all the substructures of training molecules in <code>tensors-*.pkl</code> files. These substructures are in SMILE strings. 

* File format: each line in vocab.txt is a SMILE string of a substructures. The *i*-th row represents the *i*-th substructure (i.e., ‘i’ here is the substructure ID). 

  

### 2.   Use your own data

f you want to train *Modof* using your own training data, you will need to process your own data into the same format as the processed data, respectively. All the code for data processing is provided under data_processing. 

To process your own data, run 

```
python ./data_preprocessing/preprocess.py --train train_file_name –-output out_file_name –-batch_size NNN --batch_id ZZZ 
```

where train_file_name is the name of the file that contains all the molecule pairs that should be used for *Modof* training. This file should have the same format as <code>train_pairs.txt</code> as above. 

For the <kbd>output</kbd> option, the above command (1) will generate n=(number of pairs) / NNN out_file_name-ZZZ.pkl files in the same directory as train_file_name. These files will be used in Modof training. For other options of this command, please check <code>–-help</code>.

<kbd>batch_size</kbd> and <kbd>batch_id</kbd> is recommended to use for large training dataset. If your training dataset is large, you can process batches of training data in a parallel way by running the above command multiple times with different batch_id. These two arguments are simply designed to speed up the data preprocessing for large dataset. If you have small training dataset, you can choose not to specify the value of <kbd>batch_size</kbd> and <kbd>batch_id</kbd>, and then the entire training data will be processed one time.

Note that the training pairs of molecules for *Modof* model are required to differ in terms of only one disconnection site. The training pairs which differ in multiple disconnection site will be filtered out by the above command. To get enough training pairs for *Modof* model, it is expected that the molecules in your own training data are very similar. 

Example:

```
python ./data_preprocessing/preprocess.py --train ./data/logp06/train_pairs.txt –-output new_tensors –-batch_size 10000 --batch_id 0 
```

## Training


To train our *Modof* model, run 

```
python ./model/train.py --depthT 3 --depthG 5 --hidden_size 64 --latent_size 8 --add_ds --beta 0.1 --warmup 2000 --beta_anneal_iter 500 --step_beta 0.05 --max_beta 0.5 --save_iter 3000 --print_iter 20 --train train_path --vocab vocab_path --save_dir model_path
```

<kbd>depthT</kbd>  specifies the depth of *tree* message passing neural networks.

<kbd>depthG</kbd>  specifies the depth of *graph* message passing neural networks.

<kbd>hidden_size</kbd> specifies the dimension of all hidden layers.

<kbd>latent_size</kbd> specifies the dimension of latent embeddings.

<kbd>add_ds</kbd> specifies whether or not to add the embedding of disconnection site into the latent embedding. This parameter is a bool value and will default to False when "--add_ds" is not present.

<kbd>beta</kbd> specifies the initial value of weight of KL loss in the total loss.

<kbd>warmup</kbd> specifies the number of steps that beta value remains unchanged at the beginning. (Each step represents an update of model on a single batch.)

<kbd>beta_anneal_iter</kbd> specifies the number of steps that beta value is reduced by a certain value after the number of training steps.

<kbd>step_beta</kbd> specifies the value used to reduce the value of beta.

<kbd>max_beta</kbd> specifies the maximum value of beta.

<kbd>save_iter</kbd> controls how often the model would be saved. In the above example, the model will be saved every 3,000 steps. The model will be saved at model_path/model.iter-*.pt

<kbd>print_iter</kbd> controls how often the intermediate result would be displayed (e.g., the accuracy of each prediction, the loss of each function).

<kbd>train</kbd> specifies the directory of training data. The program will extract the training pairs from all "pkl" files under this directory. The train path defaults to be the path of our provided dataset if not specified.

<kbd>vocab</kbd> specifies the path of vocabulary of all substructures in the training data. You can generate the vocab file for your own training data with the provided code as will be described below. The vocab path defaults to be the path of our provided vocab file if not specified.

<kbd>save_dir</kbd> specifies the path to save the trained model.  The model path defaults to be "./result" if not specified.

Use the command <code>python ./model/train.py -h</code> to check the meaning of other parameters.



**Generating Vocabulary file**: In the above command, the training of *Modof* model requires a vocabulary file that contains all the substructures in the molecules in all the training files under the train_path. This file should have the same format as in <code>vocab.txt</code> as above. 

To generate the vocab file for your own training data, run

```
python ./model/mol_tree.py --train train_path --out vocab_path
```



**Running time:** It can take no more than **4 hours** to train a *modof* model using a GPU with our provided training data for 6,000 steps.  Typically, our model can produce decent results with 6,000 steps of training.

## Test

To test a trained model, you can run the file <code>./model/optimize.py</code> with following command:

```
python ./model/optimize.py --test test_path --vocab vocab_path --model model_path --save_dir test_result_path --hidden_size 64 --latent_size 8 --depthT 3 --depthG 5 --iternum 5 --num N -s 0.6
```

Note that the option "hidden_size", "latent_size", "depthT" and "depthG" must be the same with the train model. 

<kbd>-s</kbd> specifies the similarity threshold of generated molecules.

<kbd>save_dir</kbd> specifies the path of results. All result files will be saved into the test_result_path directory

<kbd>test_path</kbd> specifies the path of test data. The test path defaults to be the provided test data (i.e., ./data/logp06/test.txt) if not specified.

<kbd>vocab_path</kbd> specifies the path of vocab file. The vocab path defaults to be the provided vocab file (i.e., ./data/logp06/vocab.txt) if not specified.

<kbd>num</kbd> specifies the number of latent embedding samples for each molecule at each iteration.
<kbd>iternum</kbd> specifies the number of iterations for optimization.



The outputs of the above command include:
* <code>*-iter[0-N].txt</code> include the optimization results of each input molecule with all the latent embedding samples.
* <code>*-iter[N]_results.txt</code> include the optimization results of each input molecule at all [num] iterations.
* <code>*-smiles.txt</code> include the best optimized molecules among all iterations, the property scores of these optimized molecules and the similarities of these optimized molecules with input molecules.



You can also run the file <code>optimize1.py</code> with the similar command to enable the Modof to optimize multiple best molecules at each iteration using the option <code>--iter_size</code> as below.

```
python ./model/optimize1.py --test test_path --vocab vocab_path --model model_path --save_dir test_result_path --hidden_size 64 --latent_size 8 --depthT 3 --depthG 5 --iternum 5 --num N -s 0.6 --iter_size 5
```

