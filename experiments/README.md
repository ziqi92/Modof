## Experiments

Due to the storage limitation of servers we used, all our files related to the experiments in the paper were accidentally deleted. Hence, the results here are **not from the exactly same model** we used to get the results in the paper. Instead, we trained a new model with the same hyper-parameters and tested the model with delta = 0.4/0.6 after 6000 training steps.

### Training

We used the command below to train the model:

```
python ./model/train.py --save_dir ./experiments --hidden_size 64 --latent_size 16 --embed_size 64 --depthT 3 --depthG 5 --add_ds --warmup 3000 --step_beta 0.05 --beta 0.1 --max_beta 0.5
```

<code>model.iter-6000</code> contains the model after 6000 training steps, while <code>model.iter-9000</code> contains the model after 9000 training steps.

<code>model.out</code> contains the log of training process. Note that the log file contains some molecules such as "O=C1OC2CC3CC2C1C3C(=O)Nc1cc(Cl)ccc1OC". This is due to the limitation of junction tree structure which cannot decompose or process these complex fused ring structures.  Hence, molecules with ring structures unable to be processed or decomposed by Junction Tree Decomposition proposed by paper [^1] cannot be the training data of our model. 

### Test

We used the command below to test the model:

```
python ./model/optimize.py -s 0.6 --save_dir ./experiments/ --embed_size 64 --hidden_size 64 --latent_size 16 --depthT 3 --depthG 5 --num 20 --iternum 5 -st 0 -si 800 --vocab ./data/logp06/vocab.txt  -m ./experiments/model.iter-6000 -t ./data/logp06/test.txt
```

The above command optimized the molecules with similarity threshold 0.6 and saved the optimized results into multiple files:

1. multiple <code>6000_1_0.6_iter\*.txt</code>  files. These files contain the optimized molecules with different samplings at each iteration.
2. <code>6000_1_0.6_iter5_result.txt</code> file. This file contains the *best* optimized molecules at each iteration.

3. <code>6000_1_0.6_iter5_smiles.txt</code> file. This file contains the overall best optimized molecules with at most 5 iterations.



[^1]: Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Junction tree variational autoencoder for molecular graph generation." *arXiv preprint arXiv:1802.04364* (2018).

