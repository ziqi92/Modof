import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import math, random, sys
from argparse import ArgumentParser
from collections import deque

import copy
import multiprocessing as mp
from multiprocessing import Pool
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import sascorer
from concurrent import futures
from mol_tree import MolTree
from molopt import MolOpt
from vocab import Vocab, common_atom_vocab
from properties import similarity, get_prop
from functools import partial
import numpy as np
import pdb
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tree(smiles, assm=True):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol_tree = MolTree(smiles)
    return mol_tree

#def get_prop(smiles
def predict(smiles, model, vocab, avocab, reselect, ori_smiles, iternum, prop="logp"):
    mol = Chem.MolFromSmiles(smiles)
    atomnum1 = mol.GetNumAtoms()
    tree = get_tree(smiles)
    score1 = get_prop(smiles, prop=prop)

    try:    
        xbatch = MolTree.tensorize([tree], vocab, avocab, target=False)
        new_smiles, sim, reselect, score11, score2 = model.test(xbatch, tree, reselect_num=reselect, prop=prop)
    except:
        ori_sim = similarity(smiles, ori_smiles, "binary")
        result = [smiles, smiles, 1.0, ori_sim, score1, score1, atomnum1, atomnum1]
        return result
    
    if smiles == new_smiles:
        ori_sim = similarity(smiles, ori_smiles, "binary")
    else:
        ori_sim = similarity(new_smiles, ori_smiles, "binary")
    
    atomnum2 = Chem.MolFromSmiles(new_smiles).GetNumAtoms()
    result = [smiles, new_smiles, sim, ori_sim, score1, score2, atomnum1, atomnum2]
    return result

def output_iter(res, res_file, prop="logp", threshold=0):
    if len(res) == 0:
        s = "num: 0 sucess: 0.0 avg(imp): 0.00(0.00) avg(ori_imp): 0.00(0.00) avg(sim): 0.00(0.00) avg(ori_sim): 0.00(0.00) avg(atom):0.00 avg(newatom):0.00 \n"
    else:
        avg_imp = sum([x[1] for x in res])/len(res)
        std_imp = math.sqrt(sum([(x[1] - avg_imp)**2 for x in res])/len(res))
        avg_sim = sum([x[3] for x in res])/len(res)
        std_sim = math.sqrt(sum([(x[3] - avg_sim)**2 for x in res])/len(res))

        avg_ori_imp = sum([x[2] for x in res])/len(res)
        std_ori_imp = math.sqrt(sum([(x[2] - avg_ori_imp)**2 for x in res])/len(res))
        avg_ori_sim = sum([x[4] for x in res])/len(res)
        std_ori_sim = math.sqrt(sum([(x[4] - avg_ori_sim)**2 for x in res])/len(res))

        if prop != "logp":
            success_rate = len([x for x in res if x[9] > threshold]) / len(res)
        else:
            success_rate = 0.0
        
        s = "num: %d success: %.4f avg(imp): %.2f(%.2f) avg(ori_imp): %.2f(%.2f) avg(sim): %.2f(%.2f) avg(ori_sim): %.2f(%.2f) avg(atom):%.2f avg(newatom):%.2f \n" % (len(res), success_rate, avg_imp, std_imp, avg_ori_imp, std_ori_imp, avg_sim, std_sim, avg_ori_sim, std_ori_sim, sum([x[7] for x in res])/len(res),sum([x[10] for x in res])/len(res))

    res_file.write(s)
    res_file.flush()

def output_best(best_res, res_file, prop="logp", threshold=0):
    if len(best_res) == 0:
        s = "num: 0 success: %.4f avg(imp): 0.00(0.00) avg(sim): 0.00(0.00) avg(atom):0.00 avg(newatom):0.00 \n" 
    else:
        avg_imp = sum([x[4] for x in best_res])/len(best_res)
        std_imp = math.sqrt(sum([(x[4] - avg_imp)**2 for x in best_res])/len(best_res))
        avg_sim = sum([x[6] for x in best_res])/len(best_res)
        std_sim = math.sqrt(sum([(x[6] - avg_sim)**2 for x in best_res])/len(best_res))
        
        if prop != "logp":
            success_rate = len([x for x in best_res if x[3] > threshold]) / len(best_res)
        else:
            success_rate = 0.0
        
        s = "num: %d success: %.4f avg(imp): %.2f(%.2f) avg(sim): %.2f(%.2f) avg(atom):%.2f avg(newatom):%.2f \n" % (len(best_res), success_rate, avg_imp, std_imp, avg_sim, std_sim, sum([x[8] for x in best_res])/len(best_res), sum([x[9] for x in best_res])/len(best_res))
    res_file.write(s)
    res_file.flush()
    


    
parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test_path")
parser.add_argument("-v", "--vocab", dest="vocab_path")
parser.add_argument("-m", "--model", dest="model_path")
parser.add_argument("-d", "--save_dir", dest="save_dir")
parser.add_argument("-st", "--start", type=int, dest="start", default=0)
parser.add_argument("-si", "--size", type=int, dest="size", default=0)
parser.add_argument('-is', '--iter_size', default=5)

parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("-r", "--reselect", type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embed_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthG', type=int, default=5)
parser.add_argument('--depthT', type=int, default=3)
parser.add_argument('--iternum', type=int, default=5)
parser.add_argument('--out_size', type=int, default=20)

parser.add_argument('--ncpus', type=int, default=mp.cpu_count())
parser.add_argument('--lr', type=float, default=2)
parser.add_argument('--num', type=int, default=20)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)
parser.add_argument('--score_func', type=int, default=1)

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)
parser.add_argument("-s", "--sim", type=float, dest="cutoff", default=0.3)
parser.add_argument("-p", "--prop", type=str, default="logp")
parser.add_argument("-th", "--threshold", type=float, default=0.9)
opts = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

sim_cutoff = float(opts.cutoff)

try:
    model = MolOpt(vocab, common_atom_vocab, opts)
    model.load_state_dict(torch.load(opts.model_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")

output_path = opts.save_dir+opts.model_path.split("-")[1]+"_"+str(opts.reselect)+"_%.1f"%(opts.cutoff)
res_file = open("%s_iter%d_result.txt" % (output_path,opts.iternum), 'w')

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

res = [[] for _ in range(opts.iternum)]
best_res = []
outfiles = [open("%s_iter%d.txt" % (output_path, i), 'w') for i in range(opts.iternum)]
output = []
best_smiles = []
time1 = time.time()
start = int(opts.start)
end = int(opts.size) + start if opts.size > 0 else len(data)
large_neg = -1000
# optimize input molecules
 
for smiles in data[start:end]:
    best = np.ones((opts.out_size, 10)) * large_neg
    best_smiles = [(None, None, smiles) for _ in range(opts.out_size)]
    sort_idx = np.argsort(best[:, 3])
    
    next_smiles = [smiles]
    
    ori_score = None
    # Optimize the input molecule for multiple iterations
    # The optimization will stop if the model cannot produce molecules with better properties at one iteration.
    # The molecules with the best properties among all iterations will be the final output.
    for i in range(opts.iternum):
        #
        iter_smiles = []
        for next_cand in next_smiles:
            if next_cand is None: continue
        
            iter_smiles.extend([next_cand] * opts.num)

        if len(iter_smiles) == 0: break
        # In each iteration, randomly optimize the molecule for "num" times
        results = []
        for smile in iter_smiles:
            result = predict(smile, model, vocab, common_atom_vocab, 
                                 opts.reselect, smiles, i, prop=opts.prop)
            results.append(result)

        next_result = np.ones((opts.iter_size, 10)) * large_neg
        next_smiles = [None for _ in range(opts.iter_size)]
        sort_next_idx = np.arange(opts.iter_size)
        
        # Save the best optimized molecule
        
        iter_best = None
        iter_best_score = large_neg
        for result in results:
            smile1, smile2, sim, ori_sim, score1, score2, atom1, atom2 = result
            
            if opts.prop == "logp" and atom2 > 38: continue
            if sim == 1.0 and atom1 == atom2: continue
            
            if ori_score is None:
                ori_score = score1
                best[:, 1] = score1
            
            if score2 > iter_best_score and ori_sim >= opts.cutoff:
                iter_best = [i, score2 - score1, score2 - ori_score, sim, ori_sim,  smile1, score1, atom1, smile2, score2, atom2, smiles]
                iter_best_score = score2
            
            tmp_smiles = [tmp[1] for tmp in best_smiles]
            if score2 > best[sort_idx[0], 3] and smile2 not in tmp_smiles and ori_sim >= opts.cutoff:
                best_smiles[sort_idx[0]] = (smile1, smile2, smiles)
                best[sort_idx[0], :] = [i, ori_score, score1, score2, score2 - ori_score, score2 - score1, ori_sim, sim, atom1, atom2]
                
                sort_idx = np.argsort(best[:, 3])
                
            if score2 - score1 > next_result[sort_next_idx[0], 5] and ori_sim >= opts.cutoff and smile2 not in next_smiles and smile2 not in tmp_smiles:
                next_smiles[sort_next_idx[0]] = smile2
                next_result[sort_next_idx[0], :] = [i, ori_score, score1, score2, score2 - ori_score, score2 - score1, ori_sim, sim, atom1, atom2]

                sort_next_idx = np.argsort(next_result[:, 3])

            s = "%s %s %s %.4f %.4f %.4f %.4f %d %d\n" % (smiles, *result)
            print(s)
            outfiles[i].write(s)
        
        if iter_best is None: 
            iter_best = [i, 0.0, results[0][4] - ori_score, results[0][2], results[0][3], results[0][0], results[0][4], results[0][6], results[0][0], results[0][4], results[0][6], smiles]

        non_smiles = [s for s in next_smiles if s is not None]
        
        outfiles[i].flush()
        res[i].append( iter_best )
        s = "iter:%d num: %d imp: %.4f ori_imp: %.4f sim: %.4f ori_sim: %.4f smile: %s score: %.4f atomnum: %d new_smile: %s new_score: %.4f new_atomnum: %d ori_smile: %s\n" % (iter_best[0], len(results), *iter_best[1:])
        print(s)
        res_file.write(s)
        res_file.flush()
    
    time2 = time.time()
    print("time: %s" % str(time2-time1))
    time1 = time2
    best_res.append( (best_smiles, best) )


# ========================================== Output iteration results =============================================

for i in range(opts.iternum):
    outfiles[i].close()
    res_file.write("iter%d:\n" % i)

    # Total number
    res_file.write("[total]\n")
    output_iter(res[i], res_file, prop=opts.prop, threshold=opts.threshold) 
    
    # Positive
    res_file.write("[positive]\n")
    tmp_res = [x for x in res[i] if x[1] > 0]
    output_iter(tmp_res, res_file, prop=opts.prop, threshold=opts.threshold)

    # Negative
    res_file.write("[negative]\n")
    tmp_res = [x for x in res[i] if x[1] < 0]
    output_iter(tmp_res, res_file, prop=opts.prop, threshold=opts.threshold)
    
    # Zero
    res_file.write("[zero]\n")
    tmp_res = [x for x in res[i] if x[1] == 0]
    output_iter(tmp_res, res_file, prop=opts.prop, threshold=opts.threshold)
     
# =========================================== Output the best results ============================================

# Total
res_file.write("[best total]\n")
tmp_res = [res[np.argmax(res[:, 3]),:] for _, res in best_res if np.max(res[:, 3]) > 0]
output_best(tmp_res, res_file, prop=opts.prop, threshold=opts.threshold)

# Positive
res_file.write("[best positive]\n")
ptmp_res = [res for res in tmp_res if res[3] > 0]
output_best(ptmp_res, res_file, prop=opts.prop, threshold=opts.threshold)

# Negative
res_file.write("[best negative]\n")
ntmp_res = [res for res in tmp_res if res[3] < 0]
output_best(ntmp_res, res_file, prop=opts.prop, threshold=opts.threshold)

# Zero
res_file.write("[best zero]\n")
ztmp_res = [res for res in tmp_res if res[3] == 0]
output_best(ztmp_res, res_file, prop=opts.prop, threshold=opts.threshold)
res_file.close()

# Save the optimized molecule into output file
smiles_file = open("%s_iter%d_smiles.txt" % (output_path, opts.iternum), 'w')
for smiles, result in best_res:
    for i in range(len(smiles)):
        smile = smiles[i]
        res = result[i]

        if smile[0] is None:
            string = "%s %s 0.00 %.4f %.4f 0.0000 -1\n" % (smile[2], smile[2], res[1], res[1]) 
        else:
            string = "%s %s %.2f %.4f %.4f %.4f %d\n" % (smile[2], smile[1], res[6], res[1], res[3], res[4], res[0])
        smiles_file.write(string)
smiles_file.close()
