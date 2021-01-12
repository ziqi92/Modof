import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import math, random, sys
from argparse import ArgumentParser
from collections import deque
import pdb
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import sascorer
from concurrent import futures
from mol_tree import MolTree
from molopt import MolOpt
from vocab import Vocab, common_atom_vocab
from properties import similarity, penalized_logp

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tree(smiles, assm=True):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol_tree = MolTree(smiles)
    return mol_tree

def predict(smiles, lr, vocab, avocab, reselect, ori_smiles, iternum, output):
    mol = Chem.MolFromSmiles(smiles)
    atomnum = mol.GetNumAtoms()
    tree = get_tree(smiles)
    score1 = penalized_logp(smiles)

    try:
        xbatch = MolTree.tensorize([tree], vocab, avocab, target=False)
        new_smiles, sim, reselect, score11, score2 = model.test(xbatch, tree, lr=lr, reselect_num=reselect)
    except:
        ori_sim = similarity(smiles, ori_smiles)
        return score1, score1, atomnum, smiles, 1.0, ori_sim, 0
    
    if smiles == new_smiles:
        s = "iter: %d sim: 0.00 ori_sim: 0.00 imp: 0.00 cannot decode\n" % (iternum)
        ori_sim = similarity(smiles, ori_smiles)
    else:
        ori_sim = similarity(new_smiles, ori_smiles)
        if reselect == 0:
            s = "iter: %d sim: %.2f ori_sim: %.4f imp: %.2f decode molecule %s\n" % (iternum, sim, ori_sim, score2-score1, new_smiles)
        elif reselect == 1:
            s = "iter: %d sim: %.2f ori_sim: %.4f imp: %.2f decode molecule %s reselect\n" % (iternum, sim, ori_sim, score2-score1, new_smiles)

    output.write(s)
    print(s)
    return score1, score2, atomnum, new_smiles, sim, ori_sim, reselect

def output_iter(res, res_file):
    if len(res) == 0:
        s = "num: 0 avg(imp): 0.00(0.00) avg(ori_imp): 0.00(0.00) avg(sim): 0.00(0.00) avg(ori_sim): 0.00(0.00) avg(atom):0.00 avg(newatom):0.00 \n"

    else:
        avg_imp = sum([x[0] for x in res])/len(res)
        std_imp = math.sqrt(sum([(x[0] - avg_imp)**2 for x in res])/len(res))
        avg_sim = sum([x[1] for x in res])/len(res)
        std_sim = math.sqrt(sum([(x[1] - avg_sim)**2 for x in res])/len(res))

        avg_ori_imp = sum([x[9] for x in res])/len(res)
        std_ori_imp = math.sqrt(sum([(x[9] - avg_ori_imp)**2 for x in res])/len(res))
        avg_ori_sim = sum([x[10] for x in res])/len(res)
        std_ori_sim = math.sqrt(sum([(x[10] - avg_ori_sim)**2 for x in res])/len(res))

        s = "num: %d avg(imp): %.2f(%.2f) avg(ori_imp): %.2f(%.2f) avg(sim): %.2f(%.2f) avg(ori_sim): %.2f(%.2f) avg(atom):%.2f avg(newatom):%.2f \n" % (len(res), avg_imp, std_imp, avg_ori_imp, std_ori_imp, avg_sim, std_sim, avg_ori_sim, std_ori_sim, sum([x[4] for x in res])/len(res),sum([x[6] for x in res])/len(res))

    res_file.write(s)
    res_file.flush()

def output_best(best_res, res_file):
    if len(best_res) == 0:
        s = "num: 0 avg(imp): 0.00(0.00) avg(sim): 0.00(0.00) avg(atom):0.00 avg(newatom):0.00 \n" 
    else:
        avg_imp = sum([x[0] for x in best_res])/len(best_res)
        std_imp = math.sqrt(sum([(x[0] - avg_imp)**2 for x in best_res])/len(best_res))
        avg_sim = sum([x[1] for x in best_res])/len(best_res)
        std_sim = math.sqrt(sum([(x[1] - avg_sim)**2 for x in best_res])/len(best_res))
       
        pdb.set_trace() 
        s = "num: %d avg(imp): %.2f(%.2f) avg(sim): %.2f(%.2f) avg(atom):%.2f avg(newatom):%.2f \n" % (len(best_res), avg_imp, std_imp, avg_sim, std_sim, sum([x[4] for x in best_res])/len(best_res), sum([x[6] for x in best_res])/len(best_res))
    res_file.write(s)
    res_file.flush()
    


    
parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test_path")
parser.add_argument("-v", "--vocab", dest="vocab_path")
parser.add_argument("-m", "--model", dest="model_path")
parser.add_argument("-d", "--save_dir", dest="save_dir")
parser.add_argument("-st", "--start", dest="start")
parser.add_argument("-si", "--size", dest="size")

parser.add_argument("-r", "--reselect", type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embed_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthG', type=int, default=5)
parser.add_argument('--depthT', type=int, default=3)
parser.add_argument('--iternum', type=int, default=1)
    
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
opts = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

sim_cutoff = float(opts.cutoff)
start = int(opts.start)
end = int(opts.size) + start

model = MolOpt(vocab, common_atom_vocab, opts)
model.load_state_dict(torch.load(opts.model_path, map_location=torch.device(device)))

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

# for each input molecule in data
for smiles in data[start:end]:
    best_score1, best_score2, best_imp, best_ori_imp, best_atom_num, best_sim, best_orisim = 0,0,None,0,0,0,0
    last_best_score2, last_best_ori_imp, last_best_orisim, last_best_atom_num = 0,0,0,0 
    best_smiles = smiles
    fail_num = 0
    reselect_num = 0
    ori_smiles = smiles
    ori_score = None
    ori_atom = 0
    # optimize the input molecule for no more than "iternum" times
    for i in range(opts.iternum):
        if best_imp is not None:
            if best_imp <= 0:
                # if the best improvement is not greater than 0,
                # stop optimization
                best_score2 = last_best_score2
                best_ori_imp = last_best_ori_imp
                best_orisim = last_best_orisim
                best_atom_num = last_best_atom_num
                break
            else:
                best_imp = None
        
        last_best_score2 = best_score2
        last_best_ori_imp = best_ori_imp
        last_best_orisim = best_orisim
        last_best_atom_num = best_atom_num
        
        outfiles[i].write(ori_smiles+"\n")
        smiles = best_smiles

        # in each iteration, randomly optimize the molecule for "num" times
        for _ in range(opts.num):
            score1, score2, atom_num, new_smiles, sim, ori_sim, reselect = predict(smiles, opts.lr, vocab, common_atom_vocab, opts.reselect, ori_smiles, i, outfiles[i])
            reselect_num += reselect
            if smiles == new_smiles:
                fail_num += 1
            if ori_score is None:
                ori_atom = atom_num
                ori_score = score1

            # save the best optimized molecule
            if (best_imp is None or score2-score1 > best_imp) and ori_sim > opts.cutoff:
                best_imp = score2-score1
                best_ori_imp = score2 - ori_score
                best_score2 = score2
                best_atom_num = Chem.MolFromSmiles(new_smiles).GetNumAtoms()
                best_smiles = new_smiles
                best_sim = sim
                best_orisim = ori_sim
        
        outfiles[i].flush()
        if best_imp is None: best_imp = 0
        if best_ori_imp is None: best_ori_imp = 0
        
        res[i].append( (best_imp, best_sim, smiles, score1, atom_num, best_smiles, best_atom_num, best_score2, ori_smiles, best_ori_imp, best_orisim) )
        s = "iter:%d num: %d sel: %d imp: %.4f ori_imp: %.4f sim: %.4f ori_sim: %.4f smile: %s score: %.4f atomnum: %d new_smile: %s new_atomnum: %d new_score: %.4f ori_smile: %s\n" % (i, fail_num, reselect_num, best_imp, best_ori_imp, best_sim, best_orisim, smiles, score1, atom_num, best_smiles, best_atom_num, best_score2, ori_smiles)
        print(s)
        res_file.write(s)
        res_file.flush()
    
    time2 = time.time()
    print("time: %s" % str(time2-time1))
    time1 = time2
    best_res.append( (best_ori_imp, best_orisim, ori_smiles, ori_score, ori_atom, best_smiles, best_atom_num, best_score2) )



# ========================================== Output iteration results =============================================

for i in range(opts.iternum):
    outfiles[i].close()
    res_file.write("iter%d:\n" % i)

    # Total number
    res_file.write("[total]\n")
    output_iter(res[i], res_file) 
    
    # Positive
    res_file.write("[positive]\n")
    tmp_res = [x for x in res[i] if x[0] > 0]
    output_iter(tmp_res, res_file)

    # Negative
    res_file.write("[negative]\n")
    tmp_res = [x for x in res[i] if x[0] < 0]
    output_iter(tmp_res, res_file)
    
    # Zero
    res_file.write("[zero]\n")
    tmp_res = [x for x in res[i] if x[0] == 0]
    output_iter(tmp_res, res_file)
     
# =========================================== Output the best results ============================================

# Total
res_file.write("[best total]\n")
output_best(best_res, res_file)

# Positive
res_file.write("[best positive]\n")
tmp_res = [x for x in best_res if x[0] > 0]
output_best(tmp_res, res_file)

# Negative
res_file.write("[best negative]\n")
tmp_res = [x for x in best_res if x[0] < 0]
output_best(tmp_res, res_file)

# Zero
res_file.write("[best zero]\n")
tmp_res = [x for x in best_res if x[0] == 0]
output_best(tmp_res, res_file)
res_file.close()

# Save the optimized molecule into output file
smiles_file = open("%s_iter%d_smiles.txt" % (output_path, opts.iternum), 'w')
for x in best_res:
    string = "%s %s %.2f %.4f %.4f %.4f\n" % (x[2], x[5], x[1], x[3], x[7], x[0])
    smiles_file.write(string)
smiles_file.close()
