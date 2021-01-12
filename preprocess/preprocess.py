#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:41:00 2019

@author: ziqi
"""
import sys, os

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(dir_path)
sys.path.insert(0, dir_path + "/model/")
from mol_tree import MolTree
from vocab import Vocab
import time
from transfer import compute_path
import multiprocessing as mp
from argparse import ArgumentParser
import pickle
import rdkit
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from sift import sift_molecules

VOCAB = None

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)

    for node in mol_tree.mol_tree.nodes:
        smiles = mol_tree.mol_tree.nodes[node]['label']
        mol_tree.mol_tree.nodes[node]['wid'] = VOCAB[smiles]

    return mol_tree

def tensorize_pair(smiles_pair):
    mol_tree0 = tensorize(smiles_pair[0], assm=False)
    mol_tree1 = tensorize(smiles_pair[1], assm=True)
    path = compute_path(mol_tree0, mol_tree1)
    return (mol_tree0, mol_tree1, path)
        
if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument('--train', type=str, default=dir_path+"/data/logp06/train_pairs.txt")
    parser.add_argument('--vocab', type=str, default=dir_path+"/data/logp06/vocab.txt")
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=mp.cpu_count())
    #parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    path = os.path.dirname(args.train)    
    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    VOCAB = Vocab(vocab)
    cpu_count = mp.cpu_count()
    print("number of cpu %d" % (cpu_count))
    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            alldata = [line.strip("\r\n ").split()[:2] for line in f]

        #split_id = args.split
        #start = split_id * 10000
        #last = min(start+10000, len(alldata))
        data = alldata #[start:last]
        
        pdata = []
        print("start")
        with ProcessPool(max_workers=cpu_count) as pool:
            future = pool.map(tensorize_pair, data, timeout=300)
        
        iterator = future.result()
        while True:
            try:
                pdata.append(next(iterator))
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
                
        pdata = sift_molecules(pdata)    
        with open(path+'/new_tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(pdata, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            alldata = [line.strip("\r\n ").split()[0] for line in f]
        
        split_id = args.split
        start = split_id * 20000
        last = min(start+20000, len(alldata))
        data = alldata[start:last]
    
        pdata = pool.map(tensorize, data)
        
        with open(path+'/new_tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(pdata, f, pickle.HIGHEST_PROTOCOL)
