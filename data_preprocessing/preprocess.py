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
import time
from transfer import compute_path
import multiprocessing as mp
from argparse import ArgumentParser
import pickle
import rdkit
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from sift import sift_molecules

def tensorize_pair(smiles_pair):
    mol_tree0 = MolTree(smiles_pair[0])
    mol_tree1 = MolTree(smiles_pair[1])
    path = compute_path(mol_tree0, mol_tree1)
    return (mol_tree0, mol_tree1, path)
        
if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument('--train', type=str, default=dir_path+"/data/logp06/train_pairs.txt", help="specify the name of file with training data to be processed")
    parser.add_argument('--output', type=str, default="tensors", help="specify the name of processed dataset.")
    parser.add_argument('--time_out', type=int, default=300, help="specify the timeout value for processing of each pair of molecules.")
    parser.add_argument('--ncpu', type=int, default=mp.cpu_count(), help="specify the number of CPUs used for preprocessing.")
    parser.add_argument('--batch_size', type=int, default=0, help="specify the number of molecule pairs to be processed within each batch." + \
                                                              "If batch_size = 0, the entire training dataset will be processed; otherwise, only a batch of dataset" + \
                                                              "will be processed. This option is recommended to be used for speeding up the preprocessing of large dataset.")
    parser.add_argument('--batch_id', type=int, default=0, help="specify the index of batch to be processed.")
    args = parser.parse_args()

    path = os.path.dirname(args.train)
    cpu_count = mp.cpu_count()
    print("number of cpu %d" % (cpu_count))
    
    # the dataset to be processed should contain molecule pairs
    with open(args.train) as f:
        alldata = [line.strip("\r\n ").split()[:2] for line in f][:20]
    
    # whether processing the entire dataset or only a batch of the dataset.
    if args.batch_size > 0:
        start = args.batch_id * args.batch_size
        last = min(start+args.batch_size, len(alldata))

        try:
            data = alldata[start:last]
        except:
            raise ValueError("Incorrect batch_size %d and batch_id %d" % (args.batch_size, args.batch_id))
    else: data = alldata
    
    if len(data) == 0:
        print("No data to be processed for batch_size %d and batch_id %d" % (args.batch_size, args.batch_id))
        sys.exit(0)
    
    pdata = []
    print("start")
    with ProcessPool(max_workers=cpu_count) as pool:
        # The function will stop the processing of a molecule pair, if its processing time is 
        # longer than args.time_out seconds.
        future = pool.map(tensorize_pair, data, timeout=args.time_out)
    
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
            print(error.traceback)
    
    # select pairs which differ in terms of only one fragment.
    pdata = sift_molecules(pdata)    
    with open("%s/%s-%d.pkl" % (path, args.output, args.batch_id), 'wb') as f:
        pickle.dump(pdata, f, pickle.HIGHEST_PROTOCOL)
