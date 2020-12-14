#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:09:46 2019

@author: ziqi
"""
import numpy as np
import os
import gc
import pdb
import sys
import math
import time
import torch
import pickle
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from molopt import MolOpt
from vocab import Vocab, common_atom_vocab
from datautils import PairTreeFolder
from torch.nn import DataParallel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
path = "/fs/scratch/PCON0005/che268/graphoptimization"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=path+"/data/logp06/")
    parser.add_argument('--vocab', type=str, default=path+"/data/vocab.txt")
    parser.add_argument('--prop', type=str, default=path+"/data/train.logP-SA")
    parser.add_argument('--save_dir', type=str, default=path+"/result/")
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--ngpus', type=int, default=None)

    parser.add_argument('--score_func', type=int, default=1)    
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--depthG', type=int, default=5)
    parser.add_argument('--depthT', type=int, default=3)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=0.000)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=3000)
    
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=5000)
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)
    parser.add_argument('--print_iter', type=int, default=20)
    parser.add_argument('--save_iter', type=int, default=3000)
    
    args = parser.parse_args()
    gc.collect()
    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)
    avocab = common_atom_vocab
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    model = MolOpt(vocab, avocab, args)
    print(model)
    
    #if torch.cuda.is_available():
    #  print("Let's use", torch.cuda.device_count(), "GPUs!")
    #  model = nn.DataParallel(model)
    
    #model.to(device)
    if args.load_epoch > 0:
        model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch), map_location=torch.device(device)))

    print("Model #Params: {0}K".format(sum([x.nelement() for x in model.parameters()]) / 1000))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = args.load_epoch
    beta = args.beta if total_step == 0 or total_step < args.warmup else min(args.max_beta, args.beta+args.step_beta*(int((total_step-args.warmup)/args.kl_anneal_iter)+1)) 
    if model.beta != beta: model.beta=beta
    losses = np.zeros(9)
    acc_rec = np.zeros(8)
    nums = np.zeros(7)
    
    t1 = time.time()
    #t4 = time.time()
    #loader = PairTreeFolder(sublists, vocab, args.batch_size)  # Remove for loop here
    #data = []
    #for batch in loader:
    #    data.append(batch)
    
    loader = PairTreeFolder(args.train, vocab, avocab, 1)  # Remove for loop here
    number_same = 0
    
    for it, batch in enumerate(loader):
        t4 = time.time()
        new_smile, sim, score1, score2 = model.validate(*batch, 0.3)
        
        if sim == 1:
            number_same += 1
        print("sim: %.4f target: %s score: %.4f predict: %s score: %.4f input: %s" % (sim, batch[3][0].smiles, score1, new_smile, score2, batch[2][0].smiles))

    print("#same molecule: %d/%d/%.4f" % (number_same, it, number_same/it))
