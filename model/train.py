#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:09:46 2019

@author: ziqi
"""
import numpy as np
import os
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
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=path+"/data/logp06/", help='data path to training data')
    parser.add_argument('--vocab', type=str, default=path+"/data/logp06/vocab.txt", help='data path to substructure vocabulary')
    parser.add_argument('--save_dir', type=str, default=path+"/result/", help='data path to the directory used to save trained models')
    parser.add_argument('--load_epoch', type=int, default=0, help='an interger used to control the loaded model (i.e., if load_epoch==1000, '+\
                        'the model save_dir+1000.pkl would be loaded)')
    parser.add_argument('--ngpus', type=int, default=None, help='the number of gpus')
    
    # size of model
    parser.add_argument('--hidden_size', type=int, default=32, help='the dimension of hidden layers')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of molecule pairs in each batch')
    parser.add_argument('--latent_size', type=int, default=32, help='the dimention of latent embeddings')
    parser.add_argument('--embed_size', type=int, default=32, help='the dimention of substructure embedding')
    parser.add_argument('--depthG', type=int, default=5, help='the depth of message passing in graph encoder')
    parser.add_argument('--depthT', type=int, default=3, help='the depth of message passing in tree encoder')
    
    
    parser.add_argument('--add_ds', action='store_true', help='a boolean used to control whether adding the embedding of disconnection site '+\
                        'into the latent embedding or not.')
    parser.add_argument('--clip_norm', type=float, default=50.0, help='')
    parser.add_argument('--beta', type=float, default=0.000, help='a float used to control the weight of kl loss')
    parser.add_argument('--beta_anneal_iter', type=int, default=500, help='')
    parser.add_argument('--step_beta', type=float, default=0.005, help='a float used to adjust the value of beta')
    parser.add_argument('--max_beta', type=float, default=0.1, help='the maximum value of beta')
    parser.add_argument('--warmup', type=int, default=3000, help='an interger used to control the number of steps keeping the inital beta value')
    parser.add_argument('--score_func', type=int, default=1, help='type of scoring function in the decoder')    

    # control the learning process
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--anneal_rate', type=float, default=0.9, help='')
    parser.add_argument('--anneal_iter', type=int, default=5000)
    parser.add_argument('--print_iter', type=int, default=20)
    parser.add_argument('--save_iter', type=int, default=3000)
    
    args = parser.parse_args()

    # read vocabulary
    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)
    avocab = common_atom_vocab

    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    model = MolOpt(vocab, avocab, args)
    print(model)
    
    # load previous trained model
    if args.load_epoch > 0:
        model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

    print("Model #Params: {0}K".format(sum([x.nelement() for x in model.parameters()]) / 1000))
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = args.load_epoch
    beta = args.beta if total_step == 0 or total_step < args.warmup else min(args.max_beta, args.beta+args.step_beta*(int((total_step-args.warmup)/args.kl_anneal_iter)+1)) 
    if model.beta != beta: model.beta=beta
    
    losses = np.zeros(8)
    acc_rec = np.zeros(8)
    nums = np.zeros(7)
    
    t1 = time.time()
    for epoch in range(args.epoch):
        kls = []
        
        loader = PairTreeFolder(args.train, vocab, avocab, args.batch_size, add_target=args.add_ds)
        for it, batch in enumerate(loader):
            
            with torch.autograd.set_detect_anomaly(True):
                total_step += 1
                model.zero_grad()
                
                total_loss, loss, acc, rec, num, _ = model(*batch, beta, total_step, add_target=args.add_ds)
                
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
            
            # cumulative loss
            losses = losses + np.array([float(total_loss)]+[float(l) for l in loss[:7]])
            nums += np.array(num)
            acc_rec = acc_rec + np.array(acc+rec)
            
            # print loss and accuracy
            if total_step % args.print_iter == 0:
                t2 = time.time()
                losses /= args.print_iter
                acc_rec /= args.print_iter
                nums /= args.print_iter
                
                s = "[%d/%d/%d] timecost: %.2f, Beta: %.3f, Loss: %.3f, KL: %.2f, " % (total_step, epoch, it, t2-t1, beta, losses[0], losses[7])
                s = s + "target: (%.4f, %.4f), del(%.2f, %.2f): (%.4f, %.4f, %.4f), " % (losses[1], acc_rec[0], nums[0], nums[1], losses[2], acc_rec[1], acc_rec[-2])
                s = s + "node(%.2f): (%.4f, %.4f), topo(%.2f, %.2f): (%.4f, %.4f, %.4f), " % (nums[2], losses[3], acc_rec[2], nums[3], nums[4], losses[4], acc_rec[3], acc_rec[-1])
                s = s + "atom1(%.2f): (%.4f, %.4f), atom2(%.2f): (%.4f, %.4f), " % (nums[5], losses[5], acc_rec[4], nums[6], losses[6], acc_rec[5])
                s = s + "PNorm: %.2f, GNorm: %.2f" % (param_norm(model), grad_norm(model))
                print(s)
                t1 = t2
                sys.stdout.flush()
                losses *= 0
                nums *= 0
                acc_rec *= 0
            
            # save model
            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))
    
            # update learning rate
            if total_step % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
    
            # update beta values
            if total_step % args.beta_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)
                model.beta = beta
