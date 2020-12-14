#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:06:33 2019

@author: ziqi
"""
import random
import time
import copy
import torch
import torch.nn as nn
import sascorer
from rdkit.Chem import Descriptors
from mol_enc import MolEncoder
from mol_dec import MolDecoder
from mol_tree import MolTree
from chemutils import set_atommap, copy_edit_mol
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from properties import similarity, penalized_logp

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).to(device).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).to(device).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors
    
class MolOpt(nn.Module):
    """ model used to optimize molecule
    """
    def __init__(self, vocab, avocab, args):
        super(MolOpt, self).__init__()
        self.vocab = vocab
        self.beta = args.beta  # Weight for KL loss
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size  #Tree has one vectors
        self.atom_size = atom_size = avocab.size()
        #self.score_func = args.score_func
        
        # embedding for substructures and atoms
        self.embedding = nn.Embedding(vocab.size(), args.embed_size, padding_idx=0).to(device)
        self.E_a = torch.eye(atom_size).to(device)
        
        # encoder
        self.encoder = MolEncoder(self.hidden_size, self.latent_size, self.atom_size, args.depthT, args.depthG, self.embedding)
        
        # decoder
        self.decoder = MolDecoder(vocab, avocab, self.hidden_size, self.latent_size, args.depthT, self.embedding, self.encoder, score_func=args.score_func)
        
        # deletion embedding
        self.del_mean = nn.Linear(self.hidden_size, self.latent_size, bias=False).to(device)
        self.del_var  = nn.Linear(self.hidden_size, self.latent_size, bias=False).to(device)
        
        # addition embedding
        self.add_mean = nn.Linear(self.hidden_size, self.latent_size, bias=False).to(device)
        self.add_var  = nn.Linear(self.hidden_size, self.latent_size, bias=False).to(device)
        
    def encode(self, tensors, orders):
        """ Encode the molecule during the test
        
        Args:
            tensors: input embeddings
            orders:  
            
        Returns:
        """
        tree_tensors, graph_tensors = tensors
        
        tree_rev_vecs, tree_node_vecs, tree_atom_vecs = self.encoder(tree_tensors, graph_tensors, orders)

        return tree_rev_vecs, tree_node_vecs, tree_atom_vecs
    
    def fuse_noise(self, tree_vecs, mol_vecs):
        """ Add noise to the learned embeding during the testing
        """
        tree_eps = torch.randn(tree_vecs.size(0), self.latent_size).to(device)
        tree_eps = tree_eps.expand(1, tree_vecs.size(1))
        
        mol_eps = torch.randn(mol_vecs.size(0), self.latent_size).to(device)
        mol_eps = mol_eps.expand(1, mol_vecs.size(1))
        return tree_eps, mol_eps

    def forward(self, x_batch, y_batch, x_trees, y_trees, beta, total_step, add_target=False):
        """
        Args:
            x_batch: features of molecule X
            y_batch: features of molecule y
            x_trees: list of trees of molecules x
            y_trees: list of trees of molecules y
            beta   : weight for kl loss
        """
        # prepare feature embeddings
        x_graphs, x_tensors, x_orders, x_scores = x_batch
        y_graphs, y_tensors, y_orders, y_scores = y_batch
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_scores = x_scores.to(device)
        y_scores = y_scores.to(device)
        
        y_masks = (y_tensors[0][-3:-1], y_tensors[1][-3:-1])
        y_tensors = (y_tensors[0][:-3] + [y_tensors[0][-1]], y_tensors[1][:-3] + [y_tensors[1][-1]])
        
        # encoding
        x_del_vecs, x_node_vecs, x_atom_vecs = self.encode(x_tensors, x_orders)
        y_add_vecs, y_node_vecs, y_atom_vecs = self.encode(y_tensors, y_orders)
        
        # latent embedding
        z_del_vecs, del_kl_loss, del_mean, del_var = self.norm_sample(x_del_vecs, self.del_mean, self.del_var)
        z_add_vecs, add_kl_loss, add_mean, add_var = self.norm_sample(y_add_vecs, self.add_mean, self.add_var)
        
        # decoding
        x_scope = x_tensors[0][-1]
        y_scope = y_tensors[0][-1]
        loss, acc, rec, num = self.decoder(x_trees, y_trees, z_del_vecs, z_add_vecs, x_node_vecs, x_scope, x_graphs, y_graphs, y_tensors, y_orders, y_masks, total_step, add_target)
        
        # loss
        kl_loss = del_kl_loss + add_kl_loss
        tar_loss, del_loss, node_loss, topo_loss, atom1_loss, atom2_loss = loss
        loss = loss + (kl_loss,)
        
        if num[-1] > 0:
            total_loss = tar_loss + del_loss + node_loss + topo_loss + atom1_loss + 0.3 * atom2_loss + self.beta * kl_loss
        else:
            total_loss = tar_loss + del_loss + node_loss + topo_loss + atom1_loss + self.beta * kl_loss
        
        return total_loss, loss, acc, rec, num, ((del_mean.data).cpu().numpy(), (del_var.data).cpu().numpy(), (add_mean.data).cpu().numpy(), (add_var.data).cpu().numpy())
    
    def predict_prop(self, x_tree_node_vecs, y_tree_node_vecs, x_del_vecs, y_add_vecs, x_scores, y_scores, x_scope, y_scope):
        
        x_tree_vecs = []
        y_tree_vecs = []
        
        for scope in x_scope:
            x_tree_vecs.append(x_tree_node_vecs[scope[0]:scope[0]+scope[1],:].sum(dim=0))
            
        for scope in y_scope:
            y_tree_vecs.append(y_tree_node_vecs[scope[0]:scope[0]+scope[1],:].sum(dim=0))
        
        x_tree_vecs = torch.stack(x_tree_vecs, dim=0)
        y_tree_vecs = torch.stack(y_tree_vecs, dim=0)
        zx_del_vecs, _, _, _ = self.norm_sample(x_del_vecs, self.del_mean, self.del_var)
        zx_add_vecs, _, _, _ = self.norm_sample(y_add_vecs, self.add_mean, self.add_var)
        zx_tree_vecs = torch.cat((zx_del_vecs, zx_add_vecs), dim=1)
        
        zy_del_vecs, _, _, _ = self.norm_sample(y_add_vecs, self.del_mean, self.del_var)
        zy_add_vecs, _, _, _ = self.norm_sample(x_del_vecs, self.add_mean, self.add_var)        
        zy_tree_vecs = torch.cat((zy_del_vecs, zy_add_vecs), dim=1)
        
        xtoy_scores = y_scores - x_scores
        ytox_scores = x_scores - y_scores
        scores = torch.cat((xtoy_scores, ytox_scores), dim=0)
        
        xtoy_pred = self.propNN(torch.cat((x_tree_vecs, zx_tree_vecs), dim=1)).squeeze()
        ytox_pred = self.propNN(torch.cat((y_tree_vecs, zy_tree_vecs), dim=1)).squeeze()
        pred = torch.cat((xtoy_pred, ytox_pred), dim=0)
        
        prop_loss = self.prop_loss(pred, scores)
        return prop_loss
        
    def norm_sample(self, diff_vecs, mean, var):
        """ Calculate the difference between x_vecs and y_vecs
        
        Args:
            diff_vecs:
            mean:
            var:
        Returns:
            z_vecs:
            kl_loss:
            z_mean:
        """
        batch_size = diff_vecs.size(0)
        z_mean = mean(diff_vecs)
        z_log_var = -torch.abs(var(diff_vecs))
        
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size 
        epsilon = torch.randn_like(z_mean).to(device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        
        return z_vecs, kl_loss, z_mean, z_log_var
    
    def test(self, x_batch, x_tree, lr=1.0, num_iter=20, reselect_num=1):
        x_graphs, x_tensors, x_orders, x_scores = x_batch
        x_tensors = make_cuda(x_tensors)
        score1 = x_scores[0]
        
        _, x_tree_node_vecs, x_tree_atom_vecs = self.encode(x_tensors, x_orders)
        
        x_tree_vecs = x_tree_node_vecs.sum(dim=0)
        
        latent_vec = torch.autograd.Variable(torch.randn((2 * self.latent_size)), requires_grad=True).to(device)
        
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(x_tree.mol))
        fp1 = AllChem.GetMorganFingerprint(mol, 2)
        
        new_mol, reselect = self.decoder.decode(x_tensors, latent_vec, x_tree_node_vecs, x_graphs, mol, reselect_num)
        
        if new_mol is None:
            return x_tree.smiles, 1.0, 0, score1, score1
        set_atommap(new_mol)
        try:
            new_smiles = Chem.MolToSmiles(new_mol)
            score2 = penalized_logp(new_smiles)
            sim = similarity(x_tree.smiles, new_smiles)
        except Exception as e:
            print(e)
            return x_tree.smiles, 1.0, reselect, score1, score1
        if score1 == score2 and sim < 1:
            print("special case: %s and %s" % (x_tree.smiles, new_smiles))
        return new_smiles, sim, reselect, score1, score2

