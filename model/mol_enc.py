#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:28:46 2019

@author: ziqi
"""
import pdb
import time
import torch
import torch.nn as nn
from mol_tree import MolTree
from chemutils import BOND_LIST
from nnutils import index_select_ND, MPNN, GRU, create_pad_tensor
MAX_NB = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MolEncoder(nn.Module):
    def __init__(self, hidden_size, latent_size, atom_size, depthT, depthG, embedding):
        super(MolEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.atom_size = atom_size
        self.bond_size = len(BOND_LIST) 
        self.depthT = depthT
        self.depthG = depthG
        self.embedding = embedding  # Embedding for substructures
        self.embedding_size = embedding.weight.shape[1]
        self.E_a = torch.eye(atom_size).to(device)
        self.E_b = torch.eye(self.bond_size).to(device)
        
        #Atom-level Message Passing
        self.W_a = nn.Linear(atom_size + self.bond_size + hidden_size, hidden_size).to(device)
        self.outputAtom = nn.Sequential(
            nn.Linear(atom_size + depthG * hidden_size, hidden_size).to(device),
            nn.ReLU()
        )
        
        #Tree-level Message Passing
        
        self.W_i = nn.Linear(self.embedding_size + hidden_size, hidden_size, hidden_size).to(device)
        self.W_g = nn.Linear(2 * hidden_size, hidden_size).to(device)
        self.W_t = nn.Linear(2 * hidden_size, hidden_size).to(device)
        self.outputNode = nn.Sequential(
            nn.Linear((depthT+1) * hidden_size, hidden_size).to(device),
            nn.ReLU()
        )
    
    def embed_tree(self, tree_tensors, hatom):
        fnode, fmess, agraph, bgraph, cgraph, dgraph, _ = tree_tensors
        finput = self.embedding(fnode)
            
        hnode = index_select_ND(hatom, 0, dgraph).sum(dim=1)
        
        hnode = self.W_i( torch.cat([finput, hnode], dim=-1) )
        #pdb.set_trace()   
        hmess1 = hnode.index_select(index=fmess[:,0], dim=0)
        hmess2 = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        hmess = self.W_g( torch.cat([hmess1, hmess2], dim=-1) )
        
        return hnode, hmess, agraph, bgraph
        
    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _, _ = graph_tensors
        hnode = self.E_a.index_select(index=fnode, dim=0)
        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        
        return hnode, hmess, agraph, bgraph
        
    def mpn(self, hnode, hmess, agraph, bgraph, depth, W_m, W_n):
        messages = MPNN(hmess, bgraph, W_m, depth, self.hidden_size)
        mess_nei = index_select_ND(messages, 0, agraph)
        node_vecs = torch.cat((hnode, mess_nei.sum(dim=1)), dim=-1)
        node_vecs = W_n(node_vecs)
        return node_vecs, messages
        
    def forward(self, tree_tensors, graph_tensors, orders):
        tensors = self.embed_graph(graph_tensors)
        hatom, _ = self.mpn(*tensors, self.depthG, self.W_a, self.outputAtom)
        hatom[0,:] = hatom[0,:] * 0

        tensors = self.embed_tree(tree_tensors, hatom)
        hnode, _ = self.mpn(*tensors, self.depthT, self.W_t, self.outputNode)
        hnode[0,:] = hnode[0,:] * 0        
        
        revise_nodes = [[edge[1] for edge in order] for order in orders]
        revise_nodes = create_pad_tensor(revise_nodes).to(device).long()
        embedding = index_select_ND(hnode, 0, revise_nodes).sum(dim=1)
        
        return embedding, hnode, hatom
    
    def encode_atom(self, graph_tensors):
        tensors = self.embed_graph(graph_tensors)
        hatom, _ = self.mpn(*tensors, self.depthG, self.W_a, self.outputAtom)
        hatom[0,:] = hatom[0,:] * 0
        return hatom
        
    def encode_node(self, tree_tensors, hatom, node_idx):
        hnode, hmess, agraph, bgraph = self.embed_tree(tree_tensors, hatom)
        hnode = index_select_ND(hnode, 0, node_idx)
        agraph = index_select_ND(agraph, 0, node_idx)
        hnode, _ = self.mpn(hnode, hmess, agraph, bgraph, self.depthT, self.W_t, self.outputNode)
        return hnode
