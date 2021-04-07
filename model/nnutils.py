#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:02:52 2019

@author: ziqi
"""
import numpy as np
import torch
import torch.nn as nn
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def unique_tensor(tensor):
    tensor = (tensor.data).cpu().numpy()
    unique_tensor = []
    visited = [-1 for _ in range(tensor.shape[0])]
    for i in range(tensor.shape[0] - 1):
        if visited[i] != -1: continue
        for j in range(i+1, tensor.shape[0]):
            if visited[j] != -1: continue
            boolean = np.allclose(tensor[i,:], tensor[j,:], atol=2e-07)
            if boolean:
                if visited[i] == -1:
                    unique_tensor.append(tensor[i,:])
                    visited[i] = len(unique_tensor) - 1
                
                visited[j] = len(unique_tensor) - 1
    
    for i in range(tensor.shape[0]):
        if visited[i] != -1: continue
        unique_tensor.append(tensor[i,:])
        visited[i] = len(unique_tensor) - 1

    unique_tensor = torch.tensor(np.stack(unique_tensor, axis=0)).to(device)
    return unique_tensor, visited
    
    
def create_pad_tensor(alist, extra_len=0):
    max_len = max([len(a) for a in alist]) + 1 + extra_len
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)
    
def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def GRU(x, h, W_z, W_r, U_r, W_h):
    #hidden_size = x.size()[-1]
    dim = x.dim()-1
    
    z_input = torch.cat([x,h],dim=dim)
    z = torch.sigmoid(W_z(z_input))
    r_1 = W_r(x).squeeze()
    r_2 = U_r(h)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = torch.squeeze(r * h, dim)
    h_input = torch.cat([x,gated_h],dim=dim)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * h + z * pre_h
    
    return new_h

def MPL(x, cur_x_nei, cur_v_nei, W_g, U_g):
    new_v = torch.relu(W_g(torch.cat([cur_x_nei, cur_v_nei], dim=2)))
    v_nei = new_v.sum(dim=1)
    z = torch.relu(U_g(torch.cat([x, v_nei], dim=1)))
    
    return new_v, z
    
def MPNN(fmess, mess_graph, W_g, depth, hidden_size):
    multi_layer_mess = []
    messages = torch.zeros(mess_graph.size(0), hidden_size).to(device)
    for i in range(depth):
        nei_message = index_select_ND(messages, 0, mess_graph)
        nei_message = nei_message.sum(dim=1)
        messages = torch.relu(W_g(torch.cat([fmess, nei_message], dim=1)))
        multi_layer_mess.append(messages)
        messages[0,:] = messages[0,:] * 0
        
    
    messages = torch.cat(multi_layer_mess, dim=1)
    messages[0,:] = messages[0,:] * 0 
    return messages

def bfs(node_stack, insert_stack, stop_stack, num):
    """ Breadth first search
    """
    temp = []
    for node, _, _ in node_stack[-num:]:
        if node is None: continue
        if len(stop_stack) == 0: node.fa_node = None
        num = len(temp)
        for i, neighbor in enumerate(node.keep_neighbors):
            if neighbor == node.fa_node: continue
            if neighbor.fa_node != node:
                neighbor.fa_node = node
            if i < len(node.keep_neighbors)-1: 
                temp.append((neighbor, True, -1))
            else:
                temp.append((neighbor, False, len(stop_stack)))
        
        if len(temp) == num:
            temp.append((None, False, len(stop_stack)))
        #pdb.set_trace()
        
        for n, neighbor in enumerate(node.insert_neighbors):
            if neighbor == node.fa_node: continue
            if neighbor.fa_node != node:
                neighbor.fa_node = node
            
            stop_stack.append((node, False, n, len(insert_stack)))
            
            if neighbor in node.subtrees:
                insert_stack.append((node, neighbor, node.subtrees[neighbor]))
            else:
                insert_stack.append((node, neighbor, None))
            
            temp.append((neighbor, False, len(stop_stack)))
        
        stop_stack.append((node, True, len(node.insert_neighbors), len(insert_stack)))
    
    num = len(temp)
    if num > 0:
        node_stack.extend(temp)
        bfs(node_stack, insert_stack, stop_stack, num)
