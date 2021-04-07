#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:28:46 2019

@author: ziqi
"""
import time
import torch
import copy
import networkx as nx
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import recall_score, accuracy_score
from nnutils import index_select_ND, GRU, MPL, bfs, unique_tensor
from mol_tree import MolTree
import molopt
from chemutils import get_mol, get_uniq_atoms, get_smiles, graph_to_mol, mol_to_graph, attach_mol_graph, bond_equal
import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"
        
class MolDecoder(nn.Module):
    def __init__(self, vocab, avocab, hidden_size, latent_size, depthT, embedding, encoder, score_func=1, tree_lstm=False):
        super(MolDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthT = depthT
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.avocab = avocab
        
        self.atom_size = avocab.size()
        self.embedding = embedding
        self.embedding_size = embedding.weight.shape[1]
        self.score_func = score_func
        self.encoder = encoder
        
        self.uniq_atom_dict = {}
        
        if score_func == 1:
            # Parameters for function used to predict parent attachments.
            self.W_a1 = nn.Linear(latent_size + self.embedding_size + 3 * hidden_size, hidden_size).to(device)
            self.U_a1 = nn.Linear(hidden_size, 1).to(device)
        
            # Parameters for function used to predict child attachments
            self.W_a2 = nn.Linear(latent_size + self.embedding_size + 2 * hidden_size, hidden_size).to(device)
            self.U_a2 = nn.Linear(hidden_size, 1).to(device)
        
            # Parameters for function used to predict disconnection site
            self.W_a = nn.Linear(2 * latent_size + hidden_size, hidden_size).to(device)

            self.U_a = nn.Linear(hidden_size, 1).to(device)
        elif score_func == 2:
            # Parameters for function used to predict parent attachments.
            self.W_a1 = nn.Linear(latent_size + self.embedding_size + 2 * hidden_size, hidden_size).to(device)
            self.U_a1 = nn.Linear(hidden_size, hidden_size).to(device)
            
            # Parameters for function used to predict child attachments.
            self.W_a2 = nn.Linear(latent_size + self.embedding_size + 1 * hidden_size, hidden_size).to(device)
            self.U_a2 = nn.Linear(hidden_size, hidden_size).to(device)

            # Parameters for function used to predict disconnection site
            self.W_a = nn.Linear(2 * latent_size_size, hidden_size).to(device)
            self.U_a = nn.Linear(hidden_size, hidden_size).to(device)

        else:
            raise ValueError("Wrong Value for Score Function (should be 1 or 2)")        
        
        # Parameters for function used to predict removal fragments
        self.W_d = nn.Linear(latent_size + hidden_size, hidden_size).to(device)
        self.U_d = nn.Linear(hidden_size, 1).to(device)

        # Parameters for function used to predict child node connections
        self.W_t = nn.Linear(latent_size + hidden_size, hidden_size).to(device)
        self.U_t = nn.Linear(hidden_size, 1).to(device)
        
        # Parameters for function used to predict child node types
        self.W_n = nn.Linear(latent_size + hidden_size, hidden_size).to(device)
        self.U_n = nn.Linear(hidden_size, self.vocab_size).to(device)
        
        #Loss Functions
        self.tart_loss = nn.CrossEntropyLoss(size_average=False)
        self.del_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.node_loss = nn.CrossEntropyLoss(size_average=False)
        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        
    def get_target_predictions(self, graphs):
        """ Get the ground truth disconnection site labels for prediction
        """
        labels = []
        for graph in graphs:
            for idx in graph.nodes:
                if graph.nodes[idx]['target']:
                    labels.append(idx)
        return labels
    
    def apply_tree_mask(self, tensors, masks):
        """ Mask the fragments to be added in the tree of molecule y
        so that the model can learn the fragments to be added with teacher forcing.

        Args:
            tensors: embeddings to be masked
            masks: node masks and edge masks
        """
        nmask, emask = masks
        fnode, fmess, agraph, bgraph, cgraph, dgraph, scope = tensors
        
        # substructure matrix mask
        agraph = (agraph * index_select_ND(emask, 0, agraph).squeeze(-1)).long()
        # edge matrix mask
        bgraph = (bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1)).long()
        
        tensors = (fnode, fmess, agraph, bgraph, cgraph, dgraph, scope)
        return tensors

    def apply_graph_mask(self, tensors, masks):
        """ Mask the fragments to be added in the graph of molecules y
        ...
        """
        amask, bmask = masks
        fnode, fmess, agraph, bgraph, cgraph, scope = tensors
        
        # atom matrix mask
        agraph = (agraph * index_select_ND(bmask, 0, agraph).squeeze(-1)).long()
        # bond matrix mask
        bgraph = (bgraph * index_select_ND(bmask, 0, bgraph).squeeze(-1)).long()
        
        tensors = (fnode, fmess, agraph, bgraph, cgraph, scope)
        return tensors
    
    def update_graph_mask(self, graph_batch, tree_tensors, graph_tensors, masks, node_idx):
        """ Update the graph mask after the prediction of node with idx `node_idx`,
        so that messages can be passed through the added new atoms in the new node.
        """
        amask, bmask = masks
        dgraph = tree_tensors[5]
        _, _, agraph, bgraph, egraph, _ = graph_tensors
        cls = index_select_ND(dgraph, 0, node_idx)
        
        # Get all the atoms within the node `node_idx`
        old_amask = copy.deepcopy(amask)
        amask.scatter_(0, cls[cls>0].unsqueeze(1), 1)
        
        # get the new edge mask from the atom mask 
        emask = amask * amask.transpose(0, 1)
        emask_idx = (emask * egraph).nonzero()
        emask_idx = egraph[emask_idx[:,0], emask_idx[:,1]].unsqueeze(1)
        emask = torch.zeros(bgraph.size(0), 1).to(device)
        emask.scatter_(0, emask_idx, 1)

        
        agraph = (agraph * index_select_ND(emask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1)).long()
        
        graph_tensors = graph_tensors[:2]+[agraph, bgraph]+graph_tensors[-2:]
        
        masks = (amask, bmask)
        return graph_tensors, masks
            
    def update_tree_mask(self, tree_batchG, tree_tensors, masks, node_idx):
        """ Update the tree mask after the prediction of node with idx `node_idx`,
        so that messages can be passed through the added new node.
        """
        nmask, emask = masks
        _, _, agraph, bgraph, cgraph, dgraph, _ = tree_tensors
        
        nmask.scatter_(0, node_idx.unsqueeze(1), 1)
        
        # Get the indices of messages/edges connected with node
        mess_idxs = []
        pairs = []
        for i in range(node_idx.size(0)):
            idx = node_idx[i].item()
            nei_idxs = [edge[1] for edge in tree_batchG.edges(idx) if nmask[edge[1]] == 1]
            pairs.extend([(idx, nei_idx) for nei_idx in nei_idxs])
        
        for pair in pairs:
            mess_idxs.append(tree_batchG[pair[0]][pair[1]]['mess_idx'])
            mess_idxs.append(tree_batchG[pair[1]][pair[0]]['mess_idx'])
        
        mess_idxs = torch.LongTensor(mess_idxs).to(device).unsqueeze(1)
        emask.scatter_(0, mess_idxs, 1)
        
        agraph = (agraph * index_select_ND(emask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1)).long()
        
        tree_tensors = tree_tensors[:2]+[agraph, bgraph, cgraph, dgraph, tree_tensors[-1]]
        masks = (nmask, emask)
        return tree_tensors, masks
    
    def get_node_embedding(self, tree_tensors, node_idx, masks, hatom):
        """ Get the embeddings of nodes using the message passing networks in encoder
        Args:
            tree_tensors: the tree embebdings used for TMPN
            node_idx: the index of nodes with embeddings to be learned 
        
        """
        nmask, emask = masks
        
        fnode, fmess, agraph, bgraph, cgraph, dgraph, scope = tree_tensors
        nei_mess = index_select_ND(agraph, 0, node_idx)
        nei_mess = nei_mess[nei_mess > 0]
        new_emask = torch.zeros(emask.shape).to(device)
        for depth in range(self.depthT):
            new_nei_mess = index_select_ND(bgraph, 0, nei_mess)
            new_nei_mess = new_nei_mess[new_nei_mess > 0]
            nei_mess = torch.unique(torch.cat([nei_mess, new_nei_mess], dim=0))
        
        new_emask.scatter_(0, nei_mess.unsqueeze(1), 1)
        new_emask = torch.mul(new_emask, emask)
        
        agraph = (agraph * index_select_ND(new_emask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(new_emask, 0, bgraph).squeeze(-1)).long()
        
        fmess = (fmess * index_select_ND(new_emask, 0, fmess).squeeze(-1)).long()
        tensors = (fnode, fmess, agraph, bgraph, cgraph, dgraph, None)
        
        # tree message passing
        hnode = self.encoder.encode_node(tensors, hatom, node_idx)
        
        return hnode
    
    def get_atom_embedding(self, tree_batch, graph_tensors, node_idx):
        """ Get the embeddings of atoms using the MPN in encoder
        """
        fatom, fbond, agraph, bgraph, egraph, scope = graph_tensors
        amask = torch.zeros(fatom.size(0), 1).to(device)
        
        clusters = []
        for i in range(node_idx.size(0)):
            node = node_idx[i].item()
            cluster = tree_batch.nodes[node]['clq']
            if len(cluster) > 2:
                clusters.extend(cluster)
        
        clusters = torch.LongTensor(clusters).unsqueeze(1).to(device)
        amask.scatter_(0, clusters, 1)
        
        emask = amask * amask.transpose(0, 1)
        emask_idx = (emask * egraph).nonzero()
        emask_idx = egraph[emask_idx[:,0], emask_idx[:,1]].unsqueeze(1)
        
        emask = torch.zeros(fbond.size(0), 1).to(device)
        emask.scatter_(0, emask_idx, 1)
        agraph = agraph * index_select_ND(emask, 0, agraph).squeeze(-1).long()
        bgraph = bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1).long()
        
        tensors = (fatom, fbond, agraph, bgraph, None, None)
        hatom = self.encoder.encode_atom(tensors)
        
        return hatom
    
            
    def forward(self, tree1_batch, tree2_batch, z_del_vecs, z_add_vecs, x_node_vecs, x_scope, x_graphs, y_graphs, tensors, orders, masks, total_step, add_target):
        batch_size = len(orders)
        x_tree_batch, x_graph_batch = x_graphs
        y_tree_batch, y_graph_batch = y_graphs
        tree_tensors, graph_tensors = tensors
        tree_mask, graph_mask = masks
        
        target_loss = []
        tart_acc = 0
        del_hiddens, del_targets = [], []
        for i in range(len(orders)):
            for idx in range(x_scope[i][0], x_scope[i][0] + x_scope[i][1]):
                if x_tree_batch.nodes[idx]['target']:
                    label = torch.LongTensor([idx-x_scope[i][0]]).to(device)
                    break
            
            #Disconnection site prediction
            scope = x_scope[i]
            target_diff_vecs = torch.cat((z_del_vecs[i,:], z_add_vecs[i,:]), dim=0).repeat((scope[1], 1))
            node_vecs = x_node_vecs[scope[0]:scope[0]+scope[1]]
            scores = self.scoring(node_vecs, target_diff_vecs, "target")
             
            if torch.argmax(scores) == label[0]:
                tart_acc += 1
                
            #
            target_loss.append( self.tart_loss(scores.reshape(1, -1), label) )
                
            # Deletion nodes embedding and targets preparation
            del_cands = [edge[1] for edge in x_tree_batch.edges(x_scope[i][0]+label.item())]
            if len(del_cands) == 1:
                continue
            del_diff_vecs = z_del_vecs[i,:].repeat( (len(del_cands), 1) )
            
            for del_idx in del_cands:
                del_targets.append(x_tree_batch.nodes[del_idx]['revise'])
                
            del_cands = torch.LongTensor(del_cands).to(device)
            del_node_vecs = index_select_ND(x_node_vecs, 0, del_cands)
            del_hidden = torch.cat((del_node_vecs, del_diff_vecs), dim=1)
            del_hiddens.append(del_hidden)
        
        # predictions of removal fragments
        del_targets = torch.FloatTensor(del_targets).to(device)
        del_one = del_targets.nonzero().size(0)
        del_zero = del_targets.size(0) - del_one
        del_hiddens = torch.cat(del_hiddens, dim=0)
        del_scores = self.predict(del_hiddens, "delete").squeeze(1)
        del_loss = self.del_loss(del_scores, del_targets) / len(orders)
        
        # Accuracy of removal fragments predictions
        dele = torch.ge(del_scores, 0).float()
        del_acc = torch.eq(dele, del_targets).float()
        del_acc = torch.sum(del_acc) / del_targets.nelement()
        del_rec = recall_score(del_targets.data.to('cpu'), dele.data.to('cpu'))
        
        target_loss = sum(target_loss) / len(orders)
        tart_acc = tart_acc / len(orders)
        
        # Prediction of New Fragment attachment
        # =====================================================================

        # Get the masked tree tensors and graph tensors
        cur_tree_tensors = self.apply_tree_mask(tree_tensors, tree_mask)
        cur_graph_tensors = self.apply_graph_mask(graph_tensors, graph_mask)
        
        maxt = max( [len(order) for order in orders] )
        off_set1, off_set2 = 1, 1
        
        max_cls_size = 2 * max( [len(y_tree_batch.nodes[node]['clq']) for node in y_tree_batch.nodes] ) + 6
        
        topo_hiddens, topo_targets = [],[]
        atom1_hiddens1, atom1_hiddens2, atom1_targets1, atom1_targets2, atom1_labels = [],[],[],[],[]
        node_hiddens, node_targets = [],[]
        atom2_hiddens1, atom2_hiddens2, atom2_targets1, atom2_targets2, atom2_labels = [],[],[],[],[]

        # index used to represent whether the embedding of disconnection site are considered when calculating 
        # the latent training data; `begin=1` represents that h = nd + sum(nx).
        if add_target:
            begin = 1
        else:
            begin = 0
        
        for t in range(begin, maxt):
            batch_list = [i for i in range(batch_size) if t < len(orders[i])]
            pred_list, nodex_idxs = [],[]
            stop_idxs, stop_list, nodey_idxs, node_target = [],[],[],[]

            # Get the ground truth labels for child connection / child node type predictions.
            for i in batch_list:
                xid, yid, ylabel = orders[i][t]
                if len(orders[i]) > t+1:
                    next_xid, _, _ = orders[i][t+1]
                    next_xid = next_xid == xid
                else:
                    next_xid = False
                
                if not next_xid:
                    stop_idxs.append(xid)
                    stop_list.append(i)
                pred_list.append((xid, yid, i))
                nodex_idxs.append(xid)
                nodey_idxs.append(yid)
                node_target.append(ylabel)
            
            nodex_idxs = torch.LongTensor(nodex_idxs).to(device)
            nodey_idxs = torch.LongTensor(nodey_idxs).to(device)
            node_target = torch.LongTensor(node_target).to(device)
            stop_idxs = torch.LongTensor(stop_idxs).to(device)
            stop_list = torch.LongTensor(stop_list).to(device)
            batch_list = torch.LongTensor(batch_list).to(device)
            
            # Get the Atom Embeddings learned from MPN in encoder and update graphs
            hatom1 = self.encoder.encode_atom(cur_graph_tensors)
            hatom2 = self.get_atom_embedding(y_tree_batch, graph_tensors, nodey_idxs)
            cur_graph_tensors, graph_mask = self.update_graph_mask(y_graph_batch, tree_tensors, graph_tensors, graph_mask, nodey_idxs)
            hatom3 = self.encoder.encode_atom(cur_graph_tensors)
            
            # Get the Node embeddings learned from MPN in encoder and update trees
            hnode1 = self.get_node_embedding(cur_tree_tensors, nodex_idxs, tree_mask, hatom1)
            tmp_add_vecs = index_select_ND(z_add_vecs, 0, batch_list)
            cur_tree_tensors, tree_mask = self.update_tree_mask(y_tree_batch, tree_tensors, tree_mask, nodey_idxs)
            hnode2 = self.get_node_embedding(cur_tree_tensors, stop_idxs, tree_mask, hatom3)
            
            # Get the embeddings and ground truth labels for node type predictions
            node_hidden = torch.cat((hnode1, tmp_add_vecs), dim=1)
            node_targets.append(node_target)
            node_hiddens.append(node_hidden)
            
            # Get the embeddings and ground truth labels for child connections prediction
            tmp_add_vecs = index_select_ND(z_add_vecs, 0, batch_list)
            topo_hidden1 = node_hidden.clone()
            topo_target1 = [1] * len(batch_list)
            tmp_add_vecs = index_select_ND(z_add_vecs, 0, stop_list)
            topo_hidden2 = torch.cat((hnode2, tmp_add_vecs), dim=1)
            topo_target2 = [0] * len(stop_idxs)
            
            topo_hidden = torch.cat((topo_hidden1, topo_hidden2), dim=0)
            topo_hiddens.append(topo_hidden)
            topo_target = topo_target1 + topo_target2
            topo_targets.append(torch.FloatTensor(topo_target).to(device))
            
            
            # Attachment Prediction to predict the attachment between the 
            # parent node and the child node (the addded new node)
            # ===============================================================
            
            # Attachemnt Prediction embeddings and ground truth labels
            clusters1, clusters2 = [],[]
            atom1_hidden2, atom2_hidden2 = [],[]
            atom1_target1, atom1_target2 = [],[]
            atom2_target1, atom2_target2 = [],[]
            atom1_label, atom2_label = [],[]
            fnode = cur_tree_tensors[0]

            
            for j, pred in enumerate(pred_list):
                xid, yid, k = pred
                
                # If the parent node is a ring (atoms > 2), we need to predict 
                # the parent attachment atoms in the parent node.
                if len(y_tree_batch.nodes[xid]['clq']) > 2:
                    # If the child node is also a ring, then the attachment
                    # between these two ring nodes can be a bond; otherwise, 
                    # the attachment contains only one single atom.
                    if len(y_tree_batch.nodes[yid]['clq']) > 2:
                        target_idx1 = y_tree_batch[xid][yid]['label']
                        
                        # atom feature
                        cluster1 = y_tree_batch.nodes[xid]['clq']
                        pad = torch.zeros((len(cluster1))).to(device).long()
                        cluster1 = torch.LongTensor(cluster1).to(device)

                        # bond feature
                        cluster2 = torch.LongTensor(y_tree_batch.nodes[xid]['bonds']).to(device).long()
                        
                        # ground truth label for prediction
                        if len(y_tree_batch[xid][yid]['anchor']) > 1:
                            target_idx1 = target_idx1 + cluster1.size(0)
                        
                        pad_cluster1 = torch.stack([cluster1, pad], dim=1)
                        cluster = torch.cat((pad_cluster1, cluster2), dim=0)
                    else:
                        # ground truth label
                        target_idx1 = y_tree_batch[xid][yid]['label']
                        
                        # atom feature
                        cluster1 = torch.LongTensor(y_tree_batch.nodes[xid]['clq']).to(device)
                        pad = torch.zeros(cluster1.size(0)).to(device).long()

                        cluster = torch.stack([cluster1, pad], dim=1)
                    
                    # Get the embeddings and labels for the attachment point prediction at parent node
                    atom1_hidden1 = index_select_ND(hatom1, 0, cluster).sum(dim=1)
                    uniq_atom1_hidden1, inverse_idx = unique_tensor(atom1_hidden1)
                    clusters1.append(uniq_atom1_hidden1)
                    target_idx1 = inverse_idx[target_idx1]
                    cands_size = uniq_atom1_hidden1.size(0)
                    target_idx2 = torch.LongTensor([i+off_set1 for i in range(cands_size)] + [0] * (max_cls_size - cands_size)).to(device)
                    atom1_label.append(target_idx1)
                    atom1_target1.append(target_idx1+off_set1)
                    atom1_target2.append(target_idx2)
                    
                    nodex = index_select_ND(hatom1, 0, cluster1).sum(dim=0)
                    nodex = self.encoder.W_i(torch.cat((self.embedding(fnode[xid]), nodex), dim=0))
                    x_atom_hidden = torch.cat((self.embedding(fnode[yid]), nodex, hnode1[j,:], z_add_vecs[k,:]), dim=0).unsqueeze(0)
                    x_atom_hidden = x_atom_hidden.expand(cands_size, x_atom_hidden.size(1))
                    atom1_hidden2.append(x_atom_hidden)
                    off_set1 = off_set1 + cands_size
                
                # ********************************************************
                
                # If the child node is a ring, we need to predict the attachment point
                # at the child node.
                if len(y_tree_batch.nodes[yid]['clq']) > 2:
                    # the attachment can be a bond if parent node is also a ring node.
                    # Get the embeddings and labels for attachment point prediction
                    if len(y_tree_batch.nodes[xid]['clq']) > 2:
                        is_ring = 2

                        # atom feature
                        cluster1 = y_tree_batch.nodes[yid]['clq']
                        pad = torch.zeros((len(cluster1))).to(device).long()
                        cluster1 = torch.LongTensor(cluster1).to(device)
                        
                        # bond feature
                        cluster2 = torch.LongTensor(y_tree_batch.nodes[yid]['bonds']).to(device)
                        
                        # ground truth label
                        target_idx1 = y_tree_batch[yid][xid]['label']
                        if len(y_tree_batch[yid][xid]['anchor']) > 1:
                            target_idx1 = target_idx1 + cluster1.size(0)
                        
                        cluster1 = torch.stack([cluster1, pad], dim=1)
                        cluster = torch.cat((cluster1, cluster2), dim=0)
                        
                    else:
                        is_ring = 1
                        target_idx1 = y_tree_batch[yid][xid]['label']
                        cluster1 = torch.LongTensor(y_tree_batch.nodes[yid]['clq']).to(device)
                        cluster2 = torch.zeros(cluster1.size(0)).to(device).long()
                        cluster = torch.stack([cluster1, cluster2], dim=1)
                    
                    attach_atoms = y_tree_batch[xid][yid]['anchor']
                    atom2_hidden1 = index_select_ND(hatom2, 0, cluster).sum(dim=1)
                    
                    # keep only unique candidate attachment point
                    uniq_atom2_hidden1, inverse_idx = unique_tensor(atom2_hidden1)
                    cands_size = uniq_atom2_hidden1.size(0)
                    
                    # If the number of candidates is greater than 1, then prediction is required.
                    if cands_size > 1:
                        clusters2.append(uniq_atom2_hidden1)
                        
                        target_idx1 = inverse_idx[target_idx1]
                        atom2_label.append(target_idx1)
                        target_idx2 = torch.LongTensor([i+off_set2 for i in range(cands_size)] + [0] * (max_cls_size - cands_size)).to(device)
                        atom2_target1.append(target_idx1+off_set2)
                        atom2_target2.append(target_idx2)
                        
                        tmp_hatom = index_select_ND(hatom1, 0, torch.LongTensor(attach_atoms).to(device)).sum(dim=0)
                        y_atom_hidden = torch.cat((self.embedding(fnode[yid]), tmp_hatom, z_add_vecs[k,:]), dim=0).unsqueeze(0)
                        y_atom_hidden = y_atom_hidden.expand(cands_size, y_atom_hidden.size(1))
                        atom2_hidden2.append(y_atom_hidden)
                        off_set2 = off_set2 + cands_size
            
            # Prepare the embeddings and targets of attachment point prediction in parent node      
            if len(clusters1) > 0:
                atom1_hidden1 = torch.cat(clusters1, dim=0)
                atom1_hidden2 = torch.cat(atom1_hidden2, dim=0)
                atom1_hiddens1.append(atom1_hidden1)
                atom1_hiddens2.append(atom1_hidden2)
                atom1_targets1.extend(atom1_target1)
                atom1_targets2.extend(atom1_target2)
                atom1_labels.extend(atom1_label)
            
            # Prepare the embeddings and targets of attachment point prediction in child node
            if len(clusters2) > 0:
                atom2_hidden1 = torch.cat(clusters2, dim=0)
                atom2_hidden2 = torch.cat(atom2_hidden2, dim=0)
                atom2_hiddens1.append(atom2_hidden1)
                atom2_hiddens2.append(atom2_hidden2)
                atom2_targets1.extend(atom2_target1)
                atom2_targets2.extend(atom2_target2)            
                atom2_labels.extend(atom2_label)
        

        # node type prediction
        node_hiddens = torch.cat(node_hiddens, dim=0)
        node_targets = torch.cat(node_targets, dim=0)
        node_scores = self.predict(node_hiddens, "node").squeeze(dim=1)
        node_loss = self.node_loss(node_scores, node_targets) / len(orders)
        _, node = torch.max(node_scores, dim=1)
        node_acc = torch.eq(node, node_targets).float()
        node_acc = torch.sum(node_acc) / node_targets.nelement()
        
        # child node connection Prediction
        topo_hiddens = torch.cat(topo_hiddens, dim=0)
        topo_scores = self.predict(topo_hiddens, "topo").squeeze(dim=1)
        topo_targets = torch.cat(topo_targets, dim=0)
        topo_loss = self.topo_loss(topo_scores, topo_targets)/len(orders)
        topo = torch.ge(topo_scores, 0).float()
        topo_acc = torch.eq(topo, topo_targets).float()
        topo_acc = torch.sum(topo_acc) / topo_targets.nelement()
        topo_rec = recall_score(topo_targets.data.to('cpu'), topo.data.to('cpu'))
        
        # attachment point prediction in parent node
        if len(atom1_labels) > 0:
            atom1_labels = torch.LongTensor(atom1_labels).to(device)
            atom1_targets1 = torch.LongTensor(atom1_targets1).to(device)
            atom1_targets2 = torch.stack(atom1_targets2, dim=0)
            atom1_hiddens1 = torch.cat(atom1_hiddens1, dim=0)
            atom1_hiddens2 = torch.cat(atom1_hiddens2, dim=0)
            atom1_scores = self.scoring(atom1_hiddens1, atom1_hiddens2, "atom1")
            atom1_loss, atom1_acc = self.atom_loss(atom1_scores, atom1_targets1, atom1_targets2, atom1_labels)
            atom1_loss = atom1_loss / atom1_targets1.size(0)
            atom1_num = atom1_targets1.size(0)
        else:
            atom1_loss = 0.0
            atom1_acc = 0.0
            atom1_num = 0
        
        # attachment point prediction in child node
        if len(atom2_labels) > 0:
            atom2_labels = torch.LongTensor(atom2_labels).to(device)
            atom2_targets1 = torch.LongTensor(atom2_targets1).to(device)
            atom2_targets2 = torch.stack(atom2_targets2, dim=0)
            atom2_hiddens1 = torch.cat(atom2_hiddens1, dim=0)
            atom2_hiddens2 = torch.cat(atom2_hiddens2, dim=0)
            atom2_scores = self.scoring(atom2_hiddens1, atom2_hiddens2, "atom2")
            atom2_loss, atom2_acc = self.atom_loss(atom2_scores, atom2_targets1, atom2_targets2, atom2_labels)
            atom2_loss = atom2_loss / atom2_targets1.size(0)
            atom2_num = atom2_targets1.size(0)
        else:
            atom2_loss = 0.0
            atom2_acc = 0.0
            atom2_num = 0
        
        loss = (target_loss, del_loss, node_loss, topo_loss, atom1_loss, atom2_loss)
        acc = (tart_acc, del_acc.item(), node_acc.item(), topo_acc.item(), atom1_acc, atom2_acc)
        rec = (del_rec, topo_rec)
        num = (del_one, del_zero, node_targets.size(0), topo_targets.size(0), topo_targets.nonzero().size(0), atom1_num, atom2_num)
        
        return loss, acc, rec, num
    
    def atom_loss(self, scores, targets1, targets2, labels):
        """ calculate the loss of predictions with scores.
        These predictions assign a score for each candidate, and predict the candidate with 
        the maximum score.

        Args:
            scores: the predicted scores for candidates of all predictions at a time step
                    for all molecules within a batch.
            targets1: the index of candidates with the maximum scores for each prediction
            targets2: the index of all candidates for each prediction
            labels: the ground truth label

        Return:
            loss: negative log likelihood loss
            acc: prediction accuracy
        """
        scores = torch.cat([torch.tensor([[0.0]]).to(device), scores], dim=0)
        scores1 = index_select_ND(scores, 0, targets1)
        scores2 = index_select_ND(scores, 0, targets2).squeeze(-1)
        
        mask = torch.zeros(scores2.size()).to(device)
        index = torch.nonzero(targets2)
        mask[index[:,0], index[:,1]] = 1
        
        loss2 = torch.sum(torch.log(torch.sum(torch.exp(scores2) * mask, dim=1)))
        loss = - torch.sum(scores1) + loss2
        
        masked_scores2 = torch.where(targets2==0, torch.FloatTensor([-10]).to(device), scores2)
        acc = torch.sum(torch.argmax(masked_scores2, dim=1) == labels).float() / labels.size(0)
        return loss, acc.item()
        
    
    def scoring(self, vector1, vector2, mode, active="tanh"):
        if self.score_func == 1:
            hidden = torch.cat((vector1, vector2), dim=1)
            scores = self.predict(hidden, mode, active=active)
        else:
            cand_vecs = self.predict(vector2, mode, active=active)
            scores = torch.bmm(vector1.unsqueeze(1), cand_vecs.unsqueeze(2)).squeeze(-1)
        return scores
    
    def predict(self, hiddens, mode, active="relu"):
        if mode == "target":
            V, U = self.W_a, self.U_a
        elif mode == "node":
            V, U = self.W_n, self.U_n
        elif mode == "topo":
            V, U = self.W_t, self.U_t
        elif mode == "delete":
            V, U = self.W_d, self.U_d
        elif mode == "atom1":
            V, U = self.W_a1, self.U_a1
        elif mode == "atom2":
            V, U = self.W_a2, self.U_a2
        else:
            raise ValueError('wrong')
        
        if active == "relu":
            return U(torch.relu(V(hiddens)))
        elif active == "tanh":
            return U(torch.tanh(V(hiddens)))

    
    def decode(self, x_tensors, latent_vecs, x_node_vecs, x_graphs, mol, reselect_num):
        """ Optimize the input molecule for better properties during testing
        Args:
            x_tensors: the embedding of input molecule
            latent_vecs: the sampled latent embedding
            x_node_vecs: the node embeddings of input molecule
            x_graphs: the graph and tree structure of input molecule
        """
        
        x_tree, x_graph = x_graphs
        tree_tensors, graph_tensors = x_tensors
        
        diff_del_vecs = latent_vecs[:self.latent_size]
        diff_add_vecs = latent_vecs[self.latent_size:]
        
        # =================================================================
        # disconnection site and delete fragments prediction

        # disconnection site prediction
        tart_diff_vecs = latent_vecs.repeat((x_node_vecs.size(0)-1, 1))
        scores = self.scoring(x_node_vecs[1:, :], tart_diff_vecs, "target")
        
        _, sort_target_idxs = torch.sort(scores, dim=0, descending=True)
        
        reselect = 0
        for i, target_idx in enumerate(sort_target_idxs[:reselect_num]):
            if i > 0:
                reselect = 1
                print("reselect target node")
            new_tree = x_tree.copy()
            new_graph = x_graph.copy()

            target_idx = target_idx.item() + 1
            tart_atoms = x_tree.nodes[target_idx]['clq']
            
            del_node_idxs = set()
            del_edge_idxs = set()
            del_atom_idxs = set()
            del_bond_idxs = set()
            
            # **************************************************************            
            # delete fragments prediction

            # if the number of connected fragments is greater than 1,
            # we need to predict removal fragments; otherwise, we don't predict removal fragments.
            if len(x_tree.edges(target_idx)) > 1:
                neighbors = torch.LongTensor([edge[1] for edge in x_tree.edges(target_idx)]).to(device)
                neighbors_vecs = index_select_ND(x_node_vecs, 0, neighbors)
                
                del_del_vecs = diff_del_vecs.repeat((neighbors.size(0), 1))
                del_scores = self.predict(torch.cat((neighbors_vecs, del_del_vecs), dim=1), "delete").squeeze()
                dele = torch.ge(del_scores, 0) * neighbors
                del_idxs = dele[dele>0]
        
                # If the disconnection site is a bond node, 
                # then It is illegal to delete two fragments or do not delete fragments
                if len(tart_atoms) == 2:
                    if len(del_idxs) == 2 or len(del_idxs) == 0:
                        continue
                # If the number of deleted fragments is 0,
                # check that the predicted disconnection site has empty valency for new bonds.
                elif len(del_idxs) == 0:
                    attach = False
                    for aid in tart_atoms:
                        atom = mol.GetAtomWithIdx(aid-1)
                        if atom.GetTotalNumHs() > 0:
                            attach = True
                            break
                    if not attach:
                        continue
                
                # get the nodes in the predicted removal fragments.      
                for del_idx in del_idxs:
                    del_idx = del_idx.item()
                    del_node_idxs.add(del_idx)
                    nei_idx = [edge[1] for edge in x_tree.edges(del_idx) if edge[1] != target_idx]
                    del_node_idxs.update(nei_idx)
            
                    while len(nei_idx) > 0:
                        new_nei_idx = []
                        for idx in nei_idx:
                            tmp = [edge[1] for edge in x_tree.edges(idx) if edge[1] not in del_node_idxs]
                            new_nei_idx.extend(tmp)
                        del_node_idxs.update(new_nei_idx)
                        nei_idx = new_nei_idx

                # the number of nodes in removal fragments should be small
                if len(del_node_idxs) < len(x_tree.nodes) / 2:
                    break
            else:
                break
        
        
        # ========================================================================
        # Remove and update deleted nodes

        # remove deleted nodes from tree x
        for idx in del_node_idxs:
            del_atoms = x_tree.nodes[idx]['clq']
            del_atom_idxs.update(del_atoms)
            
            edge_idx1 = [x_tree[edge[0]][edge[1]]['mess_idx'] for edge in x_tree.edges(idx)]
            edge_idx2 = [x_tree[edge[1]][edge[0]]['mess_idx'] for edge in x_tree.edges(idx)]        
            del_edge_idxs.update(edge_idx1)
            del_edge_idxs.update(edge_idx2)
            
            new_tree.remove_node(idx)
        
        # remove atoms in those deleted nodes from graph x
        for idx in del_atom_idxs:
            if idx in tart_atoms: continue
            new_graph.remove_node(idx)
        
        # update atom idxs in graph x
        atom_mapping = {}
        for idx, jdx in enumerate(new_graph.nodes):
            atom_mapping[jdx] = idx+1
            new_graph.nodes[jdx]['bonds'] = []
            new_graph.nodes[jdx]['rings'] = []
        new_graph = nx.relabel_nodes(new_graph, atom_mapping)
       
        # update bond idxs in graph x
        for mess_idx, (i, j) in enumerate(new_graph.edges):
            new_graph[i][j]['mess_idx'] = mess_idx + 1
        
        # update atom idxs in tree x
        node_mapping = {}
        for idx, jdx in enumerate(new_tree.nodes):
            node_mapping[jdx] = idx+1
            
            clq = new_tree.nodes[jdx]['clq']
            new_tree.nodes[jdx]['clq'] = [atom_mapping[aid] for aid in clq]

            for aid in new_tree.nodes[jdx]['clq']:
                if len(clq) > 2:
                    new_graph.nodes[aid]['rings'].append(idx+1)
                else:
                    new_graph.nodes[aid]['bonds'].append(idx+1)
            
            bonds = new_tree.nodes[jdx]['bonds']
            new_tree.nodes[jdx]['bonds'] = [[atom_mapping[aid] for aid in bond] for bond in bonds]

        # update message idxs and anchor atom idxs in tree x
        new_tree = nx.relabel_nodes(new_tree, node_mapping)
        for mess_idx, (i, j) in enumerate(new_tree.edges):
            new_tree[i][j]['mess_idx'] = mess_idx + 1
            anchor = new_tree[i][j]['anchor']
            new_tree[i][j]['anchor'] = [atom_mapping[idx] for idx in anchor]


        # =======================================================================
        # new fragments prediction
 
        # prepare embeddings
        tree_tensors, graph_tensors = MolTree.tensorize_decoding(new_tree, new_graph, self.vocab, self.avocab, extra_len=8)
        
        target_idx = node_mapping[target_idx]
        tree_tensors = [tensor.to(device).long() for tensor in tree_tensors[:-1]] + [tree_tensors[-1]]
        graph_tensors = [tensor.to(device).long() for tensor in graph_tensors[:-1]] + [graph_tensors[-1]]
        nodes = [target_idx]
        amap = []
        step = 0
        max_step = 20
        
        while len(nodes) > 0:
            parent_node = torch.LongTensor([nodes[0]]).to(device)
            try:
                masks = (torch.ones((tree_tensors[0].size(0), 1)).to(device), torch.ones((tree_tensors[1].size(0), 1)).to(device))
            except Exception as e:
                print(e)
            
            mol = graph_to_mol(new_graph)
            
            hatom1 = self.encoder.encode_atom(graph_tensors) 
            
            node_embedding = self.get_node_embedding(tree_tensors, parent_node, masks , hatom1)[0,:]
            
            # child node connection prediction
            topo_hidden = torch.cat((node_embedding, diff_add_vecs), dim=0)
            topo = self.predict(topo_hidden, "topo")
            step += 1
            if step > max_step: break
            
            # stop if topo is false
            if topo.item() <= 0:
                del nodes[0]
                continue
            else:
                # predict the node type
                node_scores = self.predict(topo_hidden, "node")
                _, sort_wid = torch.sort(node_scores, dim=0, descending=True)
                  
                for wid in sort_wid[:10]:
                    # try whether the predicted the new node can be added into the molecule without violating valency law
                    success = self.try_add_mol(new_tree, new_graph, nodes[0], self.vocab.get_smiles(wid), node_embedding, hatom1, diff_add_vecs, amap, mol)
                    
                    if success:
                        # add this new node into the embeddings, trees and graphs of molecule
                        node_idx = len(new_tree.nodes)
                        tree_tensors = self.update_tensor(new_tree, tree_tensors, [node_idx], tree=True)                       

                        atom_idxs = new_tree.nodes[node_idx]['clq']
                        graph_tensors = self.update_tensor(new_graph, graph_tensors, atom_idxs)
                        
                        break
                
                if not success:
                    del nodes[0]
                    continue
                else:
                    nodes.append(len(new_tree.nodes))

        mol = graph_to_mol(new_graph)
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None: pdb.set_trace()
        return mol, reselect
    
    def update_tensor(self, graph, tensors, node_idxs, tree=False):
        """ Add new nodes into the graph structure and embeddings
        """
        #
        if tree:
            fnode, fmess, agraph, bgraph, cgraph, dgraph, _ = tensors
        else:
            fnode, fmess, agraph, bgraph, cgraph, dgraph = tensors
        
        for node_idx in node_idxs:
            if tree:
                node_wid = self.vocab[graph.nodes[node_idx]['label']]
                clq = graph.nodes[node_idx]['clq']
                dgraph = torch.cat((dgraph, torch.tensor([clq + [0] * (dgraph.size(1)-len(clq))]).to(device).long()), dim=0)
            else:
                node_wid = self.avocab[graph.nodes[node_idx]['label']]
            
            if node_idx >= fnode.size(0):
                fnode = torch.cat((fnode, torch.tensor([node_wid]).to(device).long()), dim=0)
                
        edge_list = [(e[0], e[1], graph[e[0]][e[1]]['mess_idx']) for e in graph.edges]
        edge_list = sorted(edge_list, key=lambda e: e[2])
            
        for edge in edge_list:
            if edge[0] not in node_idxs and edge[1] not in node_idxs: continue
            if edge[2] < fmess.shape[0]: continue
                
            if tree:
                fmess = torch.cat((fmess, torch.tensor([[edge[0], edge[1], 0]]).to(device).long()), dim=0)
            else:
                fmess = torch.cat((fmess, torch.tensor([[edge[0], edge[1], graph[edge[0]][edge[1]]['label']]]).to(device).long()), dim=0)
            
            # update the neighbor edges of edge[0] in the agraph
            if edge[0] < agraph.size(0):
                mess_idx = [graph[e[1]][e[0]]['mess_idx'] for e in graph.edges(edge[0])]
                agraph[edge[0]] = torch.tensor(mess_idx + [0] * (agraph.size(1) - len(mess_idx))).to(device).long()
            else:
                mess_idx = [graph[e[1]][e[0]]['mess_idx'] for e in graph.edges(edge[0])]
                agraph = torch.cat((agraph, torch.tensor([mess_idx + [0] * (agraph.size(1) - len(mess_idx))]).to(device).long()), dim=0)
            
            tmp = []
            for w in graph.predecessors(edge[0]):
                if w == edge[1]: continue
                tmp.append( graph[w][edge[0]]['mess_idx'] )
            bgraph = torch.cat((bgraph, torch.tensor([tmp+[0] * (bgraph.size(1) - len(tmp))]).to(device).long()), dim=0)
   
            if tree:
                anchor = graph[edge[0]][edge[1]]['anchor']
                cgraph = torch.cat((cgraph, torch.tensor([anchor + [0] * (cgraph.size(1) - len(anchor))]).to(device).long()), dim=0) 
        
        if tree:
            return [fnode, fmess, agraph, bgraph, cgraph, dgraph] + tensors[6:]
        else:
            return [fnode, fmess, agraph, bgraph] + tensors[4:]

    def try_add_mol(self, tree, graph, node_idx, smiles, hnode, hatoms, diff_add_vec, amap, old_mol):
        """ determine whether the predicted new node can be attached to the parent node or not.
        """
        atoms = tree.nodes[node_idx]['clq']
        x_label = tree.nodes[node_idx]['label']
        mol = get_mol(smiles)
        attach_atom = None
        
        node_atom = index_select_ND(hatoms, 0, torch.tensor(tree.nodes[node_idx]['clq']).to(device).long())
        fnode = self.encoder.W_i(torch.cat((self.embedding(torch.tensor(self.vocab[x_label]).to(device).long()), torch.sum(node_atom, dim=0)), dim=0))
        
        # If the parent node is a bond
        if len(atoms) == 2:
            # the candidate attachment atom must be connected with only one edge
            attach_atom = [old_mol.GetAtomWithIdx(idx-1) for idx in atoms if len(graph.edges(idx)) == 1]
            if len(attach_atom) > 0:
                attach_atom = attach_atom[0]
            else:
                attach_atom = None

        # If the parent node is a ring
        elif len(atoms) > 3:
            x_label = torch.tensor(self.vocab[tree.nodes[node_idx]['label']]).to(device)
            y_label = torch.tensor(self.vocab[smiles]).to(device)
            
            atom_hidden1 = torch.cat((self.embedding(y_label), fnode, hnode, diff_add_vec), dim=0)
            
            # If the child node is not a ring
            if mol.GetNumAtoms() < 3:
                # the attachment is an atom
                # predict the scores of all atoms in parent node
                uniq_node_atom, inverse_idx = unique_tensor(node_atom)
                atom_hidden1 = atom_hidden1.repeat(uniq_node_atom.size(0), 1)
                atom_scores = self.scoring(atom_hidden1, uniq_node_atom, "atom1")

                # select the top-ranked atom as candidate
                _, ranked_idxs = torch.sort(atom_scores, dim=0, descending=True)
                for idx in ranked_idxs:
                    for idx1, idx2 in enumerate(inverse_idx):
                        if idx != idx2: continue
                        atom = old_mol.GetAtomWithIdx(atoms[idx1]-1)
                        label, cands = atom_cand(atom, mol, amap)
                        if label and len(graph.nodes[atoms[idx1]]['bonds']) == 0 and len(graph.nodes[atoms[idx1]]['rings']) <= 2:
                            attach_atom = atom
                            break
                    if attach_atom is not None:
                        break
                    
            # If the child node is a ring
            else:
                # the attachment can be an atom or a bond
                # predict the scores of atoms and bonds in parentnode
                node_atom1 = index_select_ND(hatoms, 0, torch.tensor(tree.nodes[node_idx]['bonds']).to(device).long()).sum(dim=1)
                node_atom2 = node_atom + torch.zeros((node_atom.size(0), 1)).to(device)
                node_atom = torch.cat((node_atom1, node_atom2), dim=0)
                uniq_node_atom, inverse_idxs = unique_tensor(node_atom)
                
                atom_hidden1 = atom_hidden1.repeat(uniq_node_atom.size(0), 1)
                atom_scores = self.scoring(atom_hidden1, uniq_node_atom, "atom1")
                
                # select the top-ranked atom or bond as candidate
                _, ranked_idxs = torch.sort(atom_scores, dim=0, descending=True)
                attach_atom = None
                num_bonds = len(tree.nodes[node_idx]['bonds']) 
                ranked_idxs = ranked_idxs.squeeze()
                for i in ranked_idxs:
                    i = i.item()
                    for idx, j in enumerate(inverse_idxs):
                        if i != j: continue
                        # If the attachment is an atom
                        if idx >= num_bonds:
                            atom = old_mol.GetAtomWithIdx(atoms[idx-num_bonds]-1)
                            label, cands = atom_cand(atom, mol, amap)
                            
                            if label and len(graph.nodes[atoms[idx-num_bonds]]['bonds']) < 2 and len(graph.nodes[atoms[idx-num_bonds]]['rings']) == 1:
                                attach_atom = atom
                                break

                        # If the attachment is a bond
                        else:
                            bond_idx = tree.nodes[node_idx]['bonds'][idx]
                            begin_atom = old_mol.GetAtomWithIdx(bond_idx[0]-1)
                            end_atom = old_mol.GetAtomWithIdx(bond_idx[1]-1)
                            bond = old_mol.GetBondBetweenAtoms(bond_idx[0]-1, bond_idx[1]-1)
                            label, cands = bond_cand(begin_atom, end_atom, int(bond.GetBondTypeAsDouble()), old_mol)
                            
                            if label and len(graph.nodes[bond_idx[0]]['bonds']) < 2 and len(graph.nodes[bond_idx[1]]['bonds']) < 2\
                                and len(graph.nodes[bond_idx[0]]['rings']) < 2 and len(graph.nodes[bond_idx[1]]['rings']) < 2:
                                attach_atom = (begin_atom, end_atom)
                                break
                    if attach_atom is not None:
                        break

        # If the parent node is a singleton
        elif len(atoms) == 1:
            if mol.GetNumAtoms() == 1:
                return False
            else:
                attach_atom = old_mol.GetAtomWithIdx(atoms[0]-1)


        # fail to add new nodes if we cannot find attachment in parent node
        if attach_atom is None:
            return False
        
        # If the child node is a bond
        if mol.GetNumAtoms() == 2:
            begin_atom = mol.GetAtomWithIdx(0)
            end_atom = mol.GetAtomWithIdx(1)
            
            match_atom = None 
            if atom_equal(begin_atom, attach_atom):
                if begin_atom.GetTotalNumHs() + attach_atom.GetTotalNumHs() >= attach_atom.GetTotalValence():
                    match_atom = begin_atom
            elif atom_equal(end_atom, attach_atom):
                if end_atom.GetTotalNumHs() + attach_atom.GetTotalNumHs() >= attach_atom.GetTotalValence():
                    match_atom = end_atom
            else:
                return False
            if match_atom is None: return False 
            graph, anchor, clq, node_amap = attach_mol_graph(graph, mol, [attach_atom.GetIdx()], [match_atom.GetIdx()])
             
            update_tree(tree, smiles, node_idx, [attach_atom.GetIdx()+1], clq, node_amap)
            update_bonds_rings(tree, graph)
            return True

        # If the child node is a singleton
        elif mol.GetNumAtoms() == 1:
            atom = mol.GetAtomWithIdx(0)
            
            if atom_equal(atom, attach_atom):
                idx = len(tree) + 1
                amap.append( (attach_atom.GetIdx(), idx) )
                atom_idx = attach_atom.GetIdx() + 1
                update_tree(tree, smiles, node_idx, [atom_idx], [atom_idx], {atom_idx: atom_idx})
                update_bonds_rings(tree, graph)
                return True
            else:
                return False
        
        # If the child node is a ring (need atom 2 prediction)
        elif mol.GetNumAtoms() > 2:
            atoms2 = mol.GetAtoms()
            mol_graph = mol_to_graph(mol)
            mol_tensor = MolTree.tensorize_graph([mol_graph], self.avocab, tree=False, extra_len=4)
             
            mol_tensor = mol_tensor[0] + (None,)
            make_tensor = lambda x: x.to(device).long() if type(x) is torch.Tensor else x
            mol_tensor = [make_tensor(tensor) for tensor in mol_tensor]
            mol_atoms = self.encoder.encode_atom(mol_tensor)
            y_label = torch.tensor(self.vocab[smiles]).to(device)
            attach_atom1 = None

            # If the attach_atom in parent node is an atom
            if type(attach_atom) is Chem.Atom:
                tmpatom = hatoms[attach_atom.GetIdx()+1, :]
                atom_hidden1 = torch.cat((self.embedding(y_label), tmpatom, diff_add_vec), dim=0)
                atom_hidden2 = mol_atoms[1:, :]
                
                uniq_atom_hidden2, inverse_idx = unique_tensor(atom_hidden2)
                atom_hidden1 = atom_hidden1.repeat(uniq_atom_hidden2.size(0), 1)
                scores = self.scoring(atom_hidden1, uniq_atom_hidden2, "atom2")
                _, ranked_idxs = torch.sort(scores, dim=0, descending=True)

                for idx in ranked_idxs:
                    for idx1, idx2 in enumerate(inverse_idx):
                        if idx != idx2: continue
                        cand_atom1 = mol.GetAtomWithIdx(idx1)
                        
                        if atom_equal(cand_atom1, attach_atom):
                            if cand_atom1.GetTotalNumHs() + attach_atom.GetTotalNumHs() >= attach_atom.GetTotalValence():
                                attach_atom1 = cand_atom1
                                break
                
                if attach_atom1 is not None:
                    graph, anchor, clq, node_amap = attach_mol_graph(graph, mol, [attach_atom.GetIdx()], [attach_atom1.GetIdx()])
                    update_tree(tree, smiles, node_idx, anchor, clq, node_amap)
                    update_bonds_rings(tree, graph)               
                    return True
                else:
                    return False

            # If the attach_atom in parent node is a bond
            else:
                bonds = [[bond.GetBeginAtomIdx()+1, bond.GetEndAtomIdx()+1] for bond in mol.GetBonds()]
                bonds = torch.tensor(bonds).to(device).long()
                tmpatom = hatoms[attach_atom[0].GetIdx()+1, :] + hatoms[attach_atom[1].GetIdx()+1, :]
                atom_hidden1 = torch.cat((self.embedding(y_label), tmpatom, diff_add_vec), dim=0)
                atom_hidden2 = index_select_ND(mol_atoms, 0, bonds).sum(dim=1)
                
                uniq_atom_hidden2, inverse_idx = unique_tensor(atom_hidden2)
                atom_hidden1 = atom_hidden1.repeat(uniq_atom_hidden2.size(0), 1)
                scores = self.scoring(atom_hidden1, uniq_atom_hidden2, "atom2")
                _, ranked_idxs = torch.sort(scores, dim=0, descending=True)
                
                atom1_idxs, atom2_idxs = [],[]            
                for idx in ranked_idxs:
                    bond_match = 0
                    for idx1, idx2 in enumerate(inverse_idx):
                        if idx != idx2: continue
                        begin_idx = bonds[idx1,0].item() - 1
                        end_idx = bonds[idx1,1].item() - 1

                        bond2 = mol.GetBondBetweenAtoms(begin_idx, end_idx)
                        bond1 = old_mol.GetBondBetweenAtoms(attach_atom[0].GetIdx(), attach_atom[1].GetIdx())
                        
                        bond_match = bond_equal(bond1, bond2)

                        atom2_begin = bond2.GetBeginAtom()
                        atom2_end = bond2.GetEndAtom()
                        if bond_match == 1:
                            if atom2_begin.GetTotalNumHs() + attach_atom[0].GetTotalNumHs() >= attach_atom[0].GetTotalValence() and \
                               atom2_end.GetTotalNumHs() + attach_atom[1].GetTotalNumHs() >= attach_atom[1].GetTotalValence():
                                atom1_idxs = [attach_atom[0].GetIdx(), attach_atom[1].GetIdx()]
                                atom2_idxs = [atom2_begin.GetIdx(), atom2_end.GetIdx()]
                                break
                        elif bond_match == 2:
                            if atom2_begin.GetTotalNumHs() + attach_atom[0].GetTotalNumHs() >= attach_atom[0].GetTotalValence() and \
                               atom2_end.GetTotalNumHs() + attach_atom[1].GetTotalNumHs() >= attach_atom[1].GetTotalValence():
                                atom1_idxs = [attach_atom[0].GetIdx(), attach_atom[1].GetIdx()]
                                atom2_idxs = [atom2_end.GetIdx(), atom2_begin.GetIdx()]
                                break
                    if bond_match == 0: continue

                if len(atom1_idxs) > 0:
                    graph, anchor, clq, node_amap = attach_mol_graph(graph, mol, atom1_idxs, atom2_idxs)
                    update_tree(tree, smiles, node_idx, anchor, clq, node_amap)
                    update_bonds_rings(tree, graph)
                    return True
                else:
                    # print("bond edge not match")
                    return False
                          
def update_bonds_rings(tree, graph):
    child_idx = len(tree)
    clq = tree.nodes[child_idx]['clq']
    for aid in clq:
        if len(clq) > 2:
            graph.nodes[aid]['rings'].append(child_idx)
        else:
            graph.nodes[aid]['bonds'].append(child_idx)

def update_tree(tree, smiles, parent_idx, anchor, clq, amap):
    """
    """
    child_idx = len(tree) + 1
    tree.add_node(child_idx)
    tree.nodes[child_idx]['label'] = smiles
                
    tree.add_edge(parent_idx, child_idx)
    tree[parent_idx][child_idx]['mess_idx'] = len(tree.edges)
    tree.add_edge(child_idx, parent_idx)
    tree[child_idx][parent_idx]['mess_idx'] = len(tree.edges)
    
    tree[parent_idx][child_idx]['anchor'] = anchor
    tree[child_idx][parent_idx]['anchor'] = anchor
    tree.nodes[child_idx]['clq'] = clq
    
    tree.nodes[child_idx]['bonds'] = [[amap[bond.GetBeginAtomIdx()], amap[bond.GetEndAtomIdx()]] for bond in Chem.MolFromSmiles(smiles).GetBonds()]

def atom_equal(atom, label):
    if type(label) is set:
        return atom.GetSymbol() == label[0] and atom.GetFormalCharge() == label[1]
    else:
        return atom.GetSymbol() == label.GetSymbol() and atom.GetFormalCharge() == label.GetFormalCharge()

def bond_cand(begin_atom, end_atom, bond_val1, mol):
    """ Find the bond candidate from molecule mol
    Args:
        begin_atom: the bond candidate should have the same atom type with begin_atom
        end_atom: the bond candidate should have the same atom type with end_atom
        bond_val1: the type of bond (single/double/triple)
    """
    bonds = mol.GetBonds()
    cands = []
    for bond in bonds:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        bond_val2 = int(bond.GetBondTypeAsDouble())
        if atom_equal(atom1, begin_atom) and atom_equal(atom2, end_atom) and bond_val1 == bond_val2:
            if atom1.GetTotalNumHs() + begin_atom.GetTotalNumHs() + bond_val1 >= atom1.GetTotalValence():
                if atom2.GetTotalNumHs() + end_atom.GetTotalNumHs() + bond_val1 >= atom2.GetTotalValence():
                    cands.append((atom1, atom2))
                    continue
        elif atom_equal(atom2, begin_atom) and atom_equal(atom1, end_atom) and bond_val1 == bond_val2:
            if atom2.GetTotalNumHs() + begin_atom.GetTotalNumHs() + bond_val1 >= atom2.GetTotalValence():
                if atom1.GetTotalNumHs() + end_atom.GetTotalNumHs() + bond_val1 >= atom1.GetTotalValence():
                    cands.append((atom2, atom1))
                    continue
   
    if len(cands) > 0:
        return True, cands
    else:
        return False, cands
 
def atom_cand(atom1, mol, amap):
    """ Find the atom candidate from molecule mol
    """
    num_bonds = mol.GetNumBonds()
    
    if num_bonds == 0: # new child node is singleton
        used_list = [atom_idx for atom_idx, _ in amap]
        if atom1.GetIdx() in used_list: return False, 0
        atom2 = mol.GetAtomWithIdx(0)
        
        if atom_equal(atom2, atom1):
            return True, [atom2]
        else:
            return False, 0
    elif num_bonds == 1: # new child node is bond
        bond = mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        if atom1.GetTotalNumHs() < bond_val:
            return False, 0
        if atom_equal(b1, atom1):
            return True, [b1]
        if atom_equal(b2, atom1):
            return True, [b2]
    elif num_bonds > 1: # new child node is ring
        cands = []
        for atom in mol.GetAtoms():
            if atom_equal(atom, atom1):
                if atom.GetTotalNumHs() + atom1.GetTotalNumHs() >= atom.GetTotalValence():
                    cands.append(atom)
        if len(cands) > 0:
            return True, cands

    return False, 0


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.W_p = nn.Linear(2 * hidden_size, hidden_size).to(device)

    def forward(self, cur_x, cur_fh, cur_fc):
        z_input = torch.cat((cur_x, cur_fh), dim=1)
        z_i = torch.sigmoid(self.W_i(z_input))
        z_o = torch.sigmoid(self.W_o(z_input))
        z_c = torch.tanh(self.W_c(z_input))
        z_f = torch.sigmoid(self.W_f(z_input))
        
        cur_xc = z_f * cur_fc + z_i * z_c
        cur_xh = z_o * torch.tanh(cur_xc)
        
        z_p = torch.sigmoid(self.W_p(torch.cat([cur_xh, cur_fh], dim=1)))
        
        cur_fh = z_p * torch.tanh(cur_fh)
        
        return cur_xh, cur_xc, cur_fh
