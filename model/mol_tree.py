import os
import pickle
import rdkit
import argparse
import sascorer
import rdkit.Chem as Chem
import networkx as nx
from rdkit.Chem import Descriptors
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, BOND_LIST
from nnutils import create_pad_tensor
from vocab import *
from properties import penalized_logp
import torch

class MolTree(object):
    
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        
        self.mol_graph = self.build_mol_graph()
        self.cliques, self.edges = tree_decomp(self.mol)
        self.mol_tree = self.build_mol_tree()
        self.order = []
        self.set_anchor()
        
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph
        
    def build_mol_tree(self):
        cliques = self.cliques
        graph = nx.DiGraph()
        
        for i, clique in enumerate(cliques):
            cmol = get_clique_mol(self.mol, clique)
            graph.add_node(i)
            graph.nodes[i]['label'] = get_smiles(cmol)
            graph.nodes[i]['clq'] = clique
            
        for edge in self.edges:
            inter_atoms = list(set(cliques[edge[0]]) & set(cliques[edge[1]]))
            
            graph.add_edge(edge[0], edge[1])
            graph.add_edge(edge[1], edge[0])
            graph[edge[0]][edge[1]]['anchor'] = inter_atoms
            graph[edge[1]][edge[0]]['anchor'] = inter_atoms
            
            if len(inter_atoms) == 1:
                graph[edge[0]][edge[1]]['label'] = cliques[edge[0]].index(inter_atoms[0])
                graph[edge[1]][edge[0]]['label'] = cliques[edge[1]].index(inter_atoms[0])
            elif len(inter_atoms) == 2:
                index1 = cliques[edge[0]].index(inter_atoms[0])
                index2 = cliques[edge[0]].index(inter_atoms[1])
                if index2 == len(cliques[edge[0]])-1:
                    index2 = -1
                graph[edge[0]][edge[1]]['label'] = max(index1, index2)
                
                index1 = cliques[edge[1]].index(inter_atoms[0])
                index2 = cliques[edge[1]].index(inter_atoms[1])
                if index2 == len(cliques[edge[1]])-1:
                    index2 = -1
                graph[edge[1]][edge[0]]['label'] = max(index1, index2)
                
        return graph
   
    def set_anchor(self):
        for i, clique in enumerate(self.cliques):
            self.mol_tree.nodes[i]['bonds'] = []
        
        for bond in self.mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            for i, clique in enumerate(self.cliques):
                if begin_idx not in clique or end_idx not in clique:
                    continue
                else:
                    self.mol_tree.nodes[i]['bonds'].append([ begin_idx, end_idx])
                    
        for edge in self.mol_tree.edges:
            inter_atoms = list(set(self.cliques[edge[0]]) & set(self.cliques[edge[1]]))
            
            if len(inter_atoms) == 1:
                self.mol_tree[edge[0]][edge[1]]['label'] = self.cliques[edge[0]].index(inter_atoms[0])
                self.mol_tree[edge[1]][edge[0]]['label'] = self.cliques[edge[1]].index(inter_atoms[0])
             
            elif len(inter_atoms) == 2:
                index1 = inter_atoms[0]
                index2 = inter_atoms[1]
                
                if [index1, index2] in self.mol_tree.nodes[edge[0]]['bonds']:
                    self.mol_tree[edge[0]][edge[1]]['label'] = self.mol_tree.nodes[edge[0]]['bonds'].index([index1, index2])
                else:
                    self.mol_tree[edge[0]][edge[1]]['label'] = self.mol_tree.nodes[edge[0]]['bonds'].index([index2, index1])
                
                if [index1, index2] in self.mol_tree.nodes[edge[1]]['bonds']:
                    self.mol_tree[edge[1]][edge[0]]['label'] = self.mol_tree.nodes[edge[1]]['bonds'].index([index1, index2])
                else:
                    self.mol_tree[edge[1]][edge[0]]['label'] = self.mol_tree.nodes[edge[1]]['bonds'].index([index2, index1])

    def set_revise(self, target_idx, revise_idxs):
        tree = self.mol_tree
        def bfs(order, visited, nodes):
            visited.extend(nodes)
            new_nodes = []
            for x in nodes:
                sorted_child = sorted([ edge[1] for edge in self.mol_tree.edges(x) if edge[1] not in visited ]) #better performance with fixed order
                for idx,y in enumerate(sorted_child):
                    order.append((x, y))
                new_nodes.extend(sorted_child)
            
            if len(new_nodes) > 0:
                bfs(order, visited, new_nodes)
        
        self.revise_nodes = []
        for i, cls in enumerate(self.cliques):
            if target_idx == i:
                tree.nodes[i]['target'] = 1
            else:
                tree.nodes[i]['target'] = 0
            
            if i in revise_idxs:
                tree.nodes[i]['revise'] = 1
                self.revise_nodes.append(i)
            else:
                tree.nodes[i]['revise'] = 0
        
        order = []
        visited = [edge[1] for edge in self.mol_tree.edges(target_idx) if edge[1] not in self.revise_nodes]
        bfs(order, visited, [target_idx])
        self.order = order
        
    @staticmethod
    def tensorize(mol_batch, vocab, avocab, target=False, add_target=False):
        scores = []
        del_num = 0
        is_break = False
        for i in range(len(mol_batch)):
            mol = mol_batch[i-del_num]
            mol.set_anchor()
            for j, clique in enumerate(mol.cliques):
                cmol = get_clique_mol(mol.mol, clique)
            for u, v in mol.mol_tree.edges:
                if len(mol.mol_tree[u][v]['anchor']) > 2:
                    print(mol.smiles)
                    del mol_batch[i-del_num]
                    del_num += 1
                    is_break = True
                    break
            if not is_break:
                scores.append(penalized_logp(mol.smiles))
            else:
                is_break = False
        scores = torch.FloatTensor(scores)
        
        tree_tensors, tree_batchG = MolTree.tensorize_graph([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = MolTree.tensorize_graph([x.mol_graph for x in mol_batch], avocab, tree=False)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]
        
        # Add anchor atom index
        cgraph = torch.zeros(len(tree_batchG.edges) + 1, 2).int()
        for u,v,attr in tree_batchG.edges(data=True):
            eid = attr['mess_idx']
            anchor = tree_batchG[u][v]['anchor']
            cgraph[eid, :len(anchor)] = torch.LongTensor(anchor)
                
        # Add all atom index
        max_cls_size = max( [len(c) for x in mol_batch for c in x.cliques] )
        dgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).long()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['clq'] = cls = [x + offset for x in attr['clq']]
            tree_batchG.nodes[v]['bonds'] = [(x + offset, y+offset) for x, y in attr['bonds']]
            dgraph[v, :len(cls)] = torch.LongTensor(cls)
        
        # Add atom mess index
        egraph = torch.zeros(len(graph_batchG)+1, len(graph_batchG)+1).long()
        for u, v, attr in graph_batchG.edges(data=True):
            eid = attr['mess_idx']
            egraph[u, v] = eid
        
        all_orders = []
        max_rev_size = max( [len(x.order) for x in mol_batch])
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, tree_tensors[0][y+offset]) for x,y in hmol.order]
            if add_target:
                target_idx = [(None, x + offset) for x in hmol.mol_tree.nodes if hmol.mol_tree.nodes[x]['target']]
                all_orders.append(target_idx + order)
            else:
                all_orders.append(order)
        
        tree_tensors = tree_tensors[:4] + (cgraph, dgraph, tree_scope)
        graph_tensors = graph_tensors[:4] + (egraph, graph_scope)
        
        if target:
            node_mask = torch.ones(len(tree_batchG)+1, 1).int()
            edge_mask = torch.ones(len(tree_batchG.edges)+1, 1).int()
            
            atom_mask = torch.zeros(len(graph_batchG)+1, 1).int()
            bond_mask = torch.ones(len(graph_batchG.edges)+1, 1).int()
            
            try:
                for v,attr in tree_batchG.nodes(data=True):
                    if attr['revise']:
                        node_mask[v] = 0
                    else:
                        atom_mask.scatter_(0, dgraph[v,:len(attr['clq'])].unsqueeze(1), 1)
            except Exception as e:
                print(e)
                pdb.set_trace()
            
            for u, v in tree_batchG.edges:
                if tree_batchG.nodes[u]['revise'] or tree_batchG.nodes[v]['revise']:
                    edge_mask[tree_batchG[u][v]['mess_idx']] = 0
            
            mask1 = torch.ones(len(graph_batchG)+1, 1).int()
            mask2 = torch.zeros(len(graph_batchG)+1, 1).int()
            masked_atoms = torch.where(atom_mask==0, atom_mask, mask2)
            masked_atoms = torch.where(atom_mask>0, masked_atoms, mask1)
            masked_atoms = masked_atoms.nonzero()[:,0]
            
            mess_list = []
            for a1 in masked_atoms[1:]:
                a1 = a1.item()
                mess = torch.LongTensor([graph_batchG[a1][edge[1]]['mess_idx'] for edge in graph_batchG.edges(a1)])
                mess_list.append(mess)
           
            try: 
                mess = torch.cat(mess_list, dim=0).unsqueeze(1)
            except:
                pdb.set_trace()
            bond_mask.scatter_(0, mess, 1)
               
            tree_tensors = tree_tensors[:-1] + (node_mask, edge_mask, tree_scope)
            graph_tensors = graph_tensors[:-1] + (atom_mask, bond_mask, graph_scope)
            return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders, scores
        else:
            return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders, scores

    @staticmethod
    def tensorize_decoding(tree, graph, vocab, avocab, extra_len=0):
        tree_tensors, tree_batchG = MolTree.tensorize_graph([tree], vocab, atom_num=0, extra_len=extra_len)
        graph_tensors, graph_batchG = MolTree.tensorize_graph([graph], avocab, tree=False, extra_len=extra_len)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]
        
        # Add anchor atom index
        cgraph = torch.zeros(len(tree_batchG.edges) + 1, 2).int()
        for u,v,attr in tree_batchG.edges(data=True):
            eid = attr['mess_idx']
            anchor = tree_batchG[u][v]['anchor']
            cgraph[eid, :len(anchor)] = torch.LongTensor(anchor)
                
        # Add all atom index
        max_cls_size = vocab.max_len
        dgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).long()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            tree_batchG.nodes[v]['clq'] = cls = [x for x in attr['clq']]
            dgraph[v, :len(cls)] = torch.LongTensor(cls)
        
        # Add atom mess index
        egraph = torch.zeros(len(graph_batchG)+1, len(graph_batchG)+1).long()
        for u, v, attr in graph_batchG.edges(data=True):
            eid = attr['mess_idx']
            egraph[u, v] = eid
        
        tree_tensors = tree_tensors[:4] + (cgraph, dgraph, tree_scope)
        graph_tensors = graph_tensors[:4] + (egraph, graph_scope)
    
        return tree_tensors, graph_tensors

    @staticmethod
    def tensorize_graph(graph_batch, vocab, tree=True, atom_num=1, extra_len=0):
        fnode,fmess = [None],[(0,0,0)]
        agraph,bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])
            
            for u, v, attr in G.edges(data='label'):
                if tree:
                    fmess.append( (u, v, 0) )
                else:
                    fmess.append( (u, v, attr) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                
                if tree:
                    anchor = G[u][v]['anchor']
                    G[u][v]['anchor'] = [a+atom_num for a in anchor]
                agraph[v].append(eid)
                bgraph.append([])
                
            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )
            
            if tree:
                atom_num += max([max(G.nodes[idx]['clq']) for idx in G.nodes]) + 1
        
        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph, extra_len=extra_len)
        bgraph = create_pad_tensor(bgraph, extra_len=extra_len)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)
    
def dfs(node, fa_nid):
    max_depth = 0
    for child in node.neighbors:
        if child.nid == fa_nid: continue
        max_depth = max(max_depth, dfs(child, node.nid))
    return max_depth + 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--train", type=str, default="./data/logp06/", help="the path of training data")
    parser.add_argument("-o", "--out", type=str, default="./data/vocab.txt", help="the path of vocabulary")
    args = parser.parse_args()
    
    cset = set()
    for file_name in os.listdir(args.train):
        if not file_name.endswith("pkl"): continue
        with open(os.path.join(args.train, file_name), 'rb') as f:
            pairs_data = pickle.load(f)
            for molx, moly, path in pairs_data:
                cset.update(set([label for _, label in molx.mol_tree.nodes(data='label')]))
                cset.update(set([label for _, label in moly.mol_tree.nodes(data='label')]))
     
    with open(args.out, 'w') as f:
        for word in cset:
            f.write("%s\n" % word)
