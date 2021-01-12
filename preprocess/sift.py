import pdb
import pickle
import sys, os
sys.path.insert(0, os.path.abspath('/fs/scratch/PCON0005/che268/graphoptimization/model/'))
from mol_tree import MolTree
from vocab import Vocab
from rdkit import Chem

def is_attach_node(tree1, tree2, idx, dict1, dict2):
    if dict1[idx] is None:
        return False
        
    new_idx = dict1[idx]
    old_neis = [edge[1] for edge in tree1.edges(idx)]
    for nei in old_neis:
        if dict1[nei] is None:
            return True
    
    new_neis = [edge[1] for edge in tree2.edges(new_idx)]
    for nei in new_neis:
        if dict2[nei] is None:
            return True
    
    return False
    
def get_dict(path):
    dict1 = dict()
    dict2 = dict()
    delete_nodes = []
    add_nodes = []
    for npair in path:
        if npair[0] is not None:
            dict1[npair[0]] = npair[1]
        else:
            add_nodes.append(npair[1])
            
        if npair[1] is not None:
            dict2[npair[1]] = npair[0]
        else:
            delete_nodes.append(npair[0])
            
    return dict1, dict2, delete_nodes, add_nodes

def sift_molecules(data):
    sift_data = []
    
    max_del_node, max_add_node = 0,0
    max_del_mol1, max_del_mol2 = None, None
    max_add_mol1, max_add_mol2 = None, None
    add_dicts, del_dicts, att_dicts = {}, {}, {}
    for ent in data:
        tree1, tree2, path = ent
        path = path[0]
        dict1, dict2, delete_nodes, add_nodes = get_dict(path[0])
        
        tmp_add, tmp_del, tmp_att = 0,0,0
        tmp_att_list = []
        for npair in path[0]:
            if npair[1] is None:
                tmp_del += 1
            
        for npair in path[0]:
            if npair[0] is None:
                tmp_add += 1
        
        if max_del_node < tmp_del:
            max_del_node = tmp_del
            max_del_mol1 = Chem.MolFromSmiles(tree1.smiles)
            max_del_mol2 = Chem.MolFromSmiles(tree2.smiles)
            
        if max_add_node < tmp_add:
            max_add_node = tmp_add
            max_add_mol1 = Chem.MolFromSmiles(tree1.smiles)
            max_add_mol2 = Chem.MolFromSmiles(tree2.smiles)
            
        for npair in path[0]:
            if npair[0] is None or npair[1] is None: continue
            if is_attach_node(tree1.mol_tree, tree2.mol_tree, npair[0], dict1, dict2):
                tmp_att += 1
                tmp_att_list.append(npair)
        
        if tmp_add not in add_dicts:
            add_dicts[tmp_add] = [ent]
        else:
            add_dicts[tmp_add].append(ent)
        
        if tmp_del not in del_dicts:
            del_dicts[tmp_del] = [ent]
        else:
            del_dicts[tmp_del].append(ent)
        
        if tmp_att not in att_dicts:
            att_dicts[tmp_att] = [ent]
        else:
            att_dicts[tmp_att].append(ent)
    
        if tmp_att == 1:
            npair = tmp_att_list[0]
            tree1.set_revise(npair[0], delete_nodes)
            tree2.set_revise(npair[1], add_nodes)
            sift_data.append((tree1, tree2, path))

    return sift_data



if __name__ == "__main__":
    file_name = "../data/logp06/tensors-1.pkl"
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    sift_data = sift_molecules(data)
    
    with open('../data/logp06/new_tensors-1.pkl', 'wb') as f:
        pickle.dump(sift_data, f, pickle.HIGHEST_PROTOCOL)
