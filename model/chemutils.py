# Modified from https://github.com/wengong-jin/iclr19-graph2graph
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from vocab import Vocab

MST_MAX_WEIGHT = 100 
MAX_NCAND = 100
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

def bond_equal(bond1, bond2):
    begin_atom1 = bond1.GetBeginAtom()
    end_atom1 = bond1.GetEndAtom()
    bond_val1 = int(bond1.GetBondTypeAsDouble())
    
    begin_atom2 = bond2.GetBeginAtom()
    end_atom2 = bond2.GetEndAtom()
    bond_val2 = int(bond2.GetBondTypeAsDouble())
    if bond_val1 != bond_val2: return 0
    if atom_equal(begin_atom1, begin_atom2) and atom_equal(end_atom1, end_atom2):
        return 1
    elif atom_equal(begin_atom1, end_atom2) and atom_equal(begin_atom2, end_atom1):
        return 2

def get_uniq_atoms(graphs, cliques, attach_atoms, label, avocab):
    local_dict = []
    adj_mat = [{} for _ in cliques]
    aidxs = []
    pdb.set_trace()
    for i, atom in enumerate(cliques):
        aidxs.append(avocab[graphs.nodes[atom]['label']])
        edges = [edge[1] for edge in graphs.edges(atom) if edge[1] in cliques]
        adj_mat[i][1] = []
        for atom2 in edges:
            adj_mat[i][1].append((avocab[graphs.nodes[atom2]['label']], graphs[atom][atom2]['label']))
        
        adj_mat[i][1].sort()
    
    unmatched_idxs = [i for i in range(len(cliques))]
    matched_idxs = {}
    visited_idxs = []
    while len(unmatched_idx) > 0:
        for i, idx in enumerate(unmatched_idxs[:-1]):
            for jdx in unmatched_idxs[i+1:]:
                if adj_mat[idx] == adj_mat[jdx]:
                    if idx not in matched_idxs:
                        matched_idxs[idx] = [jdx]
                    else:
                        matched_idxs[idx].append(jdx)
        
                    if idx not in visited_idxs:
                        visited_idxs.append(idx)
                    
                    visited_idxs.append(jdx)
        
    return None

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for nid in chiralN:
                mol.GetAtomWithIdx(nid).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
   
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 2]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1 # Must be selected in the tree
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction
                        
    edges = [(u[0],u[1],MST_MAX_WEIGHT-v) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    
    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

#Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

#Concatenate previous nodes and all its neighbors and 
def graph_to_mol(graph):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for idx in graph.nodes:
        atom_label = graph.nodes[idx]['label']
        atom = Chem.Atom(atom_label[0])
        atom.SetFormalCharge(atom_label[1])
        atom.SetAtomMapNum(idx)
        new_mol.AddAtom(atom)
    
    for edge in graph.edges:
        beginatom_idx = edge[0]-1
        endatom_idx = edge[1]-1
        if new_mol.GetBondBetweenAtoms(beginatom_idx, endatom_idx) is not None: continue
        bond_label = BOND_LIST[graph[edge[0]][edge[1]]['label']]
        new_mol.AddBond(beginatom_idx, endatom_idx, bond_label)
    mol = new_mol.GetMol()
    mol.UpdatePropertyCache() 
    
    return mol

#Concatenate previous nodes and all its neighbors and 
def mol_to_graph(mol):
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

def attach_mol_graph(graph, mol, atom1_idxs, atom2_idxs):
    num_atoms = mol.GetNumAtoms()
    amap = {}
    for idx in range(num_atoms):
        if idx in atom2_idxs:
            amap[idx] = atom1_idxs[atom2_idxs.index(idx)] + 1
            continue
        atom = mol.GetAtomWithIdx(idx)
        atom_idx = len(graph)+1
        
        graph.add_node(atom_idx)
        graph.nodes[atom_idx]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
        graph.nodes[atom_idx]['bonds'] = []
        graph.nodes[atom_idx]['rings'] = []
        amap[idx] = atom_idx        
    
    new_mol = None
    if mol.GetAtomWithIdx(0).GetIsAromatic():
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        Chem.Kekulize(new_mol)
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        btype = BOND_LIST.index( bond.GetBondType() )
        if end_idx not in atom2_idxs or begin_idx not in atom2_idxs:
            map_begin_idx = amap[begin_idx]
            map_end_idx = amap[end_idx]
            graph.add_edge(map_begin_idx, map_end_idx)
            graph.add_edge(map_end_idx, map_begin_idx)
            graph[map_begin_idx][map_end_idx]['label'] = btype
            graph[map_begin_idx][map_end_idx]['mess_idx'] = len(graph.edges) - 1
            graph[map_end_idx][map_begin_idx]['label'] = btype
            graph[map_end_idx][map_begin_idx]['mess_idx'] = len(graph.edges)

    anchor = [amap[idx] for idx in atom2_idxs]
    clq = [amap[idx] for idx in amap]
    if 0 in clq: pdb.set_trace()  
    return graph, anchor, clq, amap
    
def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid:{} for nei in prev_nodes + neighbors}

    for nei_id,ctr_atom,nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom
    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()

def check_singleton(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() > 2]
    singletons = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() == 1]
    if len(singletons) > 0 or len(rings) == 0: return True

    n_leaf2_atoms = 0
    for atom in cand_mol.GetAtoms():
        nei_leaf_atoms = [a for a in atom.GetNeighbors() if not a.IsInRing()] #a.GetDegree() == 1]
        if len(nei_leaf_atoms) > 1: 
            n_leaf2_atoms += 1

    return n_leaf2_atoms == 0

def check_aroma(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() >= 3]
    if len(rings) < 2: return 0 #Only multi-ring system needs to be checked
    
    get_nid = lambda x: 0 if x.is_leaf else x.nid
        
    benzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.benzynes] 
    penzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.penzynes] 
    if len(benzynes) + len(penzynes) == 0: 
        return 0 #No specific aromatic rings

    n_aroma_atoms = 0
    for atom in cand_mol.GetAtoms():
        if atom.GetAtomMapNum() in benzynes+penzynes and atom.GetIsAromatic():
            n_aroma_atoms += 1

    cur_mol = attach_mols(cur_mol, children, [], global_amap) #father is already attached
    cur_mol = attach_mols(cur_mol, children, [], global_amap) #father is already attached
    for nei_node in children:
        if not nei_node.is_leaf:
            dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)
