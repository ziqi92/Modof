import pickle
import rdkit
from mol_tree import PathNode, MolTreeNode
from collections import defaultdict

def sortEdge(edge):
    if edge[0] is None:
        return edge[1][0]
    
    return edge[0][0] if edge[0][0] < edge[0][1] else edge[0][1]

class SubtreeList():

    #def __init__(self, tree1, tree2, edges, node_dict1, node_dict2):
    def __init__(self, tree1, tree2, path):
        self.tree1 = tree1
        self.tree2 = tree2
        #edges.sort(key = sortEdge)
        self.path = path[0]
        
        # A list for Subtree Object
        #self.subtrees = []
        
        #self.node_dict1 = node_dict1
        #self.node_dict2 = node_dict2
        
        self.subtree_list = {}
        self.node_list = {}
        
        self.insert_edges = []
        self.delete_edges = []
        self.keep_edges = []
        
        self.node_tree_idx = {}
        
        self.get_subtree_list()
        self.set_main_subtree()
        
        self.delete_subtrees = [self.root]

    def get_subtree_list(self):
        self.node_dict = {}
        for i, np in enumerate(self.path[0]):
            # A node in tree1 can be mapped to specific node in tree2
            if np[0] is not None and np[1] is not None:
                node1 = self.tree1.nodes[np[0]-1]
                node2 = self.tree2.nodes[np[1]-1]
                node1.cnode = node2
                node1.ttype = 1
                node2.ttype = 2
                
                node1.isdelete = False
                node2.cnode = node1
                
            elif np[0] is not None:
                node1 = self.tree1.nodes[np[0]-1]
                node1.cnode = None
                node1.ttype = 1
                node1.isdelete = False
            elif np[1] is not None:
                node2 = self.tree2.nodes[np[1]-1]
                node2.cnode = None
                node2.ttype = 2
                
                
        # np denotes node pair
        node_keep_neighbors1 = defaultdict(list)
        node_delete_neighbors1 = defaultdict(list)
        node_insert_neighbors1 = defaultdict(list)
        node_keep_neighbors2 = defaultdict(list)
        node_delete_neighbors2 = defaultdict(list)
        node_insert_neighbors2 = defaultdict(list)
        
        # ep denotes edge pair
        self.path[1].sort(key = sortEdge)
        for i, ep in enumerate(self.path[1]):
            edge1 = ep[0]
            edge2 = ep[1]
            if edge1 is not None and edge2 is not None:
                node1 = self.tree1.nodes[edge1[0]-1]
                node2 = self.tree1.nodes[edge1[1]-1]
                node_keep_neighbors1[edge1[0]].append(node2)
                node_keep_neighbors1[edge1[1]].append(node1)
                
                node1 = self.tree2.nodes[edge2[0]-1]
                node2 = self.tree2.nodes[edge2[1]-1]
                node_keep_neighbors2[edge2[0]].append(node2)
                node_keep_neighbors2[edge2[1]].append(node1)
                
                self.get_subtree(node1, node2, edge2)
            
            elif edge1 is not None:
                node11 = self.tree1.nodes[edge1[0]-1]
                node12 = self.tree1.nodes[edge1[1]-1]
                
                node_delete_neighbors1[edge1[0]].append(node12)
                node_delete_neighbors1[edge1[1]].append(node11)
        
        node_subtrees = defaultdict(dict)
        for i, ep in enumerate(self.path[1]):
            if ep[1] is not None and ep[0] is None:
                edge = ep[1]
                node1 = self.tree2.nodes[edge[0]-1]
                node2 = self.tree2.nodes[edge[1]-1]
                subtree1 = None if edge[0] not in self.node_tree_idx else self.node_tree_idx[edge[0]]
                subtree2 = None if edge[1] not in self.node_tree_idx else self.node_tree_idx[edge[1]]
                node_insert_neighbors2[edge[0]].append(node2)
                node_insert_neighbors2[edge[1]].append(node1)
                
                if subtree1 is not None:
                    node_subtrees[edge[1]][node1] = self.subtree_list[subtree1]
                
                if subtree2 is not None:
                    node_subtrees[edge[0]][node2] = self.subtree_list[subtree2]
                    
        for i, node in enumerate(self.tree2.nodes):
            node.keep_neighbors = node_keep_neighbors2[i+1]
            node.insert_neighbors = node_insert_neighbors2[i+1]
            node.neighbors = node_keep_neighbors2[i+1] + node_insert_neighbors2[i+1]
            node.subtrees = node_subtrees[i+1]
            
        for i, node in enumerate(self.tree1.nodes):
            node.delete_neighbors = node_delete_neighbors1[i+1]
            node.keep_neighbors = node_keep_neighbors1[i+1]
            node.insert_neighbors = []
        
    def get_subtree(self, node1, node2, edge):
        if edge[0] in self.node_tree_idx:
            # If the first node in this edge has been added in a subtree
            subtree_root = self.node_tree_idx[edge[0]]
            self.node_tree_idx[edge[1]] = subtree_root
            self.subtree_list[subtree_root].add_node(node2)
            
            if node2.wid not in self.node_list:
                self.node_list[node2.wid] = [(subtree_root, node2)]
            else:
                self.node_list[node2.wid].append((subtree_root, node2))
            
            node2.tree_root = subtree_root
            
        elif edge[1] in self.node_tree_idx:
            # If the second node in this edge has been added in a subtree
            subtree_root = self.node_tree_idx[edge[1]]
            self.node_tree_idx[edge[0]] = subtree_root
            self.subtree_list[subtree_root].add_node(node1)
            
            if node1.wid not in self.node_list:
                self.node_list[node1.wid] = [(subtree_root, node1)]
            else:
                self.node_list[node1.wid].append((subtree_root, node1))
                
        else:
            # If no subtree exists, or the start node and end node don't belong to any subtrees, create a new subtree
            if edge[0] < edge[1]:
                subtree_node = node1
                other_node = node2
            else:
                subtree_node = node2
                other_node = node1
                
            subtree = Subtree(subtree_node)
            subtree.add_node(other_node)
            self.subtree_list[subtree_node] = subtree
            
            self.node_tree_idx[edge[0]] = subtree_node
            self.node_tree_idx[edge[1]] = subtree_node
            
            if node2.wid not in self.node_list:
                self.node_list[node2.wid] = [(subtree_node, node2)]
            else:
                self.node_list[node2.wid].append((subtree_node, node2))
                
            if node1.wid not in self.node_list:
                self.node_list[node1.wid] = [(subtree_node, node1)]
            else:
                self.node_list[node1.wid].append((subtree_node, node1))
            
    def set_main_subtree(self):
        if len(self.subtree_list) == 1:
            root = list(self.subtree_list.keys())[0]
            self.max_subtree = self.subtree_list[root]
            self.root = self.max_subtree.root
            return
        elif len(self.subtree_list) == 0:
            print("no subtree in mol %s and mol %s" % (self.tree1.smiles, self.tree2.smiles))
            raise ValueError("no subtree in mol %s and mol %s" % (self.tree1.smiles, self.tree2.smiles))
        
        max_size = 0
        root = 0
        
        for key in self.subtree_list:
            if len(self.subtree_list[key].nodes) > max_size:
                max_size = len(self.subtree_list[key].nodes)
                root = key
        
        self.max_subtree = self.subtree_list[root]
        self.root = self.subtree_list[root].root
        
    def print_path(self):
        print(self.edges)
        print(self.nodes)
        for edge in self.delete_edges:
            print("Delete Edge: (%d, %d)" % (edge[0], edge[1]))
        
        print("Get Subtree List " + str(self.subtree_list))
        root, subtree = self.set_main_subtree()
        print("From the maximum subtree rooted at %d : %s" % (root, str(subtree)))
        
        stack = [self.nodes[root]]
        visited = []
        while stack:
            node = stack[-1]
            
            if node not in visited:
                insert_neighbors = [n for n in node.insert_neighbors if n not in stack]
                for n in insert_neighbors:
                    n.fa_node = node
                    if n in node.subtrees:
                        print("Connect node %d with subtree %d at node %d" % (node.nid, node.subtrees[n], n.nid))
                    else: 
                        print("Insert node %d at node %d" % (n.nid, node.nid))
                stack.extend(insert_neighbors)
                
                neighbors = [n for n in node.neighbors if n not in stack]
                for n in neighbors:
                    n.fa_node = node
                    print("Keep node %d at node %d" % (n.nid, node.nid))                
                stack.extend(neighbors)
                
                visited.append(node)
            else:
                del stack[-1]

def get_subtree(tree, edge, x_node_vecs, x_mess_dict):
    subtree_list = {}
    node_tree_idx = {}
    node_list = {}
    # ========================= Get Subtree List ===============================
    
    tree.nodes[0].keep_neighbors = []
    for i, node in enumerate(tree.nodes[1:]):
        fa_node = node.fa_node
        node_idx = node.idx
        idx = x_mess_dict[node.fa_node.idx, node.idx]
        if not edge[idx]:
            if fa_node in node_tree_idx:
                new_node = MolTreeNode(node.smiles)
                new_node.wid = node.wid
                new_node.neighbors = [node.fa_node.cnode]
                new_node.idx = node_idx

                node.cnode = new_node
                node.fa_node.cnode.neighbors.append(new_node)
                node_tree_idx[node] = node_tree_idx[node.fa_node]

                tree_node = node_tree_idx[node.fa_node]
                subtree_list[tree_node].add_node(new_node)
            else:
                new_fa_node = MolTreeNode(node.fa_node.smiles)
                new_fa_node.wid = fa_node.wid
                new_fa_node.idx = fa_node.idx
                new_node = MolTreeNode(node.smiles)
                new_node.wid = node.wid
                new_node.idx = node_idx
                new_fa_node.neighbors = [new_node]
                new_node.neighbors = [new_fa_node]

                node.cnode = new_node
                node.fa_node.cnode = new_fa_node

                subtree_list[new_fa_node] = Subtree(new_fa_node)
                subtree_list[new_fa_node].add_node(new_node)

                node_tree_idx[fa_node] = new_fa_node
                node_tree_idx[node] = new_fa_node

                if node.fa_node.wid in node_list:
                    node_list[node.fa_node.wid].append((new_fa_node, new_fa_node))
                else:
                    node_list[node.fa_node.wid] = [(new_fa_node, new_fa_node)]

            fa_node = node_tree_idx[node]
            if node.wid in node_list:
                node_list[node.wid].append((fa_node, node))
            else:
                node_list[node.wid] = [(fa_node, node)]

    # ========================= Subtree Embedding ==============================
    max_idx, max_num = 0, 0
    if len(subtree_list) > 1:
        for idx in subtree_list:
            if len(subtree_list[idx].nodes) > max_num:
                max_num = len(subtree_list[idx].nodes)
                max_idx = idx

        max_subtree = subtree_list[max_idx]
    else:
        max_subtree = subtree_list[list(subtree_list.keys())[0]]

    for i, node in enumerate(max_subtree.nodes):
        node.idx = i
        node.nid = i

    return subtree_list, max_subtree, node_tree_idx, node_list
      
class Subtree():
    
    def __init__(self, root):
        self.root = root
        self.nodes = [root]
        
    def add_node(self, node):
        self.nodes.append(node)
        
    def set_embedding(self, node_vecs):
        self.embedding = torch.stack([node_vecs[node.nid] for node in self.nodes]).sum(dim=1)
        
if __name__ == "__main__":
    fn = "../data/logp06/tensors-0.pkl"
    with open(fn,'rb') as f:
        data = pickle.load(f)
    
    for i in range(len(data)):
        path = data[i][2]
        print(path[0])
        rev_dict, edges = convertEdge(path[0])
        slist = SubtreeList(data[i][0], data[i][1], edges, rev_dict)
        slist.get_subtree_list()
        slist.print_path()
