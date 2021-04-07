import os
import networkx as nx
import argparse
import numpy as np
import multiprocessing as mp
from vocab import Vocab
from collections import Counter, defaultdict

length = lambda x: sum([x[k] for k in x]) if len(x) > 0 else 0

MAXIMUM_COST = 20

def convert_nx(moltree):
    G = nx.Graph()
    graph = moltree.mol_tree
    counts_dict = dict()
    for node in graph.nodes:
        node_wid = graph.nodes[node]['label']
        G.add_node(node)
        G.nodes[node]['num'] = len(G.edges(node))
        G.nodes[node]['label'] = node_wid
    
    for node in graph.nodes:
        counts = dict()
        for _,nei in graph.edges(node):
            nei_wid = graph.nodes[nei]['label']
            
            if nei_wid not in counts:
                counts[nei_wid] = []
            tmp = []
            for _,nei_nei in graph.edges(nei):
                if nei_nei != node:
                    nei_nei_wid = graph.nodes[nei_nei]['label']
                    tmp.append(nei_nei_wid)
            counts[nei_wid].append(tmp)   
            
        counts_dict[node] = counts
        G.nodes[node]['counts'] = counts
        node_wid = graph.nodes[node]['label']
        for _, nei in graph.edges(node):
            if nei < node:
                nei_wid = graph.nodes[nei]['label']
                G.add_edge(node, nei)
                G[node][nei]['label1'] = node_wid
                G[node][nei]['label2'] = nei_wid
                G[node][nei]['counts1'] = counts_dict[node]
                G[node][nei]['counts2'] = counts_dict[nei]

    return G

def compute_path(tree1, tree2):
    g1 = convert_nx(tree1)
    g2 = convert_nx(tree2)
    g, cost = nx.optimal_edit_paths(g1, g2, node_subst_cost=node_subst_cost, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost, edge_subst_cost=edge_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost)
    path = g[0]
    #print("edit distance between %s and %s" % (tree1.smiles, tree2.smiles))
    #print_path(g1, g2, path)
    return path, g, cost

#def print_path(tree1, tree2, path):
#    for npair in path[0]:
#        if npair[0] is None:
#            print("add node %s (cost: %.4f)" % (tree2.nodes[npair[1]]['smile'], MAXIMUM_COST-1))
#        elif npair[1] is None:
#            print("delete node %s (cost: %.4f)" % (tree1.nodes[npair[0]]['smile'], MAXIMUM_COST))
#        else:
#            print("match node %s (cost: %.4f)" % (tree1.nodes[npair[0]]['smile'], node_subst_cost(tree1.nodes[npair[0]], tree2.nodes[npair[1]])))
#            
#    for epair in path[1]:
#        if epair[0] is None:
#            print("add edge (%s, %s) (cost: %.4f)" % (tree2.nodes[epair[1][0]]['smile'], tree2.nodes[epair[1][1]]['smile'], MAXIMUM_COST * 3/2 -1))
#        elif epair[1] is None:
#            print("delete edge (%s, %s) (cost: %.4f)" % (tree1.nodes[epair[0][0]]['smile'], tree1.nodes[epair[0][1]]['smile'], MAXIMUM_COST * 3/2))
#        else:
#            print("match edge (%s, %s) (%s, %s) (cost: %.4f)" % (tree1.nodes[epair[0][0]]['smile'], tree1.nodes[epair[0][1]]['smile'], tree2.nodes[epair[1][0]]['smile'], tree2.nodes[epair[1][1]]['smile'], edge_subst_cost(tree1[epair[0][0]][epair[0][1]], tree2[epair[1][0]][epair[1][1]])))
#    
def node_subst_cost(node1, node2):
    if node1['label'] != node2['label']:
        return MAXIMUM_COST*2
    else:
        count1 = node1['counts']
        count2 = node2['counts']
        cost, common = get_cost(count1, count2)
        if common == 0: return MAXIMUM_COST*2
        else: return cost

def node_del_cost(node):
    return MAXIMUM_COST * 1/2
    
def node_ins_cost(node):
    return MAXIMUM_COST * 1/2 + 1
    
def edge_subst_cost(edge1, edge2):
    counts11 = edge1['counts1']
    counts12 = edge1['counts2']
    counts21 = edge2['counts1']
    counts22 = edge2['counts2']
    if edge1['label1'] == edge1['label2'] == edge2['label1'] == edge2['label2']:
        cost11, common1 = get_cost(counts11, counts21)
        cost12, common2 = get_cost(counts12, counts22)
        
        cost21, common1 = get_cost(counts11, counts22)
        cost22, common2 = get_cost(counts12, counts21)
        return min(cost11+cost12, cost21+cost22)
    if edge1['label1'] == edge2['label1']:
        if edge1['label2'] == edge2['label2']:
            cost1, common1 = get_cost(counts11, counts21)
            cost2, common2 = get_cost(counts12, counts22)
            if common1==common2==0: return MAXIMUM_COST*4
            else: return cost1+cost2
    elif edge1['label2'] == edge2['label1']:
        if edge1['label1'] == edge2['label2']:
            cost1, common1 = get_cost(counts12, counts21)
            cost2, common2 = get_cost(counts11, counts22)
            if common1==common2==0: return MAXIMUM_COST*4
            else: return cost1+cost2
    
    return MAXIMUM_COST*4

def edge_del_cost(edge):
    return MAXIMUM_COST * 3/2

def edge_ins_cost(edge):
    return MAXIMUM_COST * 3/2 - 1

def get_cost(count1, count2):
    cost = 0
    common = 0
    total = 0
    for nei in set([*count1.keys()]+[*count2.keys()]):
        if nei in count1 and nei in count2:
            common = 1
            mat = np.ones((len(count1[nei]), len(count2[nei])))
            for i, list1 in enumerate(count1[nei]):
                for j, list2 in enumerate(count2[nei]):
                    counter1 = Counter(list1)
                    counter2 = Counter(list2)
                    sub_counter = (counter1-counter2) + (counter2-counter1)
                    tmp_cost = length(sub_counter)
                    tmp_total = length(counter1) + length(counter2)
                    if tmp_total == 0:
                        mat[i][j] = 0
                    else:
                        tmp_cost = tmp_cost / tmp_total
                        mat[i][j] = tmp_cost
            
            if mat.size == 1:
                cost += mat[0][0] * 2
                total += 4
            else:
                index = np.unravel_index(np.argsort(mat, axis=None), mat.shape)
                visited1 = []
                visited2 = []
                if len(count1[nei]) < len(count2[nei]):
                    tmp_cost = np.zeros((len(count1[nei]), 1))
                    for i, idx in enumerate(index[0]):
                        idx2 = index[1][i]
                        if idx not in visited1 and idx2 not in visited2:
                            tmp_cost[idx] = mat[idx][idx2]
                            visited1.append(idx)
                            visited2.append(idx2)
                else:
                    tmp_cost = np.zeros((len(count2[nei]), 1))
                    for i, idx2 in enumerate(index[1]):
                        idx = index[0][i]
                        if idx not in visited1 and idx2 not in visited2:
                            tmp_cost[idx2] = mat[idx][idx2]
                            visited1.append(idx)
                            visited2.append(idx2)
                tmp_cost = sum(tmp_cost) * 2 + abs(len(count2[nei]) - len(count1[nei]))
                total += len(count1[nei]) + len(count2[nei]) + 2
                cost += tmp_cost
        elif nei in count2:
            tmp_total = len(count2[nei])+1
            total += tmp_total
            cost += tmp_total
        elif nei in count1:
            tmp_total = len(count1[nei])+1
            total += tmp_total
            cost += tmp_total
            
    cost = cost / total * MAXIMUM_COST * 2
    return cost, common
