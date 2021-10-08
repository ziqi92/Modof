import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import os, random, re
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PairTreeFolder(object):

    def __init__(self, path, vocab, avocab, batch_size, num_workers=0, shuffle=True, y_assm=True, replicate=None, add_target=False):
        self.path = path
        self.batch_size = batch_size
        self.vocab = vocab
        self.avocab = avocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle
        self.add_target = add_target

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        number_of_files = 0
        for fn in os.listdir(self.path):
            if not fn.endswith("pkl"): continue
            
            number_of_files += 1
            fn = os.path.join(self.path, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch
            
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()
            
            dataset = PairTreeDataset(batches, self.vocab, self.avocab, self.y_assm, add_target=self.add_target)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])
            
            for b in dataloader:
                yield b
            del data, batches, dataset, dataloader
        
        if number_of_files == 0: raise ValueError("The names of data files must end with 'pkl'. " + \
                                            "No such file exist in the train path")

class MolTreeFolder(object):
    
    def __init__(self, data_folder, vocab, batch_size, num_workers=1, shuffle=True, assm=True, replicate=None, prop_file=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.prop = False        

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate
        if prop_file is not None:
            self.prop_file = prop_file
            self.prop = True

    def __iter__(self):
        if self.prop:
            prop_data = np.loadtxt(self.prop_file)

        for fn in self.data_files:
            if 'tensors' not in fn: continue
            idx = int(re.split('[.|-]', fn)[1])
            
            fn = os.path.join(self.data_folder, fn)
            with open(fn,'rb') as f:
                data = pickle.load(f)
            
            if self.prop:
                prop = prop_data[20000*idx:min(20000*(idx+1), len(prop_data))]
                data = [(data[i], prop[i]) for i in range(len(data))]

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()
            if len(batches) == 0:
                continue
            
            if self.prop:
                dataset = PropDataset(batches, self.vocab, self.assm)
            else:
                dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, avocab, y_assm, add_target):
        self.data = data
        self.vocab = vocab
        self.avocab = avocab
        self.y_assm = y_assm
        self.add_target = add_target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch_data = self.data[idx]
        
        tree1_batch = [dpair[0] for dpair in batch_data]
        tree2_batch = [dpair[1] for dpair in batch_data]
        
        
        x_batch = MolTree.tensorize(tree1_batch, self.vocab, self.avocab, target=False, add_target=self.add_target)
        y_batch = MolTree.tensorize(tree2_batch, self.vocab, self.avocab, target=True, add_target=self.add_target)
        
        return x_batch, y_batch, tree1_batch, tree2_batch

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

class PropDataset(Dataset):
    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mol_trees = [data[0] for data in self.data[idx]]
        prop_data = [data[1] for data in self.data[idx]]
        return tensorize(mol_trees, self.vocab, assm=self.assm), prop_data
        
