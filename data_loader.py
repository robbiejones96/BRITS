import os
import time

import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import read_json

class MySet(Dataset):
    def __init__(self, file_path, seed):
        super(MySet, self).__init__()
        self.content = read_json(file_path)
        self.dim = len(self.content[0]["forward"][0]["x_t"])

        if seed is not None:
            np.random.seed(seed)

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = self.content[idx]
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

    def get_dim():
        return self.dim

def collate_fn(batch):
    forward = [seq['forward'] for seq in batch]
    backward = [seq['backward'] for seq in batch]


    def to_tensor_dict(batch):
        x_t = torch.FloatTensor([[x['x_t'] for x in seq] for seq in batch])
        masks = torch.FloatTensor([[x['masks'] for x in seq] for seq in batch])
        deltas = torch.FloatTensor([[x['deltas'] for x in seq] for seq in batch])

        evals = torch.FloatTensor([[x['evals'] for x in seq] for seq in batch])
        eval_masks = torch.FloatTensor([[x['eval_masks'] for x in seq] for seq in batch])

        return {'x_t': x_t, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    ret_dict['is_train'] = torch.FloatTensor([seq["is_train"] for seq in batch])
    return ret_dict

def get_loader(file_path, batch_size, seed):
    """
    Reads the file storing the JSON data and loads it in batches

    :param file_path: path to the JSON file with data
    :param batch_size: int for batch size
    :param seed: random seed for reproducibility
    :returns: tuple storing (data_iter, dimension)
        * data_iter: DataLoader that we can use to iterate over the data
        * dimension: the dimension of x_t which we use to initialize the model
    """
    data_set = MySet(file_path, seed)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 0, \
                              shuffle = True, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return (data_iter, data_set.dim)
