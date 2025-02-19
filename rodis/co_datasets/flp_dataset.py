"""UFLP (Ucapacitated Facility Location Problem) dataset."""

import glob
import os
import pickle5 as pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData
import networkx as nx


class Rwd_FLPDataset(torch.utils.data.Dataset):
  def __init__(self, data_file):
    self.data_file = data_file
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points, rest_line= line.split(' output ')
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
   
    # Extract solution
    solution, cost= rest_line.split(' cost ')
    solution = solution.split(' ')
    solution = [int(loc) for loc in solution]
    node_labels = np.zeros(points.shape[0])
    node_labels[solution] = 1 
    
    # Extract reward(time_cost in tsp)
    cost= np.array([float(cost)])
    
    return points, node_labels, cost


  def __getitem__(self, idx):
   
    points, node_labels, cost = self.get_example(idx)
    
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(points).float(),
        torch.from_numpy(node_labels).long(),
        torch.from_numpy(cost),
    )
