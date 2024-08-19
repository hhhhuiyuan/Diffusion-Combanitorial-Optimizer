"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import pickle5 as pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData
import networkx as nx


class MISDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, data_label_dir=None):
    self.data_file = data_file
    self.file_lines = glob.glob(data_file)
    self.data_label_dir = data_label_dir
    print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    with open(self.file_lines[idx], "rb") as f:
      graph = pickle.load(f)

    num_nodes = graph.number_of_nodes()

    if self.data_label_dir is None:
      node_labels = [_[1] for _ in graph.nodes(data='label')]
      if node_labels is not None and node_labels[0] is not None:
        node_labels = np.array(node_labels, dtype=np.int64)
      else:
        node_labels = np.zeros(num_nodes, dtype=np.int64)
    else:
      base_label_file = os.path.basename(self.file_lines[idx]).replace('.gpickle', '_unweighted.result')
      node_label_file = os.path.join(self.data_label_dir, base_label_file)
      with open(node_label_file, 'r') as f:
        node_labels = [int(_) for _ in f.read().splitlines()]
      node_labels = np.array(node_labels, dtype=np.int64)
      assert node_labels.shape[0] == num_nodes

    edges = np.array(graph.edges, dtype=np.int64)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    # add self loop
    self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    edges = np.concatenate([edges, self_loop], axis=0)
    edges = edges.T

    return num_nodes, node_labels, edges

  def __getitem__(self, idx):
    num_nodes, node_labels, edge_index = self.get_example(idx)
    graph_data = GraphData(x=torch.from_numpy(node_labels),
                           edge_index=torch.from_numpy(edge_index))

    point_indicator = np.array([num_nodes], dtype=np.int64)
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        graph_data,
        torch.from_numpy(point_indicator).long(),
    )


class Rwd_MISDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, data_label_dir=None, sub_opt_data = False, weighted = False):
    self.data_file = data_file
    self.file_lines = glob.glob(data_file)
    self.data_label_dir = data_label_dir
    self.subopt_flag = sub_opt_data
    self.weighted = weighted
    print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    with open(self.file_lines[idx], "rb") as f:
      graph = pickle.load(f)

    num_nodes = graph.number_of_nodes()

    if self.data_label_dir is None:
      node_labels = [_[1] for _ in graph.nodes(data='label')]
      if node_labels is not None and node_labels[0] is not None:
        node_labels = np.array(node_labels, dtype=np.int64)
      else:
        node_labels = np.zeros(num_nodes, dtype=np.int64)
    # else:
    #   base_label_file = os.path.basename(self.file_lines[idx]).replace('.gpickle', '_unweighted.result')
    #   node_label_file = os.path.join(self.data_label_dir, base_label_file)
    #   with open(node_label_file, 'r') as f:
    #     node_labels = [int(_) for _ in f.read().splitlines()]
    #   node_labels = np.array(node_labels, dtype=np.int64)
    #   assert node_labels.shape[0] == num_nodes

    if self.subopt_flag:
      suboptimal_rate = np.random.uniform(0.2, 0.5)
      sample_size = int(suboptimal_rate * np.sum(node_labels == 1))
      indices_of_ones = np.where(node_labels == 1)[0]
      sampled_indices = np.random.choice(indices_of_ones, size=sample_size, replace=False)
      node_labels[sampled_indices] = 0

    edges = np.array(graph.edges, dtype=np.int64)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    # add self loop
    self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    edges = np.concatenate([edges, self_loop], axis=0)
    edges = edges.T

    if not self.weighted:
      mis_obj = np.array([np.sum(node_labels)], dtype=np.int64)
    else:
      chosen_nodes = np.where(node_labels == 1)[0]
      weight_mapping = nx.get_node_attributes(graph, 'weight')
      mis_obj = np.array([sum(weight_mapping[i] for i in chosen_nodes)], dtype=np.int64)
   
    if not self.weighted:
      return num_nodes, node_labels, edges, mis_obj
    else:
      weights = np.array([weight_mapping.get(i) for i in range(num_nodes)], dtype=np.int64)
      return num_nodes, node_labels, weights, edges, mis_obj

  def __getitem__(self, idx):
    if not self.weighted:
      num_nodes, node_labels, edge_index, mis_obj = self.get_example(idx)
    else:
      num_nodes, node_labels, weights, edge_index, mis_obj = self.get_example(idx)
  
    graph_data = GraphData(x=torch.from_numpy(node_labels),
                           edge_index=torch.from_numpy(edge_index))

    point_indicator = np.array([num_nodes], dtype=np.int64)
    
    if not self.weighted:
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(mis_obj),
      )
    else:
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(weights).long(),
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(mis_obj),
      )
