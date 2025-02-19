"""DAG Scheculing Problem Graph Dataset"""

import numpy as np
import torch
import pickle
import networkx as nx

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class DAGGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, rwd_label=True, order_type='mat'):
    self.data_file = data_file
    self.rwd_label = rwd_label
    self.order_type = order_type
    with open(self.data_file, 'rb') as f:
      self.raw_data = pickle.load(f)
    self.data_size = len(self.raw_data)
    print(f'Loaded "{data_file}" with {self.data_size} graphs')

  def __len__(self):
    return self.data_size

  def get_example(self, idx):
    # Select sample
    graph = self.raw_data[idx]

    # Extract nodes and enges
    job_nodes = [data['features'] for node, data in graph.nodes(data=True)]
    np_job_nodes = np.array(job_nodes)
    dependency_adj = nx.to_numpy_matrix(graph)
    
    # Extract schedule order and makespan if rwd_label=True
    if self.rwd_label:
      order = graph.graph['order']
      makespan = graph.graph['makespan']
      scheduler = graph.graph['scheduler']
      #print(graph.edges, order)
      return np_job_nodes, dependency_adj, order, makespan, scheduler
    
    else:
      return np_job_nodes, dependency_adj

  def __getitem__(self, idx):
    if self.rwd_label:
      np_job_nodes, dependency_adj, order, makespan, job_scheduler = self.get_example(idx)
      
      if self.order_type == 'mat':
        order_mat = np.zeros((np_job_nodes.shape[0], np_job_nodes.shape[0]))
        for i in range(len(order) - 1):
          order_mat[order[i + 1], order[i]] = 1
      else:
        order_mat = np.array(order)

      if job_scheduler == 'shortest_first':
        scheduler = [1,0,0]
      elif job_scheduler == 'critical_path':
        scheduler = [0,1,0]
      else:
        scheduler = [0,0,1]
      #resize all graph to a same size
      return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(np_job_nodes).float(),
        torch.from_numpy(dependency_adj).float(),
        torch.from_numpy(order_mat).long(),
        torch.from_numpy(np.array([makespan])).float(),
        torch.tensor(scheduler).long(),
    )
    else:
      np_job_nodes, dependency_adj = self.get_example(idx)
      return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(np_job_nodes).float(),
        torch.from_numpy(dependency_adj).float(),
    )


class AddEdge_DAGGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, probelm_only=False, split='train'):
    self.data_file = data_file
    self.probelm_only = probelm_only
    self.split = split
    with open(self.data_file, 'rb') as f:
      self.raw_data = pickle.load(f)
    self.data_size = len(self.raw_data)
    print(f'Loaded "{data_file}" with {self.data_size} graphs')

  def __len__(self):
    return self.data_size

  def get_example(self, idx):
    # Select sample
    graph = self.raw_data[idx]

    # Extract nodes and edges
    job_nodes = [data['features'] for node, data in graph.nodes(data=True)]
    np_job_nodes = np.array(job_nodes)
    dependency_adj = nx.to_numpy_matrix(graph)
    
    # Extract schedule order and makespan if probelm_only=False
    if self.probelm_only:
      return np_job_nodes, dependency_adj
       
    else:
      order = graph.graph['order']
      makespan = graph.graph['makespan']
      scheduler = graph.graph['scheduler']
      if self.split == 'train':
        add_edges = graph.graph['added_edge']
        return np_job_nodes, dependency_adj, add_edges, order, makespan, scheduler
      else:
        return np_job_nodes, dependency_adj, order, makespan, scheduler

  def __getitem__(self, idx):
    if self.probelm_only:
      np_job_nodes, dependency_adj = self.get_example(idx)
      return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(np_job_nodes).float(),
        torch.from_numpy(dependency_adj).float(),
      )
      
    else:
      if self.split == 'train':
        np_job_nodes, dependency_adj, add_edges, order, makespan, job_scheduler = self.get_example(idx)
        order_mat = np.zeros((np_job_nodes.shape[0], np_job_nodes.shape[0]))
        for i, j in add_edges:
          order_mat[i, j] = 1
      else:
        np_job_nodes, dependency_adj, order, makespan, job_scheduler = self.get_example(idx)
        order_mat = np.array(order)
   
      if job_scheduler == 'shortest_first':
        scheduler = [1,0,0]
      elif job_scheduler == 'critical_path':
        scheduler = [0,1,0]
      else:
        scheduler = [0,0,1]
      
      #resize all graph to a same size
      return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(np_job_nodes).float(),
        torch.from_numpy(dependency_adj).float(),
        torch.from_numpy(order_mat).long(),
        torch.from_numpy(np.array([makespan])).float(),
        torch.tensor(scheduler).long(),
    )