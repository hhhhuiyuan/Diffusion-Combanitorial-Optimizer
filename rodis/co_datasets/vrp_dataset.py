"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch
import pickle

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class VRPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor

    with open(self.data_file, 'rb') as f:
      self.raw_data = pickle.load(f)
    
    self.data_size = len(self.raw_data)
    print(f'Loaded "{data_file}" with {self.data_size} samples')

  def __len__(self):
    return self.data_size

  def get_example(self, idx):
    sample = self.raw_data[idx]
    depot_loc, node_locs, demand, capacity, tour, cost = sample

    # Extract points
    depot = np.array(depot_loc).reshape(1, -1)
    points = np.array(node_locs)
    demand = np.array(demand).reshape(-1, 1)
    
    points = np.concatenate((points, demand), axis=1)

    return depot, points, tour, cost

  def __getitem__(self, idx):
    depot, points, tour, cost = self.get_example(idx)
    if self.sparse_factor <= 0:
      adj_matrix = np.zeros((points.shape[0]+1, points.shape[0]+1))
      for i in range(len(tour)-1):
        if not tour[i] == 0 and not tour[i + 1] == 0:
            adj_matrix[tour[i], tour[i + 1]] = 1
      
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(depot).float(),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          #torch.tensor(tour).long(),
          torch.from_numpy(np.array([cost])).float(),
      )
    # else:
      # # Return a sparse graph where each node is connected to its k nearest neighbors
      # # k = self.sparse_factor
      # sparse_factor = self.sparse_factor
      # kdt = KDTree(points, leaf_size=30, metric='euclidean')
      # dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      # edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      # edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      # edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      # tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      # tour_edges[tour[:-1]] = tour[1:]
      # tour_edges = torch.from_numpy(tour_edges)
      # tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      # tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      # graph_data = GraphData(x=torch.from_numpy(points).float(),
      #                        edge_index=edge_index,
      #                        edge_attr=tour_edges)

      # point_indicator = np.array([points.shape[0]], dtype=np.int64)
      # edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      # return (
      #     torch.LongTensor(np.array([idx], dtype=np.int64)),
      #     graph_data,
      #     torch.from_numpy(point_indicator).long(),
      #     torch.from_numpy(edge_indicator).long(),
      #     torch.from_numpy(tour).long(),
      # )

class ModEdge_VRPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor

    with open(self.data_file, 'rb') as f:
      self.raw_data = pickle.load(f)
    
    self.data_size = len(self.raw_data)
    print(f'Loaded "{data_file}" with {self.data_size} samples')

  def __len__(self):
    return self.data_size

  def get_example(self, idx):
    sample = self.raw_data[idx]
    depot_loc, node_locs, demand, capacity, tour, cost, added_cords = sample

    # Extract points
    depot = np.array(depot_loc).reshape(1, -1)
    points = np.array(node_locs)
    demand = np.array(demand).reshape(-1, 1)
    
    points = np.concatenate((points, demand), axis=1)

    return depot, points, added_cords, cost

  def __getitem__(self, idx):
    depot, points, added_cords, cost = self.get_example(idx)
    if self.sparse_factor <= 0:
      adj_matrix = np.zeros((points.shape[0]+1, points.shape[0]+1))
      for i,j in added_cords:
        adj_matrix[i, j] = 1
      
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(depot).float(),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          #torch.tensor(tour).long(),
          torch.from_numpy(np.array([cost])).float(),
      )