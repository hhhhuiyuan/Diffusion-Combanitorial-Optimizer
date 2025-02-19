import numpy as np
import torch
from torchmetrics import Metric
import scipy.sparse

def split_sparse_matrix(sparse_matrix, size_indicator):
    """
    Split a sparse matrix into multiple submatrices based on size_indicator.

    Args:
        sparse_matrix (scipy.sparse.coo_matrix): The original sparse matrix.
        size_indicator (list): The list containing the number of nodes in each graph.

    Returns:
        list: A list of sparse submatrices.
    """
    submatrices = []
    start_idx = 0
    size_indicator = size_indicator.reshape(-1).tolist()
    
    for num_nodes in size_indicator:
        end_idx = start_idx + num_nodes

        # Extract the submatrix for the current graph
        mask = (sparse_matrix.row >= start_idx) & (sparse_matrix.row < end_idx)
        submatrix = scipy.sparse.coo_matrix(
            (sparse_matrix.data[mask],
             (sparse_matrix.row[mask] - start_idx, sparse_matrix.col[mask] - start_idx)),
            shape=(num_nodes, num_nodes)
        )

        submatrices.append(submatrix)
        start_idx = end_idx

    return submatrices


def mis_decode_np(predictions, adj_matrix):
  """Decode the labels to the MIS."""
  solution = np.zeros_like(predictions.astype(int))
  sorted_predict_labels = np.argsort(- predictions)
  csr_adj_matrix = adj_matrix.tocsr()

  for i in sorted_predict_labels:
    next_node = i

    if solution[next_node] == -1:
      continue

    solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
    solution[next_node] = 1

  return (solution == 1).astype(int)

class Solution_Metric(Metric):
    '''
    Metric class to unpack ground truth tour cost and predicted tour cost
    '''
    def __init__(self):
        super().__init__()
        self.add_state('total_gt_cost', default=[], dist_reduce_fx="cat")
        self.add_state('total_pred_cost', default=[], dist_reduce_fx="cat")
        self.add_state('total_subopt_gap', default=[], dist_reduce_fx="cat")

    def update(self, metrics) -> None:
        iterator = iter(metrics)
        gt_key = next(iterator)
        self.total_gt_cost.append(torch.tensor(metrics[gt_key]).cuda())
        pred_key = next(iterator)
        self.total_pred_cost.append(torch.tensor(metrics[pred_key]).cuda())
        gap_key = next(iterator)
        self.total_subopt_gap.append(torch.tensor(metrics[gap_key]).cuda())

    def compute(self):
        #import pdb; pdb.set_trace()
        # print(type(self.total_gt_cost))
        # print(len(self.total_gt_cost))
        # for tmp in self.total_gt_cost:
        #   print(tmp)

        # ddp will turn the metric list to a tensor
        if type(self.total_gt_cost) == list:
           return torch.stack(self.total_gt_cost).mean(), torch.stack(self.total_pred_cost).mean(), torch.stack(self.total_subopt_gap).mean()
        else:
          return self.total_gt_cost.mean(), self.total_pred_cost.mean(), self.total_subopt_gap.mean()
