"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
from concorde.tsp import TSPSolver

from co_datasets.tsp_graph_dataset import Rwd_TSPGraphDataset, TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
from utils.tsp_algorithms import get_lower_adj_matrix
from utils.mis_utils import Solution_Metric
from scipy.spatial.distance import cdist
from heuristics import solve_w_heuristics


class Rwd_TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = Rwd_TSPGraphDataset(
        data_file=self.args.training_split,
        sparse_factor=self.args.sparse_factor,
    )

    self.test_dataset = TSPGraphDataset(
        data_file=self.args.test_split,
        sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = TSPGraphDataset(
        data_file=self.args.validation_split,
        sparse_factor=self.args.sparse_factor,
    )

    self.val_metrics = Solution_Metric()
    self.test_metrics = Solution_Metric()

  def forward(self, x, adj, t, reward, edge_index, node_count=None, edge_count=None):
    return self.model(x, t, reward, adj, edge_index, node_count=node_count, edge_count=edge_count)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _, time_cost = batch
      t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)

    else:
      _, graph_data, point_indicator, edge_indicator, _, time_cost = batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
    
    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
        adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)  #bs * n * n
    xt = xt + 0.05 * (2*torch.rand_like(xt)-1)
    # xt = xt * 2 - 1
    # xt = xt * (1.0 + 0.05 * torch.rand_like(xt)) #each entry in [-1,1]

    if not self.sparse:
      t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    else:
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1) 
        
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      time_cost = time_cost.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1,1)
        
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1) 
      points = points.reshape(-1, 2) 
    
    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        time_cost.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute loss
    #import pdb; pdb.set_trace()
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss, prog_bar=True, on_step=True)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss, prog_bar=True, on_step=True)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, points, xt, t, target, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          target.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_greedy_decoding(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
      tsp_solver = TSPEvaluator(np_points)
      gt_cost = tsp_solver.evaluate(np_gt_tour)
      target = torch.tensor([[gt_cost]])
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      edge_index = edge_index.reshape((2, -1))
      np_edge_index = edge_index.cpu().numpy()
      points = points.reshape((-1, 2))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      tsp_solver = TSPEvaluator(np_points)
      gt_cost = tsp_solver.evaluate(np_gt_tour)
      target = torch.tensor([[gt_cost]])

    stacked_tours = []

    # if self.args.parallel_sampling > 1:
    #   if not self.sparse:
    #     points = points.repeat(self.args.parallel_sampling, 1, 1)
     
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      # if self.args.parallel_sampling > 1:
      #   if not self.sparse:
      #     xt = xt.repeat(self.args.parallel_sampling, 1, 1)
      #   else:
      #     xt = xt.repeat(self.args.parallel_sampling, 1)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        # if self.diffusion_type == 'gaussian':
        #   xt = self.gaussian_denoise_step(
        #       points, xt, t1, device, edge_index, target_t=t2)
        # else:
        xt = self.categorical_denoise_step(
            points, xt, t1, target, device, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      #greedy sampling by adding edge to an empty graph until a tour is formed
      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      # Refine using 2-opt
      if self.args.refine:
        solved_tours, opt_iterations = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device)
      else:
        solved_tours = tours
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)
    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)
    
    metrics = {
          f"{split}/gt_cost": gt_cost,
          f"{split}/solved_cost": best_solved_cost,
          f"{split}/subopt_gap": best_solved_cost - gt_cost,
      }
    if self.args.refine:
      other_metrics = {
            f"{split}/2opt_iterations": opt_iterations,
            f"{split}/merge_iterations": merge_iterations,
        } 
    else:
      other_metrics = {
            f"{split}/merge_iterations": merge_iterations,
        } 
    for k, v in other_metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    
    return metrics
  
  def test_heuristic_decoding(self, batch, batch_idx, split='test'):
      edge_index = None
      np_edge_index = None
      device = batch[-1].device
      if not self.sparse:
        real_batch_idx, points, adj_matrix, gt_tour, _ = batch
        np_points = points.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()[0]
      # else:
      #   real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      #   route_edge_flags = graph_data.edge_attr
      #   points = graph_data.x
      #   edge_index = graph_data.edge_index
      #   num_edges = edge_index.shape[1]
      #   batch_size = point_indicator.shape[0]
      #   adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      #   points = points.reshape((-1, 2))
      #   edge_index = edge_index.reshape((2, -1))
      #   np_points = points.cpu().numpy()
      #   np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      #   np_edge_index = edge_index.cpu().numpy()
  
      stacked_tours = []
      #opt_iterations, merge_iterations = 0, 0
  
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          points = points.repeat(self.args.parallel_sampling, 1, 1)
        else:
          points = points.repeat(self.args.parallel_sampling, 1)
          edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)
  
      for _ in range(self.args.sequential_sampling):
        xt = torch.randn_like(adj_matrix.float())
        if self.args.parallel_sampling > 1:
          if not self.sparse:
            xt = xt.repeat(self.args.parallel_sampling, 1, 1)
          else:
            xt = xt.repeat(self.args.parallel_sampling, 1)
          xt = torch.randn_like(xt)
  
        if self.diffusion_type == 'gaussian':
          xt.requires_grad = True
        else:
          xt = (xt > 0).long()
  
        if self.sparse:
          xt = xt.reshape(-1)
  
        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                          T=self.diffusion.T, inference_T=steps)
  
        # Diffusion iterations
        for i in range(steps):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1]).astype(int)
          t2 = np.array([t2]).astype(int)
  
          if self.diffusion_type == 'gaussian':
            xt = self.gaussian_denoise_step(
                points, xt, t1, device, edge_index, target_t=t2)
          else:
            xt = self.categorical_denoise_step(
                points, xt, t1, device, edge_index, target_t=t2)
  
        if self.diffusion_type == 'gaussian':
          adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        else:
          adj_mat = xt.float().cpu().detach().numpy() + 1e-6
  
        if self.args.save_numpy_heatmap:
          self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)
    
        #decoding the output of diffusion model using heuristic
        '''
        adj_mat: bs x n x n numpy 
        np_points: n x 2 numpy
        n: number of nodes
        '''
        #mask some edges by change the distance to a large number, say 5
        edge_matrix = cdist(np_points, np_points, 'euclidean')
        #symmetrize and negate the output adjacency matrix
        adj_mat = np.squeeze(adj_mat, axis=0)
        adj_mat = 1 - np.maximum(adj_mat, adj_mat.T)

        #import pdb; pdb.set_trace()

        heuristic_tours, heuristic_costs = solve_w_heuristics(get_lower_adj_matrix(5 * adj_mat + edge_matrix))       
        best_tour = heuristic_tours[min(heuristic_costs)]
        stacked_tours.append(best_tour)
  
      #solved_tours = np.concatenate(stacked_tours, axis=0)
  
      tsp_solver = TSPEvaluator(np_points)
      gt_cost = tsp_solver.evaluate(np_gt_tour)
  
      total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
      all_solved_costs = [tsp_solver.evaluate(stacked_tours[i]) for i in range(total_sampling)]
      best_solved_cost = np.min(all_solved_costs)
      #import pdb; pdb.set_trace()
  
      metrics = {
          f"{split}/gt_cost": gt_cost,
          f"{split}/solved_cost": best_solved_cost,
          f"{split}/subopt_gap": best_solved_cost - gt_cost,
      }
      return metrics

  def on_test_epoch_start(self) -> None:
    self.print("Starting final test...")
    self.test_metrics.reset()

  def test_step(self, batch, batch_idx, split='test'):
    #only supports batch size of 1 currrently
    if self.args.decoding_strategy == 'greedy':
      metrics = self.test_greedy_decoding(batch, batch_idx, split=split)
    elif self.args.decoding_strategy == 'heuristic':
      metrics = self.test_heuristic_decoding(batch, batch_idx, split=split)
    
    (self.test_metrics if split=='test' else self.val_metrics).update(metrics)
    for k, v in metrics.items():
      self.log(k, v, on_step=True, sync_dist=True)
    
    return metrics

  def on_test_epoch_end(self):
    avg_gt_cost, avg_pred_cost, avg_gap = self.test_metrics.compute()
    self.print(f"--Test Avg GT Cost: {avg_gt_cost},"\
             f"--Test Avg Pred Cost: {avg_pred_cost},"\
             f"--Test Avg Gap: {avg_gap}.")

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def on_validation_epoch_start(self) -> None:
    self.print("Starting validate...")
    self.val_metrics.reset()
  
  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
  
  def on_validation_epoch_end(self) -> None:
    avg_gt_cost, avg_pred_cost, avg_gap = self.val_metrics.compute()
    self.print(f"Epoch {self.current_epoch}:"\
             f"--Val Avg GT Cost: {avg_gt_cost},"\
             f"--Val Avg Pred Cost: {avg_pred_cost},"\
             f"--Val Avg Gap: {avg_gap}.")