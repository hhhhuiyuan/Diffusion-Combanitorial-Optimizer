"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
from concorde.tsp import TSPSolver

from co_datasets.dag_graph_dataset import DAGGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
from utils.tsp_algorithms import get_lower_adj_matrix
from utils.mis_utils import Solution_Metric
from utils.dag_utils import DAG_Evaluator

from scipy.spatial.distance import cdist
from heuristics import solve_w_heuristics


class DAGModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = DAGGraphDataset(
        data_file=self.args.training_split,
    )

    self.test_dataset = DAGGraphDataset(
        data_file=self.args.test_split,
        order_type='mat',
    )

    self.validation_dataset = DAGGraphDataset(
        data_file=self.args.validation_split,
        order_type='mat',
    )

    self.val_metrics = Solution_Metric()
    self.test_metrics = Solution_Metric()

    resource_dim = 1
    node_feature_dim = 1 + resource_dim  # (duration, resources)
    self.dag_eval = DAG_Evaluator(resource_dim, node_feature_dim)

    def print_grad(module, grad_input, grad_output):
      if grad_input is None or grad_output is None:
        print('Module:', module)
        print('Gradient input:', grad_input)
        print('Gradient output:', grad_output)

    # # Register the hook for each module in the model
    # for name, module in self.model.named_modules():
    #   module.register_full_backward_hook(print_grad)

  def categorical_training_step(self, batch, batch_idx):
    _, job_nodes, dep_adj_mats, order_mats, makespans, _ = batch
    dep_edge_mask = (dep_adj_mats == 1)
    
    t = np.random.randint(1, self.diffusion.T + 1, job_nodes.shape[0]).astype(int)
    
    # Sample from diffusion
    order_mats_onehot = F.one_hot(order_mats.long(), num_classes=2).float()
    xt = self.diffusion.sample(order_mats_onehot, t)  #bs * n * n

    xt[dep_edge_mask] = 1 
    #xt = xt * 2 - 1 #each entry in [-1,1]
    xt = xt + 0.005 * (2 * torch.rand_like(xt) - 1)

    if not self.sparse:
      t = torch.from_numpy(t).float().view(job_nodes.shape[0])
    
    device = order_mats.device
    # Denoise
    x0_pred = self.model(
        job_nodes.float().to(device),
        #dep_adj_mats.float().to(device),
        t.float().to(device),
        makespans.float().to(device),
        xt.float().to(device),
    )
    
    # Compute loss
    #import pdb; pdb.set_trace()
    
    loss_func = nn.CrossEntropyLoss()
    #a1 = x0_pred.permute(0,2,3,1)*(~dep_edge_mask).unsqueeze(-1)
    #a1 = a1.permute(0,3,1,2)
    #a2 = order_mats*(~dep_edge_mask).long()
    b1 = x0_pred.permute(0,2,3,1)[(~dep_edge_mask), :].permute(1,0)
    b2 = order_mats[~dep_edge_mask].long()
    loss = loss_func(b1.unsqueeze(0), b2.unsqueeze(0))
    self.log("train/loss", loss, prog_bar=True, on_step=True)
    return loss

  #def gaussian_training_step(self, batch, batch_idx):
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
      raise NotImplementedError("gaussian diffusion is not supported now")
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, job_nodes, xt, t, target, device, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
    
      x0_pred = self.model(
        job_nodes.float().to(device),
        #dep_adj_mats.float().to(device),
        t.float().to(device),
        target.float().to(device),
        xt.float().to(device),
    )
      
      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, job_nodes.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  #def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
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
    device = batch[-1].device
    _, job_nodes, dep_adj_mats, orders, makespans, schdulers = batch
    stacked_orders = []
    
    if self.args.parallel_sampling > 1:
      job_nodes = job_nodes.repeat(self.args.parallel_sampling, 1, 1)
      dep_adj_mats = dep_adj_mats.repeat(self.args.parallel_sampling, 1, 1, 1)
     
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(dep_adj_mats.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).float()

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      dep_edge_mask = (dep_adj_mats == 1)
      
      ref_cost = makespans.cpu().numpy()[0]

      # tuning parameter target, set to be 0.5*heuristic makespan
      target = makespans * torch.tensor(self.args.dag_target_factor).float()
      
      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)
        xt[dep_edge_mask] = 1
        xt = xt + 0.005 * (2 * torch.rand_like(xt) - 1)
        xt = self.categorical_denoise_step(
            job_nodes, xt, t1, target, device, target_t=t2)

      if self.diffusion_type == 'gaussian':
        slu_adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        slu_adj_mat = xt.float().cpu().detach().numpy() + 1e-6
      
      #extract the problem graph for evaluation
      np_job_nodes = job_nodes.cpu().numpy()[0]
      np_dep_adj = dep_adj_mats.cpu().numpy()[0]
      schduler = schdulers.cpu()[0].tolist()
      np_ref_order = orders.cpu().numpy()[0]
      ref_cost = makespans.cpu()[0]
      #assert ref_cost == self.dag_eval.evaluate(np_job_nodes, np_dep_adj, np_ref_order)

      #greedy sampling by adding edge to an empty graph until a tour is formed
      #for now, batched decoding is not supported
      slu_order, slu_cost = self.dag_eval.decode_greedy(np_job_nodes, np_dep_adj, slu_adj_mat[0], schduler, self.args.dag_decode_factor)
      stacked_orders.append(slu_order)
    
    if slu_cost:
      metrics = {
            f"{split}/ref_cost": ref_cost,
            f"{split}/solved_cost": slu_cost,
            f"{split}/cost_gap": slu_cost - ref_cost,
            f"{split}/invalid": 0,
        }
    else:
      metrics = {
            f"{split}/ref_cost": ref_cost,
            f"{split}/solved_cost": ref_cost,
            f"{split}/cost_gap": 0,
            f"{split}/invalid": 1,
        }

    return metrics
  
  # def test_heuristic_decoding(self, batch, batch_idx, split='test'):
  #     edge_index = None
  #     np_edge_index = None
  #     device = batch[-1].device
      
  #     real_batch_idx, points, adj_matrix, gt_tour, _ = batch
  #     np_points = points.cpu().numpy()[0]
  #     np_gt_tour = gt_tour.cpu().numpy()[0]
      
  #     stacked_tours = []
  #     #opt_iterations, merge_iterations = 0, 0
  
  #     if self.args.parallel_sampling > 1:
  #       if not self.sparse:
  #         points = points.repeat(self.args.parallel_sampling, 1, 1)
  #       else:
  #         points = points.repeat(self.args.parallel_sampling, 1)
  #         edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)
  
  #     for _ in range(self.args.sequential_sampling):
  #       xt = torch.randn_like(adj_matrix.float())
  #       if self.args.parallel_sampling > 1:
  #         if not self.sparse:
  #           xt = xt.repeat(self.args.parallel_sampling, 1, 1)
  #         else:
  #           xt = xt.repeat(self.args.parallel_sampling, 1)
  #         xt = torch.randn_like(xt)
  
  #       if self.diffusion_type == 'gaussian':
  #         xt.requires_grad = True
  #       else:
  #         xt = (xt > 0).long()
  
  #       if self.sparse:
  #         xt = xt.reshape(-1)
  
  #       steps = self.args.inference_diffusion_steps
  #       time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
  #                                         T=self.diffusion.T, inference_T=steps)
  
  #       # Diffusion iterations
  #       for i in range(steps):
  #         t1, t2 = time_schedule(i)
  #         t1 = np.array([t1]).astype(int)
  #         t2 = np.array([t2]).astype(int)
  
  #         if self.diffusion_type == 'gaussian':
  #           xt = self.gaussian_denoise_step(
  #               points, xt, t1, device, edge_index, target_t=t2)
  #         else:
  #           xt = self.categorical_denoise_step(
  #               points, xt, t1, device, edge_index, target_t=t2)
  
  #       if self.diffusion_type == 'gaussian':
  #         adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
  #       else:
  #         adj_mat = xt.float().cpu().detach().numpy() + 1e-6
  
  #       if self.args.save_numpy_heatmap:
  #         self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)
    
  #       #decoding the output of diffusion model using heuristic
  #       '''
  #       adj_mat: bs x n x n numpy 
  #       np_points: n x 2 numpy
  #       n: number of nodes
  #       '''
  #       #mask some edges by change the distance to a large number, say 5
  #       edge_matrix = cdist(np_points, np_points, 'euclidean')
  #       #symmetrize and negate the output adjacency matrix
  #       adj_mat = np.squeeze(adj_mat, axis=0)
  #       adj_mat = 1 - np.maximum(adj_mat, adj_mat.T)

  #       #import pdb; pdb.set_trace()
  #       heuristic_tours, heuristic_costs = solve_w_heuristics(get_lower_adj_matrix(5 * adj_mat + edge_matrix))       
  #       best_tour = heuristic_tours[min(heuristic_costs)]
  #       stacked_tours.append(best_tour)
  
  #     #solved_tours = np.concatenate(stacked_tours, axis=0)
  
  #     tsp_solver = TSPEvaluator(np_points)
  #     gt_cost = tsp_solver.evaluate(np_gt_tour)
  
  #     total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
  #     all_solved_costs = [tsp_solver.evaluate(stacked_tours[i]) for i in range(total_sampling)]
  #     best_solved_cost = np.min(all_solved_costs)
  #     #import pdb; pdb.set_trace()
  
  #     metrics = {
  #         f"{split}/gt_cost": gt_cost,
  #         f"{split}/solved_cost": best_solved_cost,
  #         f"{split}/subopt_gap": best_solved_cost - gt_cost,
  #     }
  #     return metrics

  def on_test_epoch_start(self) -> None:
    self.print("Starting final test...")
    self.test_metrics.reset()

  def test_step(self, batch, batch_idx, split='test'):
    #only supports batch size of 1 currrently
    if self.args.decoding_strategy == 'greedy':
      metrics = self.test_greedy_decoding(batch, batch_idx, split=split)
    elif self.args.decoding_strategy == 'heuristic':
      raise NotImplementedError("Heuristic decoding is not supported now")
      #metrics = self.test_heuristic_decoding(batch, batch_idx, split=split)
    
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