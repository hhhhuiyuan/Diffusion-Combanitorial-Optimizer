"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.vrp_dataset import ModEdge_VRPGraphDataset, VRPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from scipy.spatial.distance import cdist
from utils.vrp_utils import solve_by_heuristic, vrp_evaluate


class ModEdge_VRPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = ModEdge_VRPGraphDataset(
        data_file=self.args.training_split,
        sparse_factor=self.args.sparse_factor,
    )

    self.test_dataset = VRPGraphDataset(
        data_file=self.args.test_split,
        sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = VRPGraphDataset(
        #data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        data_file=self.args.validation_split,
        sparse_factor=self.args.sparse_factor,
    )

  def forward(self, x, adj, t, reward, edge_index):
    return self.model(x, t, reward, adj, edge_index)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    _, depot, points, adj_matrix, time_cost = batch
    
    zero_demand = torch.zeros((depot.shape[0], 1)).unsqueeze(1).to(points.device)
    depot = torch.cat((depot, zero_demand), dim=2)
    points = torch.cat((depot, points), dim=1)
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
   
    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt + 0.05 * (2*torch.rand_like(xt)-1)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        time_cost.float().to(adj_matrix.device),
        edge_index,
    )
    
    # Compute loss
    num_nodes = adj_matrix.shape[1]
    num_add_edges = adj_matrix[0].sum()
    
    pos_weight = float(num_nodes * num_nodes - num_add_edges) / num_add_edges 
    loss_func = nn.CrossEntropyLoss(weight = torch.tensor([1/(1+pos_weight), pos_weight/(1+pos_weight)]).to(adj_matrix.device))

    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss, prog_bar=True, on_step=True)
    return loss

  # def gaussian_training_step(self, batch, batch_idx):
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
      raise NotImplementedError("Gaussian diffusion is not supported for VRP")
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

      if self.args.sparse_noise:
        xt = torch.sparse_coo_tensor(edge_index, xt, size=(points.shape[0], points.shape[0])).to_dense()
        xt = xt.unsqueeze(0)
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

  # def test_greedy_decoding(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    
    else:
      if self.args.sparse_noise:
        _, points, adj_matrix, gt_tour = batch
        np_points = points.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()[0]
        bs = points.shape[0]
        num_nodes = points.shape[1]
        points = points.reshape((-1, 2))

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

    stacked_tours = []

    # if self.args.parallel_sampling > 1:
    #   if not self.sparse:
    #     points = points.repeat(self.args.parallel_sampling, 1, 1)
    #   else:
    #     points = points.repeat(self.args.parallel_sampling, 1)
    #     edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      # if self.args.parallel_sampling > 1:
      #   if not self.sparse:
      #     xt = xt.repeat(self.args.parallel_sampling, 1, 1)
      #   else:
      #     xt = xt.repeat(self.args.parallel_sampling, 1)
      
      if not self.args.sparse_noise:
        xt = torch.randn_like(xt)
        if self.diffusion_type == 'gaussian':
          xt.requires_grad = True
        else:
          xt = (xt > 0).long()
      else:
        if not self.args.tsp_size:
          raise ValueError("Sparse noise requires specifying tsp_size")
        noise_prob = 1/self.args.tsp_size * self.args.sparse_noise
        xt = torch.full_like(xt, noise_prob)
        xt = torch.bernoulli(xt)

      if self.sparse and not self.args.sparse_noise:
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
          if self.args.sparse_noise:
            nonzero_index = torch.nonzero(xt)
            nonzero_index[:, 1] += nonzero_index[:, 0] * num_nodes
            nonzero_index[:, 2] += nonzero_index[:, 0] * num_nodes
            edge_index = nonzero_index[:, 1:].t()
            xt = xt[xt != 0]

            xt = self.categorical_denoise_step(
                points, xt, t1, device, edge_index, target_t=t2, problem_size=num_nodes)
      
          else:
            xt = self.categorical_denoise_step(
                points, xt, t1, device, edge_index, target_t=t2)

      if self.args.sparse_noise:
        nonzero_index = torch.nonzero(xt)
        nonzero_index[:, 1] += nonzero_index[:, 0] * num_nodes
        nonzero_index[:, 2] += nonzero_index[:, 0] * num_nodes
        edge_index = nonzero_index[:, 1:].t()
        np_edge_index = edge_index.cpu().detach().numpy()
        xt = xt[xt != 0]

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      #greedy sampling by adding edge to an empty graph until a tour is formeds
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

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

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
      device = batch[-1].device
      
      _, depot, points, adj_matrix, gt_cost = batch
    
      zero_demand = torch.zeros((depot.shape[0], 1)).unsqueeze(1).to(points.device)
      depot = torch.cat((depot, zero_demand), dim=2)
      points = torch.cat((depot, points), dim=1)
      bs = points.shape[0]
      if bs ==1:
        target = torch.tensor([[gt_cost]])
      else:
        target = gt_cost*1.2

      stacked_tours = []
      all_costs = []
  
      if self.args.parallel_sampling > 1:
          points = points.repeat(self.args.parallel_sampling, 1, 1)
  
      for _ in range(self.args.sequential_sampling):
        xt = torch.randn_like(adj_matrix.float())
        if self.args.parallel_sampling > 1:
            xt = xt.repeat(self.args.parallel_sampling, 1, 1)
            xt = torch.randn_like(xt)
  
        if self.diffusion_type == 'gaussian':
          xt.requires_grad = True
        else:
          xt = (xt > 0).long()

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

        #decoding the output of diffusion model using heuristic
        '''
        adj_mat: bs x (n+1) x (n+1) numpy 
        np_points: (n+1) x 2 numpy
        n: number of nodes
        '''

        #No.0 node is the depot 
        edge_matrix = torch.cdist(points[:, :, :2].cpu(), points[:, :, :2].cpu()).numpy()
        demands = points[:, :, 2].cpu().numpy()
        
        #symmetrize the output solution adj matrix
        adj_mat = np.maximum(adj_mat, np.transpose(adj_mat, (0, 2, 1)))

        #decode a batch of instances
        heuristic_tours, heuristic_costs = solve_by_heuristic(edge_matrix, demands, adj_mat, num_mask = self.args.vrp_decode_factor) 
        
        # debug decoding
        # print(gt_tour)
        # opt_solution = adj_matrix.cpu().numpy()
        # heuristic_tours, heuristic_costs = solve_by_heuristic(edge_matrix, demands, opt_solution, num_mask = 0)      
      
      stacked_tours.extend(heuristic_tours)
      all_costs.extend(heuristic_costs)

      gt_cost = np.mean(gt_cost.cpu().numpy().reshape(-1)) 
      best_solved_cost = np.mean(all_costs)
    
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
      raise NotImplementedError("Greedy decoding is not supported for VRP")
      #metrics = self.test_greedy_decoding(batch, batch_idx, split=split)
    elif self.args.decoding_strategy == 'heuristic':
      metrics = self.test_heuristic_decoding(batch, batch_idx, split=split)
    
    for k, v in metrics.items():
        self.log(k, v, on_step=True, sync_dist=True)
    return metrics
  
  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
