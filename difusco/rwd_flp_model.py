"""Lightning module for training the DIFUSCO MIS model."""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.flp_dataset import Rwd_FLPDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
from utils.flp_utils import flp_decode, flp_evaluate


class Rwd_FLPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=True)

    self.train_dataset = Rwd_FLPDataset(
        data_file = self.args.training_split,
    )

    self.test_dataset = Rwd_FLPDataset(
        data_file = self.args.test_split,
    )

    self.validation_dataset = Rwd_FLPDataset(
        data_file = self.args.validation_split,
    )

  def forward(self, x, t, reward, adj=None, edge_index=None):
    if self.sparse:
      return self.model(x, t, reward, edge_index=edge_index)
    else:
      return self.model(x, t, reward, graph=adj, edge_index=edge_index)

  def categorical_training_step(self, batch, batch_idx):
    _, points, node_labels, cost = batch
    bs = points.shape[0]
    num_nodes = points.shape[1]
    
    if self.sparse:
      points = points.reshape(-1, 2)
      node_labels = node_labels.reshape(-1)
    
    t = np.random.randint(1, self.diffusion.T + 1, bs).astype(int)
    node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()

    if self.sparse:
      flp_obj = cost.view(-1)
      flp_obj = flp_obj.repeat_interleave(num_nodes, dim=0)
      flp_obj = flp_obj.view(-1, 1)
    else:
      flp_obj = cost
  
    # Sample from diffusion
    if self.sparse:
      node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)
      t = torch.from_numpy(t).long()
      t = t.repeat_interleave(num_nodes, dim=0).numpy()
      xt = self.diffusion.sample(node_labels_onehot, t)
    else:
      xt = self.diffusion.sample(node_labels_onehot, t).unsqueeze(2)

    xt = xt + 0.05 * (2*torch.rand_like(xt)-1)
    t = torch.from_numpy(t).float()

    if self.sparse:
      xt = xt.squeeze(-1)
      xt = torch.cat((points, xt), dim=1)
      t = t.reshape(-1)
  
      edges = torch.combinations(torch.arange(num_nodes), r=2)
      num_edges = edges.shape[0]
      edges = edges.repeat(bs, 1)
      for i in range(bs):
        edges[i*num_edges: (i+1)*num_edges] += i * num_nodes
      edge_index = edges.to(node_labels.device).permute(1, 0)
    else:
      xt = torch.cat((points, xt), dim=2)
      edges = torch.cdist(points, points)
      edge_index=None
    
    # Denoise
    if self.sparse:
      x0_pred = self.forward(
          xt.float().to(node_labels.device),
          t.float().to(node_labels.device),
          flp_obj.float().to(node_labels.device),
          edge_index = edge_index,
      )
    else:
      x0_pred = self.forward(
          xt.float().to(node_labels.device),
          t.float().to(node_labels.device),
          flp_obj.float().to(node_labels.device),
          adj = edges.float().to(node_labels.device),
          edge_index = edge_index,
      )
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, node_labels)
    self.log("train/loss", loss, prog_bar=True, sync_dist=True)
    return loss

  # def gaussian_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index
    device = node_labels.device

    # Sample from diffusion
    node_labels = node_labels.float() * 2 - 1
    node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
    node_labels = node_labels.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long()
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
    xt, epsilon = self.diffusion.sample(node_labels, t)

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(device).reshape(2, -1)
    epsilon = epsilon.reshape(-1)
    
    # Denoise
    epsilon_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      raise NotImplementedError
      #return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, xt, t, device, reward, adj= None, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      if self.sparse:
        x0_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            reward.float().to(device),
            edge_index = edge_index.long().to(device) if edge_index is not None else None,
        )
      else:
        x0_pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          reward.float().to(device),
          adj = adj.float().to(device),
          edge_index = edge_index,
      )
        
      if self.sparse:
        x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
        xt = xt[:, 2]
      else:
        x0_pred_prob = x0_pred.permute((0, 2, 1)).contiguous().softmax(dim=-1)
        xt = xt[:, :, 2]
      
      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      
      return xt

  # def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, draw=False, split='test'):
    _, points, node_labels, opt_cost = batch

    device = batch[-1].device
    bs = points.shape[0]
    num_nodes = points.shape[1]
    num_facility = sum(node_labels[0]).cpu()
  
    np_points = points.squeeze(0).cpu().numpy()
    ref_objective = opt_cost
    stacked_predict_labels = []
    
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(node_labels.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)
      xt = (xt > 0).long()

      if self.sparse:
        points = points.reshape(-1, 2) 
        edges = torch.combinations(torch.arange(num_nodes), r=2)
        num_edges = edges.shape[0]
        edges = edges.repeat(bs, 1)
        for i in range(bs):
          edges[i*num_edges: (i+1)*num_edges] += i * num_nodes
        edge_index = edges.to(node_labels.device).permute(1, 0)

        if self.args.parallel_sampling > 1:
          edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)
      
      else:
        edges = torch.cdist(points, points)
        edge_index=None
      
      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      target = ref_objective.view(-1,1) * self.args.flp_target_factor
      #target = torch.ones_like(ref_objective.view(-1,1)) * self.args.flp_target_factor
      
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)
        
        if self.sparse:
          xt = xt.reshape(-1, 1)
          xt = torch.cat((points, xt), dim=1)
        else:
          xt = xt.unsqueeze(-1)
          xt = torch.cat((points, xt), dim=2)
        
        if self.sparse:
          xt = self.categorical_denoise_step(
                xt, t1, device, target, edge_index=edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
                xt, t1, device, target, adj=edges, edge_index=edge_index, target_t=t2)

      predict_labels = xt.float().cpu().detach().numpy() + 1e-6
    
    if not self.sparse:
      predict_labels = predict_labels.squeeze(0)
    stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [flp_decode(predict_labels, num_facility) for predict_labels in splitted_predict_labels]
    solved_costs = [flp_evaluate(np_points, solution) for solution in solved_solutions]
    
    ref_objective = ref_objective.float().view(-1).cpu()
    best_solved_cost = torch.max(torch.stack(solved_costs)).float()

    metrics = {
        f"{split}/gt_cost": ref_objective,
        f"{split}/solved_cost": best_solved_cost,
        f"{split}/cost_gap": best_solved_cost - ref_objective,
    }
    for k, v in metrics.items():
      self.log(k, v, on_step=True, sync_dist=True, batch_size=batch_size)
    
    return metrics

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
