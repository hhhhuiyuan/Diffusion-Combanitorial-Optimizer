"""Lightning module for training the DIFUSCO MIS model."""

import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.mis_dataset import MISDataset, Rwd_MISDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np, split_sparse_matrix


class Rwd_MISModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=True)

    data_label_dir = None
    if self.args.training_split_label_dir is not None:
      data_label_dir = self.args.training_split_label_dir

    self.train_dataset = Rwd_MISDataset(
        data_file = self.args.training_split,
        data_label_dir = data_label_dir,
        sub_opt_data = self.args.subopt_data,
        weighted = self.args.weighted,
    )

    self.test_dataset = Rwd_MISDataset(
        data_file = self.args.test_split,
        weighted = self.args.weighted,
    )

    self.validation_dataset = Rwd_MISDataset(
        data_file = self.args.validation_split,
        weighted = self.args.weighted,
    )

  def forward(self, x, t, reward, edge_index, node_count=None, edge_count=None):
    return self.model(x, t, reward, edge_index=edge_index, node_count=node_count, edge_count=edge_count)

  def categorical_training_step(self, batch, batch_idx):
    if not self.args.weighted:
      _, graph_data, point_indicator, obj = batch
    else:
      _, graph_data, weights, point_indicator, obj = batch
    
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index
    edge_count = torch.bincount(graph_data.batch[graph_data.edge_index[0]])
    node_count = point_indicator[0]
    
    # apply rwd condtioning on x           
    # mis_obj = mis_obj * (1.0 + 0.0005 * torch.rand_like(mis_obj.float()))
    
    # apply rwd condtioning on e
    obj = obj * (1.0 + 0.0005 * torch.rand_like(obj.float()))
    
    if self.XE_rwd_cond == 'E':
      mis_obj = obj
      if self.guidance:
        rwd_mask = torch.bernoulli(0.1 * torch.ones_like(mis_obj).to(node_labels.device))
        mis_obj = torch.cat((mis_obj, rwd_mask), dim=1)
      mis_obj = mis_obj.repeat_interleave(edge_count, dim=0)
     
    else:
      mis_obj = obj.view(-1)
      mis_obj = mis_obj.repeat_interleave(point_indicator.reshape(-1), dim=0)
      mis_obj = mis_obj.view(-1, 1)
      if self.guidance:
        raise NotImplementedError
    
    # if self.guidance:
    #   rwd_mask = torch.bernoulli(0.1 * torch.ones_like(mis_obj).to(node_labels.device))
    #   mis_obj = torch.cat((mis_obj, rwd_mask), dim=1)
    
    # Sample from diffusion
    node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()
    node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)
    
    t = torch.from_numpy(t).long()
    t_X = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()

    xt = self.diffusion.sample(node_labels_onehot, t_X)
    xt = xt + 0.05 * (2*torch.rand_like(xt)-1)

    if self.XE_rwd_cond == 'E':
      t = t.to(node_labels.device)
      t = t.repeat_interleave(edge_count, dim=0)
    else:
      t = torch.from_numpy(t_X).float()
  
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    
    edge_index = edge_index.to(node_labels.device)
    
    if self.args.weighted:
      weights = weights.view(1, -1).squeeze(0)
      xt = torch.stack((weights, xt), dim=1)
    
    # Denoise
    if self.XE_rwd_cond == 'XE':
      x0_pred = self.forward(
          xt.float().to(node_labels.device),
          t.float().to(node_labels.device),
          mis_obj.float().to(node_labels.device),
          edge_index,
          node_count,
          edge_count,
      )
    else:
      x0_pred = self.forward(
          xt.float().to(node_labels.device),
          t.float().to(node_labels.device),
          mis_obj.float().to(node_labels.device),
          edge_index,
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
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, xt, t, device, reward, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)

      if self.guidance:
        rwd_mask = torch.zeros_like(reward).to(device)
        cond_rwd = torch.cat((reward, rwd_mask), dim=1)
        x0_pred_cond = self.forward(
            xt.float().to(device),
            t.float().to(device),
            cond_rwd.float().to(device),
            edge_index.long().to(device) if edge_index is not None else None,
        )
        rwd_mask = torch.ones_like(reward).to(device)
        uncond_rwd = torch.cat((reward, rwd_mask), dim=1)
        x0_pred_uncond = self.forward(
            xt.float().to(device),
            t.float().to(device),
            uncond_rwd.float().to(device),
            edge_index.long().to(device) if edge_index is not None else None,
        )

        x0_pred_prob_cond = x0_pred_cond.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
        x0_pred_prob_uncond = x0_pred_uncond.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
        x0_pred_prob = x0_pred_prob_cond * (x0_pred_prob_cond/x0_pred_prob_uncond) ** self.args.guidance
        x0_pred_prob = x0_pred_prob/x0_pred_prob.sum(dim=-1, keepdim=True)
      
      else:
        x0_pred = self.forward(
            xt.float().to(device),
            t.float().to(device),
            reward.float().to(device),
            edge_index.long().to(device) if edge_index is not None else None,
        )
        x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      
      if self.args.weighted:
        xt = xt[:, 1]
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
    device = batch[-1].device

    if not self.args.weighted:
      _, graph_data, size_indicator, mis_obj= batch
    else:
      _, graph_data, weights, size_indicator, mis_obj = batch
    
    node_labels = graph_data.x
    edge_index = graph_data.edge_index
    ref_objective = mis_obj
    size_indicator = size_indicator.cpu().numpy()
    split_indices = np.cumsum(size_indicator)[:-1]
    edge_count = torch.bincount(graph_data.batch[graph_data.edge_index[0]])
    
    if self.args.weighted:
      split_weights = np.split(weights.view(-1).cpu().numpy(), split_indices)

    stacked_predict_labels = []
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy()
    adj_mat = scipy.sparse.coo_matrix(
        (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    )
    split_adj_mats = split_sparse_matrix(adj_mat, size_indicator)
    
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(node_labels.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()
      xt = xt.reshape(-1)

      if self.args.parallel_sampling > 1:
        edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      target = ref_objective.float()
      
      if self.XE_rwd_cond == 'E':
        target = target.view(-1)
        target = target.repeat_interleave(edge_count, dim=0)
        target = target.view(-1, 1)

      else:
        target = target.view(-1)
        target = target.repeat_interleave(size_indicator.reshape(-1), dim=0)
        target = target.view(-1, 1)
      
      if self.args.weighted:
        weights = weights.view(1, -1).squeeze(0)
      
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

        if self.args.weighted:
          xt = torch.stack((weights, xt), dim=1)
        
        if self.diffusion_type == 'gaussian':
          raise NotImplementedError
          # xt = self.gaussian_denoise_step(
          #     xt, t1, device, edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
              xt, t1, device, target, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
      else:
        predict_labels = xt.float().cpu().detach().numpy() + 1e-6
      
      predict_labels = np.split(predict_labels, split_indices)
      stacked_predict_labels.extend(predict_labels)

    solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels, adj_mat in zip(stacked_predict_labels, split_adj_mats)]
    solved_solutions = [torch.from_numpy(solution) for solution in solved_solutions]
    
    if self.args.weighted:
      solved_costs = [torch.dot(solved_solutions[i], torch.from_numpy(split_weights[i])) for i in range(len(solved_solutions))]
    else:
      solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    
    ref_objective = torch.mean(ref_objective.float().view(-1).cpu())
    avg_solved_cost = torch.mean(torch.stack(solved_costs).float())

    metrics = {
        f"{split}/gt_cost": ref_objective,
        f"{split}/solved_cost": avg_solved_cost,
        f"{split}/subopt_gap": ref_objective - avg_solved_cost,
    }
    for k, v in metrics.items():
      self.log(k, v, on_step=True, sync_dist=True, batch_size=batch_size)
    
    return metrics

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
