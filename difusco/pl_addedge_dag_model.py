"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
from concorde.tsp import TSPSolver
import wandb

from co_datasets.dag_graph_dataset import DAGGraphDataset, AddEdge_DAGGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
from utils.tsp_algorithms import get_lower_adj_matrix
from utils.mis_utils import Solution_Metric
from utils.dag_utils import DAG_Evaluator

from scipy.spatial.distance import cdist
from heuristics import solve_w_heuristics


class AddEdge_DAGModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super().__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = AddEdge_DAGGraphDataset(
        data_file=self.args.training_split,
        split='train'
    )

    self.test_dataset = AddEdge_DAGGraphDataset(
        data_file=self.args.test_split,
        split='val'
    )

    self.validation_dataset = AddEdge_DAGGraphDataset(
        data_file=self.args.validation_split,
        split='test'
    )

    self.val_metrics = Solution_Metric()
    self.test_metrics = Solution_Metric()

    resource_dim = 1
    node_feature_dim = 1 + resource_dim  # (duration, resources)
    self.dag_eval = DAG_Evaluator(resource_dim, node_feature_dim)

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
    
    if self.guidance:
      mask_ratio = self.args.held_out_ratio
      rwd_mask = torch.bernoulli(mask_ratio * torch.ones_like(makespans).to(device))
      makespans = torch.cat((makespans, rwd_mask), dim=1)
    
    # Denoise
    x0_pred = self.model(
        job_nodes.float().to(device),
        t.float().to(device),
        makespans.float().to(device),
        xt.float().to(device),
    )
    
    num_nodes = order_mats.shape[1]
    num_add_edges = order_mats[0].sum()

    pos_weight = float(num_nodes * num_nodes - num_add_edges) / num_add_edges 
    loss_func = nn.CrossEntropyLoss(weight = torch.tensor([1/(1+pos_weight), pos_weight/(1+pos_weight)]).to(device))
    
    #a1 = x0_pred.permute(0,2,3,1)*(~dep_edge_mask).unsqueeze(-1)
    #a1 = a1.permute(0,3,1,2)
    #a2 = order_mats*(~dep_edge_mask).long()
    #b1 = x0_pred.permute(0,2,3,1)[(~dep_edge_mask), :].permute(1,0)
    #b2 = order_mats[~dep_edge_mask].long()
    #loss = loss_func(b1.unsqueeze(0), b2.unsqueeze(0))
    b1 = x0_pred
    b2 = order_mats.long()
    loss = loss_func(b1, b2)
    self.log("train/loss", loss, prog_bar=True, on_step=True)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      raise NotImplementedError("gaussian diffusion is not supported now")
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, job_nodes, xt, t, target, device, target_t=None, dep_mask= None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      t_batched = t.repeat(job_nodes.shape[0])

      xt[dep_mask] = 1
      
      if self.guidance:
        rwd_mask = torch.zeros_like(target).to(device)
        cond_rwd = torch.cat((target, rwd_mask), dim=1)
        x0_pred_cond = self.model(
          job_nodes.float().to(device),
          #dep_adj_mats.float().to(device),
          t_batched.float().to(device),
          cond_rwd.float().to(device),
          xt.float().to(device),
        )
        rwd_mask = torch.ones_like(target).to(device)
        uncond_rwd = torch.cat((target, rwd_mask), dim=1)
        x0_pred_uncond = self.model(
          job_nodes.float().to(device),
          #dep_adj_mats.float().to(device),
          t_batched.float().to(device),
          uncond_rwd.float().to(device),
          xt.float().to(device),
        )
        
        x0_pred_prob_cond = x0_pred_cond.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        x0_pred_prob_uncond = x0_pred_uncond.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        x0_pred_prob = x0_pred_prob_cond * (x0_pred_prob_cond/x0_pred_prob_uncond) ** self.args.guidance
        x0_pred_prob = x0_pred_prob/x0_pred_prob.sum(dim=-1, keepdim=True)        
      
      else:
        x0_pred = self.model(
          job_nodes.float().to(device),
          #dep_adj_mats.float().to(device),
          t_batched.float().to(device),
          target.float().to(device),
          xt.float().to(device),
        )
        if not self.sparse:
          x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
          x0_pred_prob = x0_pred.reshape((1, job_nodes.shape[0], -1, 2)).softmax(dim=-1)

      # xt_list = []
      # for i in range(job_nodes.shape[0]):
      #   xt_list.append(self.categorical_posterior(target_t, t, x0_pred_prob[i], xt[i]))
      # xt = torch.stack(xt_list) 
      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)   
      return xt
    

  def test_greedy_decoding(self, batch, batch_idx, split='test'):
    device = batch[-1].device
    _, job_nodes, dep_adj_mats, orders, makespans, schdulers = batch
    
    #extract the problem graph for evaluation
    np_job_nodes = job_nodes.cpu().numpy()
    np_dep_adj = dep_adj_mats.cpu().numpy()
    schduler = schdulers.cpu().tolist() 
    np_ref_order = orders.cpu().numpy()

    slu_cost_dict = {}
    opt_gap_dict = {}
    
    if self.args.dag_target_factor:
      # tuning parameter target, set to be self.args.dag_target_factor * heuristic makespan
      target_factors = [self.args.dag_target_factor]   
    else:
      # search tuning parameter target from 0.1-0.8 incremented by 0.1
      target_factors = [round(0.1 * float(i), 2) for i in range(6, 10, 1)]
        
    for target_fac in target_factors:
      target = torch.ones_like(makespans) * torch.tensor(target_fac).float().to(device)
          
      xt = torch.randn_like(dep_adj_mats.float())
      
      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).float()

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      dep_edge_mask = (dep_adj_mats == 1)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)
        #xt = xt + 0.005 * (2 * torch.rand_like(xt) - 1)
        xt = self.categorical_denoise_step(job_nodes, xt, t1, target, device, target_t=t2, dep_mask=dep_edge_mask)

      if self.diffusion_type == 'gaussian':
        slu_adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        slu_adj_mat = xt.float().cpu().detach().numpy() + 1e-6  
    
      slu_mat = slu_adj_mat
      all_slu_order, slu_cost, ref_cost = self.dag_eval.batch_decode_greedy(np_job_nodes, np_dep_adj, slu_mat, schduler, self.args.dag_decode_factor)
      slu_cost_dict[target_fac] = slu_cost  
      opt_gap_dict[target_fac] = slu_cost - ref_cost 
    
    subopt_gap = min(opt_gap for opt_gap in opt_gap_dict.values())
    metrics = {
            f"{split}/ref_cost": ref_cost,
            f"{split}/solved_cost": slu_cost_dict,
            f"{split}/cost_gap": opt_gap_dict,
            f"{split}/subopt_gap": subopt_gap
          }
    
    return metrics
  
  def on_test_epoch_start(self) -> None:
    self.print("Starting final test...")
    self.test_metrics.reset()

  def test_step(self, batch, batch_idx, split='test'):
    # only supports batch size of 1 currrently
    if self.args.decoding_strategy == 'heuristic':
      metrics = self.test_greedy_decoding(batch, batch_idx, split=split)
    elif self.args.decoding_strategy == 'greedy':
      raise NotImplementedError("Greedy decoding is not supported now")
    
    #(self.test_metrics if split=='test' else self.val_metrics).update(metrics)
    for k, v in metrics.items():
      if type(v) == dict:
        for tar, v_tar in v.items():
          self.log(f"{k}_{tar}", v_tar, sync_dist=True)
      else:
        self.log(k, v, sync_dist=True)
      # wandb.log({k: v})
    return

  # def on_test_epoch_end(self):
  #   avg_gt_cost, avg_pred_cost, avg_gap = self.test_metrics.compute()
  #   self.print(f"--Test Avg GT Cost: {avg_gt_cost},"\
  #            f"--Test Avg Pred Cost: {avg_pred_cost},"\
  #            f"--Test Avg Gap: {avg_gap}.")

  def on_validation_epoch_start(self) -> None:
    self.print("Starting validate...")
    self.val_metrics.reset()
  
  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
  
  # def on_validation_epoch_end(self) -> None:
  #   avg_gt_cost, avg_pred_cost, avg_gap = self.val_metrics.compute()
  #   self.print(f"Epoch {self.current_epoch}:"\
  #            f"--Val Avg GT Cost: {avg_gt_cost},"\
  #            f"--Val Avg Pred Cost: {avg_pred_cost},"\
  #            f"--Val Avg Gap: {avg_gap}.")