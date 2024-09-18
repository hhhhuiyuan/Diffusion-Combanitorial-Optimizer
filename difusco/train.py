"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_tsp_model import TSPModel
from rwd_tsp_model import Rwd_TSPModel
from pl_mis_model import MISModel
from rwd_mis_model import Rwd_MISModel
from pl_dag_model import DAGModel
from pl_addedge_dag_model import AddEdge_DAGModel
from pl_modedge_vrp_model import ModEdge_VRPModel
from pl_flp_model import FLPModel
from rwd_flp_model import Rwd_FLPModel
from pl_vrp_model import VRPModel

import numpy as np
import random


def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--seed', type=int, default=1023)
  parser.add_argument('--task', type=str, required=True)
  parser.add_argument('--subopt_data', action='store_true')

  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=64)
  parser.add_argument('--test_examples', type=int, default=320)

  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--val_batch_size', type=int, default=64)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr_scheduler', type=str, default='constant')

  parser.add_argument('--num_workers', type=int, default=1)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='gaussian')
  parser.add_argument('--diffusion_schedule', type=str, default='linear')
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='linear')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)
  parser.add_argument('--decoding_strategy', type=str, default="greedy sampling")

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default=None)
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_val', action='store_true')
  parser.add_argument('--refine', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--grad_accumulation', type=int, default=1)
  parser.add_argument('--model', type=str, default='gnn')
  parser.add_argument('--flp_target_factor', type=float, default=1.0) #set to 0 for searching over [0,1] with increment 0.1
  parser.add_argument('--dag_target_factor', type=float, default=0) #set to 0 for searching over [0,1] with increment 0.1
  parser.add_argument('--dag_decode_factor', type=float, default=0.5) #set to int specifying number of nodes to add
  parser.add_argument('--vrp_decode_factor', type=int, default=5) #set int specifying number of nodes to add
  parser.add_argument('--hetero_sizes', action='store_true')
  parser.add_argument('--multigpu', action='store_true')
  parser.add_argument('--weighted', action='store_true')
  parser.add_argument('--separate_rwd_emb', action='store_true')
  parser.add_argument('--flp_sparse', action='store_true', help='only use node feature and use sparse implementation for GNN forward')
  parser.add_argument('--mis_sparse', action='store_true', help='only use node feature and use sparse implementation for GNN forward')
  parser.add_argument('--sparse_noise', type=int, default=0, help='noise distribution has sparsity: sparse_noise * solution dim, set 0 to disable sparse noise')
  parser.add_argument('--query_factor', type=int, default=1, help='number of extra queris in training is query_factor*number_of_nodes')
  parser.add_argument('--tsp_size', type=int, default=None)
  parser.add_argument('--XE_rwd_cond', type=str, choices=['X', 'E', 'XE'], default='E')
  parser.add_argument('--guidance', type=float, default=0, help='strength of classifier free guidance, 0 if no guidance')

    
  args = parser.parse_args()
  return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
  set_seed(args.seed)
  epochs = args.num_epochs
  project_name = args.project_name

  if args.task == 'tsp':
    model_class = TSPModel
    saving_mode = 'min'
  elif args.task == 'rwd_tsp':
    model_class = Rwd_TSPModel
    saving_mode = 'min'
  elif args.task == 'vrp':
    model_class = VRPModel
    saving_mode = 'min'
  elif args.task == 'rwd_vrp':
    model_class = ModEdge_VRPModel
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'min'
  elif args.task == 'rwd_mis':
    model_class = Rwd_MISModel
    saving_mode = 'min'
  elif args.task == 'dag':
    model_class = DAGModel
    saving_mode = 'min'
  elif args.task == 'addedge_dag':
    model_class = AddEdge_DAGModel
    saving_mode = 'min'
  elif args.task == 'flp':
    model_class = FLPModel
    saving_mode = 'min'
  elif args.task == 'rwd_flp':
    model_class = Rwd_FLPModel
    saving_mode = 'min'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)

  wandb_path = os.path.join(args.storage_path, f'models')
  os.makedirs(wandb_path, exist_ok=True)
  wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
  wandb_logger = WandbLogger(
      name=args.wandb_logger_name,
      project=project_name,
      save_dir=wandb_path,
      id=args.resume_id or wandb_id,
  )
  rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

  checkpoint_callback = ModelCheckpoint(
      monitor='val/subopt_gap', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'),
  )
  lr_callback = LearningRateMonitor(logging_interval='step')


  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
      #callbacks=[TQDMProgressBar(refresh_rate=20), lr_callback],
      logger=wandb_logger,
      check_val_every_n_epoch=1,
      # dag task needs trun off static graph when use batch size=1 and grad accumulation
      # runing with one gpu do not need DDP
      strategy=DDPStrategy(static_graph=False, find_unused_parameters=True) if args.multigpu else "auto",
      precision=16 if args.fp16 else 32,
      fast_dev_run= True if args.debug else False,
      num_sanity_val_steps=0,
      # quiet val if wanted 
      limit_val_batches = None if args.do_val else 0,
      accumulate_grad_batches=args.grad_accumulation,
  )
 
  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_calfivtlback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)
  
  trainer.logger.finalize("success")


if __name__ == '__main__':
  args = arg_parser()
  main(args)
