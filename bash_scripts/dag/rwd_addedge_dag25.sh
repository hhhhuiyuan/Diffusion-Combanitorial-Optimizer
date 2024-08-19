#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/rwd_addedge_dag25.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='disabled'

python -u difusco/train.py \
  --seed 1023 \
  --task "addedge_dag" \
  --hetero_sizes \
  --num_workers 0 \
  --model "gnn" \
  --project_name "RWD_ADDEDGE_DAG25" \
  --wandb_logger_name "dag_tar0.8_add5" \
  --dag_target_factor 0.8 \
  --dag_decode_factor 5 \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/dag25/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/dag25_addedge/train.pkl" \
  --validation_split "/data/shared/huiyuan/dag25_addedge/val.pkl" \
  --test_split "/data/shared/huiyuan/dag25_addedge/val.pkl" \
  --batch_size 4 \
  --grad_accumulation 32 \
  --num_epochs 100 \
  --validation_examples 128 \
  --test_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --refine \
  --do_train \
  --do_val \
  #--debug
  #--do_test \
  
  