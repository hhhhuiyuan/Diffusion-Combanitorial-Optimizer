#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/dag/rwd_addedge_dag52.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --seed 1023 \
  --task "addedge_dag" \
  --model "gnn" \
  --project_name "RWD_ADDEDGE_DAG52COMBO" \
  --wandb_logger_name "dag_combo52_tar_search" \
  --dag_target_factor 0.0 \
  --dag_decode_factor 5 \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/dag_combo52/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/opt/hpcaas/.mounts/fs-08d5b24cdba02ca24/shared/huiyuan/dag_combo/dag52combo/train.pkl" \
  --validation_split "/opt/hpcaas/.mounts/fs-08d5b24cdba02ca24/shared/huiyuan/dag_combo/dag52combo/val.pkl" \
  --test_split "/opt/hpcaas/.mounts/fs-08d5b24cdba02ca24/shared/huiyuan/dag_combo/dag52combo/val.pkl" \
  --batch_size 128 \
  --num_epochs 200 \
  --validation_examples 128 \
  --test_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --refine \
  --do_train \
  --do_val \
  #--debug \
  #--do_test \
  
  