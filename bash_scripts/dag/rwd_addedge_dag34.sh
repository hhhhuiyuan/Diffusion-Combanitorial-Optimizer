#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
##SBATCH -A mengdigroup
##SBATCH -p pli

#submit by $sbatch bash_scripts/dag/rwd_addedge_dag34.sh

module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE="offline"

python -u difusco/train.py \
  --seed 1023 \
  --task "addedge_dag" \
  --model "gnn" \
  --project_name "RWD_ADDEDGE_DAG34COMBO" \
  --wandb_logger_name "dag_combo34_tar_search" \
  --dag_target_factor 0 \
  --dag_decode_factor 5 \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/dag_combo34/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "../data/shared/huiyuan/dag_combo/dag34combo/train.pkl" \
  --validation_split "../data/shared/huiyuan/dag_combo/dag34combo/val.pkl" \
  --test_split "../data/shared/huiyuan/dag_combo/dag34combo/val.pkl" \
  --batch_size 128 \
  --num_epochs 100 \
  --validation_examples 128 \
  --test_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "heuristic"\
  --do_train \
  --do_val \
  #--debug\
  #--do_test \
  
  