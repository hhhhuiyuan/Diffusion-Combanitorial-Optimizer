#!/bin/bash
#SBATCH --output=/data/home/huiyuan23/DIFUSCO/slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/tsp50_single/rwd_tsp50_single.sh

export PATH=/data/home/huiyuan23/miniconda3/envs/Digress/bin:$PATH

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='disabled'

python -u difusco/train.py \
  --seed 1023 \
  --task "rwd_tsp" \
  --project_name "RWD_DifusCO_TSP50" \
  --wandb_logger_name "tsp50_rwd_subopt_gdc0" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp50_rwd/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/tsp50_mid/tsp50_train_subopt_farthest.txt" \
  --validation_split "/data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
  --test_split "/data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
  --batch_size 128 \
  --num_epochs 50 \
  --validation_examples 128 \
  --val_batch_size 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  --guidance 0\
  #--debug
  #--refine \
  #--do_test \
  
  
  
  