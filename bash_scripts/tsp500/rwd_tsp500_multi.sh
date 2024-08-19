#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4

#submit by $sbatch bash_scripts/tsp500/rwd_tsp500_multi.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --seed 1023 \
  --task "rwd_tsp" \
  --multigpu \
  --project_name "DifusCO_TSP500" \
  --wandb_logger_name "tsp500_rwd" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp500_rwd/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/tsp500/tsp500_train_subopt.txt" \
  --validation_split "/data/shared/huiyuan/tsp500/tsp500_test.txt" \
  --test_split "/data/shared/huiyuan/tsp500/tsp500_test.txt" \
  --sparse_factor 50 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy" \
  --do_train \
  --do_val \
  #--debug
  #--refine \
  #--do_test \
  
  
  
  