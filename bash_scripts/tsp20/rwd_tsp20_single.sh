#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/tsp20/rwd_tsp20_single.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --seed 1023 \
  --task "rwd_tsp" \
  --project_name "RWD_DifusCO_TSP" \
  --wandb_logger_name "tsp20_rwd_scalex" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp20_rwd/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "data/tsp20_rwd/tsp20_subopt.txt" \
  --validation_split "data/tsp20/tsp20_val.txt" \
  --test_split "data/tsp20/tsp20_val.txt" \
  --batch_size 128 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  #--refine \
  #--debug
  #--do_test \
  
  
  
  