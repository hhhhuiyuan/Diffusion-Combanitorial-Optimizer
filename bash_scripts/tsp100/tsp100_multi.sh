#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4

#submit by $sbatch bash_scripts/tsp100_multi.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python difusco/train.py \
  --seed 1023 \
  --task "tsp" \
  --project_name "DifusCO_TSP100" \
  --wandb_logger_name "baseline_multi" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp100/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/tsp100/tsp100_solver_train.txt" \
  --validation_split "/data/shared/huiyuan/tsp100/tsp100_solver_val.txt" \
  --test_split "/data/shared/huiyuan/tsp100/tsp100_solver_test.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 32 \
  --test_examples 32 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  --do_test \
  #--debug
  
  