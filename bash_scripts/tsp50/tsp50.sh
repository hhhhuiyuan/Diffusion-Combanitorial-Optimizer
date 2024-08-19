#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/tsp50/tsp50.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python difusco/train.py \
  --seed 1023 \
  --task "tsp" \
  --project_name "RWD_DifusCO_TSP50" \
  --wandb_logger_name "tsp50_subopt_baseline" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp50/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/tsp50/tsp50_uncond_nearopt_train.txt" \
  --validation_split "/data/shared/huiyuan/tsp50/tsp50_test.txt" \
  --test_split "/data/shared/huiyuan/tsp50/tsp50_test.txt" \
  --batch_size 128 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  #--debug
  #--refine \
  #--do_test \
  
  
  