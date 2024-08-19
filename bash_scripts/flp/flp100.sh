#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2

#submit by $sbatch bash_scripts/flp/flp100.sh

#source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --task "flp" \
  --multigpu \
  --wandb_logger_name "flp100_opt_baseline" \
  --project_name "RWD_FLP100_20" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/flp100/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/flp100_new/flp_train_opt.txt" \
  --validation_split "/data/shared/huiyuan/flp100_new/flp_test.txt" \
  --test_split "/data/shared/huiyuan/flp100_new/flp_test.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --do_train \
  --do_val \
  #--debug
  #--do_test \