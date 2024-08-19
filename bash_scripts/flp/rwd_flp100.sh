#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:2

#submit by $sbatch bash_scripts/flp/rwd_flp100.sh

#source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --task "rwd_flp" \
  --multigpu \
  --wandb_logger_name "flp100_rwd_lr005" \
  --flp_target_factor 1.0 \
  --project_name "RWD_FLP100_20" \
  --diffusion_type "categorical" \
  --learning_rate 0.00005 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/flp100/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/flp100_new/flp_train_subopt.txt" \
  --validation_split "/data/shared/huiyuan/flp100_new/flp_test.txt" \
  --test_split "/data/shared/huiyuan/flp100_new/flp_test.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --do_train \
  --do_val \
  #--debug \
  #--flp_sparse \
  #--separate_rwd_emb\
  #--do_test \