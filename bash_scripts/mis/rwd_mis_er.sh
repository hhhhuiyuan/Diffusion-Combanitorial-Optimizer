#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/mis/rwd_mis_er.sh

#source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='disabled'

python -u difusco/train.py \
  --task "rwd_mis" \
  --wandb_logger_name "wmis100_rwdE_mix" \
  --project_name "RWD_MISSAT100" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/mis_er100/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/mis100/weighted_ER100_mixed_0.05/*gpickle" \
  --validation_split "/data/shared/huiyuan/mis100/weighted_ER100_test/*gpickle" \
  --test_split "/data/shared/huiyuan/mis100/weighted_ER100_test/*gpickle" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --do_train \
  --do_val \
  --weighted \
  --XE_rwd_cond E\
  #--debug \
  #--do_test \