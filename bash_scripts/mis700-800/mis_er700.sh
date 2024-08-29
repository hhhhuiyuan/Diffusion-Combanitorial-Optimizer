#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:4

#submit by $sbatch bash_scripts/mis700-800/mis_er700.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python -u difusco/train.py \
  --task "mis" \
  --multigpu \
  --wandb_logger_name "mis7-800_mis_baseline" \
  --project_name "RWD_MISER_7-800" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/mis_er7-800/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/mis700-800/ER_gurobi_train/*gpickle" \
  --validation_split "/data/shared/huiyuan/mis700-800/ER_gurobi_val/*gpickle" \
  --test_split "/data/shared/huiyuan/mis700-800/ER_gurobi_val/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --do_train \
  --do_val \
  #--debug \
  #--weighted \
  #--use_activation_checkpoint \
  #--do_test \