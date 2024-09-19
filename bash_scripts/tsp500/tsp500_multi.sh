#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4

#submit by $sbatch bash_scripts/tsp500/tsp500_multi.sh
export PATH=/data/home/huiyuan23/miniconda3/envs/Digress/bin:$PATH

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python difusco/train.py \
  --multigpu \
  --seed 1023 \
  --task "tsp" \
  --project_name "DifusCO_TSP500" \
  --wandb_logger_name "baseline_opt" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp500/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --ckpt_path "/data/shared/huiyuan/tsp100_ckpt/baseline_opt_epoch=49-step=293350.ckpt" \
  --resume_weight_only \
  --training_split "/data/shared/huiyuan/tsp500/tsp500_train_opt.txt" \
  --validation_split "/data/shared/huiyuan/tsp500/tsp500_test_new.txt" \
  --test_split "/data/shared/huiyuan/tsp500/tsp500_test_new.txt" \
  --sparse_factor 50 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 128 \
  --val_batch_size 1 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  #--debug \
  #--do_test \
  