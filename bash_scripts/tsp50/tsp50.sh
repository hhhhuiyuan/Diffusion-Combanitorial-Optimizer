#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -A mengdigroup
#SBATCH -p pli

#submit by $sbatch bash_scripts/tsp50/tsp50.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT_PLI

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='offline'

python difusco/train.py \
  --seed 1023 \
  --task "tsp" \
  --project_name "RWD_DifusCO_TSP50_large" \
  --wandb_logger_name "tsp50_baseline_subopt" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp50/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "../data/shared/huiyuan/tsp50_large/tsp50_train_subopt_farthest.txt" \
  --validation_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
  --test_split "../data/shared/huiyuan/tsp50/tsp50_test_1023.txt" \
  --batch_size 256 \
  --num_epochs 50 \
  --validation_examples 128 \
  --val_batch_size 256 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_train \
  --do_val \
  --multigpu \
  #--debug
  #--refine \
  #--do_test \
  
  
  