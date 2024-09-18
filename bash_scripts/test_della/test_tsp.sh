#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='baseline_subopt'

python difusco/train.py \
  --seed 1023 \
  --task "tsp" \
  --diffusion_type "categorical" \
  --storage_path "eval/tsp50/$(date +%Y-%m-%d)/$MODEL_NAME" \
  --ckpt_path "../outputs/tsp50/2024-09-16/04-17-52/models/RWD_DifusCO_TSP50_large/fiw3y75c/checkpoints/epoch=49-step=73300.ckpt" \
  --training_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
  --validation_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
  --test_split "../data/shared/huiyuan/tsp50/tsp50_test_1023.txt" \
  --test_examples 128 \
  --val_batch_size 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --do_test \
  
  
  