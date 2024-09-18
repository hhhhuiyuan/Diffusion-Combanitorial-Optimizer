#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_mis.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='baseline_opt'

python difusco/train.py \
  --seed 1023 \
  --task "mis" \
  --diffusion_type "categorical" \
  --storage_path "eval/tsp50/$(date +%Y-%m-%d)/$MODEL_NAME" \
  --ckpt_path "../outputs/mis_er/2024-09-04/13-47-49/models/RWD_MIS_ER100_0.15/57nz8osc/checkpoints/epoch=49-step=20000.ckpt" \
  --training_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
  --validation_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
  --test_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
  --test_examples 128 \
  --val_batch_size 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --weighted \
  --do_test \
  
  
  