#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_mis_search_tar.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='search tar'
FIXED_GDC_CKPT="../outputs/mis_er100/2024-09-23/11-33-40/models/wmis100_rwd_bs128/luqq7gwg/checkpoints/epoch=48-step=19600.ckpt"

for guidance in $(seq 0.0001 0.0001 0.0001) $(seq 1 1 10); do
  for seed in $(seq 1023 1 1023); do
    python difusco/train.py \
      --seed $seed \
      --task "rwd_mis" \
      --diffusion_type "categorical" \
      --storage_path "eval/mis100/$(date +%Y-%m-%d)/$MODEL_NAME" \
      --ckpt_path $FIXED_GDC_CKPT \
      --training_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --validation_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --test_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --test_examples 256 \
      --val_batch_size 128 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --decoding_strategy "greedy"\
      --weighted \
      --guidance $guidance\
      --do_test \
      --inference_target_factor 0 
  done
done
  
  