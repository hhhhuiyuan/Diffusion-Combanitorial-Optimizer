#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp100_rwd.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='gdc_epc41_useedge'
#mid
#SIZE='mid'
#GDC_CKPT="../data/shared/huiyuan/tsp100mid_ckpts/tsp100mid_gdc_useedge/checkpoints/epoch=44-step=27000.ckpt"
#GDC_CKPT="../data/shared/huiyuan/tsp100mid_ckpts/tsp100mid_gdc_ckpts/checkpoints/epoch=47-step=28800.ckpt"

#large
#SIZE='large'
#GDC_CKPT="../outputs/tsp100_rwd/2024-09-19/20-25-50/models/tsp100_rwd_gdc1.5/h8b73xjk/checkpoints/epoch=48-step=287483.ckpt"

#small
SIZE='small'
GDC_CKPT="../outputs/tsp100_rwd/2024-09-25/10-52-23/models/tsp100_rwd_gdc_useedge/xh2v92r9/checkpoints/epoch=41-step=8400.ckpt"

for guidance in $(seq 0.0001 0.0001 0.0001) $(seq 1 1 10); do
  for seed in $(seq 1023 1 1027); do
    python difusco/train.py \
      --seed $seed \
      --task "rwd_tsp" \
      --diffusion_type "categorical" \
      --storage_path "eval/tsp100_$SIZE/$(date +%Y-%m-%d)/$MODEL_NAME" \
      --ckpt_path $GDC_CKPT \
      --training_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
      --validation_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
      --test_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
      --test_examples 1280 \
      --val_batch_size 128 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --decoding_strategy "greedy"\
      --do_test \
      --guidance $guidance \
      --tsp_use_edge 
  done  
done
  
  