#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp500_rwd.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='gdc_epc25'
GDC_CKPT='../outputs/tsp500_rwd/2024-09-20/07-53-11/models/tsp500_rwd_gdc1.5/xc40w0zf/checkpoints/epoch=25-step=52000.ckpt'


for guidance in $(seq 0.0001 0.0001 0.0001) $(seq 1 1 10); do
  for seed in $(seq 1023 1 1027); do
    python difusco/train.py \
      --seed $seed \
      --task "rwd_tsp" \
      --diffusion_type "categorical" \
      --storage_path "eval/tsp500/$(date +%Y-%m-%d)/$MODEL_NAME" \
      --ckpt_path $GDC_CKPT \
      --training_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
      --validation_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
      --test_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
      --sparse_factor 50 \
      --test_examples 128 \
      --val_batch_size 1 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --decoding_strategy "greedy"\
      --do_test \
      --guidance $guidance \
      #--tsp_use_edge \
  done  
done
  
  