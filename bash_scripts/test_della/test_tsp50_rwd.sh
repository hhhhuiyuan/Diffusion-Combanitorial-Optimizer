#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp_rwd.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='gdc_epc49'
#small
#GDC_CKPT='../outputs/tsp50_rwd/2024-09-21/22-55-19/models/tsp50_gcd1.5_useedge/wnrv8s9f/checkpoints/epoch=44-step=4500.ckpt'
#large
GDC_CKPT='../data/shared/huiyuan/tsp50large_ckpts/tsp50large_gdc/checkpoints/last.ckpt'
#mid
#GDC_CKPT="../data/shared/huiyuan/tsp50mid_ckpts/tsp50mid_gdc_useedge/checkpoints/last.ckpt"

for guidance in $(seq 0.0001 0.0001 0.0001) $(seq 1 1 10); do
  for seed in $(seq 1023 1 1027); do
    python difusco/train.py \
      --seed $seed \
      --task "rwd_tsp" \
      --diffusion_type "categorical" \
      --storage_path "eval/tsp50_large/$(date +%Y-%m-%d)/$MODEL_NAME" \
      --ckpt_path $GDC_CKPT \
      --training_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
      --validation_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
      --test_split "../data/shared/huiyuan/tsp50/tsp50_test_1023.txt" \
      --test_examples 1280 \
      --val_batch_size 256 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --decoding_strategy "greedy"\
      --do_test \
      --guidance $guidance \
      #--tsp_use_edge \
  done  
done
  
  