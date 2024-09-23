#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_mis_rwd.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='gdc_fixbug'
#RWD_CKPT='../outputs/mis_er100/2024-09-10/17-10-35/models/RWD_MIS_ER100_0.15/tdecxxjr/checkpoints/epoch=49-step=20000.ckpt'
GDC_CKPT='../outputs/mis_er100/2024-09-10/16-58-26/models/RWD_MIS_ER100_0.15/akzbgd6h/checkpoints/epoch=49-step=20000.ckpt'
FIXED_GDC_CKPT="../outputs/mis_er100/2024-09-23/11-33-40/models/wmis100_rwd_bs128/luqq7gwg/checkpoints/epoch=48-step=19600.ckpt"

for guidance in $(seq 0.0001 0.0001 0.0001) $(seq 1 1 10); do
  for seed in $(seq 1023 1 1027); do
    python difusco/train.py \
      --seed $seed \
      --task "rwd_mis" \
      --diffusion_type "categorical" \
      --storage_path "eval/mis100/$(date +%Y-%m-%d)/$MODEL_NAME" \
      --ckpt_path $FIXED_GDC_CKPT \
      --training_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --validation_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --test_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
      --test_examples 1280 \
      --val_batch_size 128 \
      --inference_schedule "cosine" \
      --inference_diffusion_steps 50 \
      --decoding_strategy "greedy"\
      --weighted \
      --guidance $guidance\
      --do_test
  done
done
  
  