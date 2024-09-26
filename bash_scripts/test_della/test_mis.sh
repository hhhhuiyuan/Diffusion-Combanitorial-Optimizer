#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_mis.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='mix20_subopt'
OPT_CKPT='../outputs/mis_er/2024-09-04/12-13-57/models/RWD_MIS_ER100_0.15/ugiv2e8r/checkpoints/epoch=49-step=20000.ckpt'
MIX20_CKPT="../outputs/mis_er/2024-09-25/11-02-28/models/baseline_mix20/w1dq4t9e/checkpoints/epoch=35-step=14400.ckpt"

for seed in $(seq 1023 1 1027); do
  python difusco/train.py \
    --seed $seed \
    --task "mis" \
    --diffusion_type "categorical" \
    --storage_path "eval/mis100/$(date +%Y-%m-%d)/$MODEL_NAME" \
    --ckpt_path $MIX20_CKPT \
    --training_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
    --validation_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
    --test_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
    --test_examples 1280 \
    --val_batch_size 128 \
    --inference_schedule "cosine" \
    --inference_diffusion_steps 50 \
    --decoding_strategy "greedy"\
    --weighted \
    --do_test
done

  
  