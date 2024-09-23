#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp500.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='baseline_subopt'
OPT_CKPT="../outputs/tsp500/2024-09-19/22-25-46/models/baseline_opt/brgwrgok/checkpoints/epoch=49-step=100000.ckpt"
SUBOPT_CKPT="../data/shared/huiyuan/tsp500_ckpts/tsp500_baseline_subopt_epoch=49-step=100000.ckpt"

for seed in $(seq 1023 1 1027); do
  python difusco/train.py \
    --seed $seed \
    --task "tsp" \
    --diffusion_type "categorical" \
    --storage_path "eval/tsp500/$(date +%Y-%m-%d)/$MODEL_NAME" \
    --ckpt_path  $SUBOPT_CKPT \
    --training_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
    --validation_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
    --test_split "../data/shared/huiyuan/tsp500/tsp500_test.txt" \
    --sparse_factor 50 \
    --test_examples 128 \
    --val_batch_size 1 \
    --inference_schedule "cosine" \
    --inference_diffusion_steps 50 \
    --decoding_strategy "greedy"\
    --do_test
done
  
  