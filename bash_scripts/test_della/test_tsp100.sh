#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/test_della/test_tsp100.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE='disabled'

MODEL_NAME='baseline_subopt'
#mid
#OPT_CKPT='../data/shared/huiyuan/tsp100mid_ckpts/tsp100mid_baseline_opt_epoch=49-step=30000.ckpt'
#SUBOPT_CKPT='../data/shared/huiyuan/tsp100mid_ckpts/tsp100mid_baseline_subopt_epoch=49-step=30000.ckpt'

#large
OPT_CKPT="../outputs/tsp100/2024-09-17/08-21-49/models/DifusCO_TSP100/4397wsrp/checkpoints/epoch=49-step=293350.ckpt"
SUBOPT_CKPT="../outputs/tsp100/2024-09-17/08-53-43/models/DifusCO_TSP100/wv44tww4/checkpoints/epoch=49-step=293350.ckpt"

for seed in $(seq 1023 1 1027); do
  python difusco/train.py \
    --seed $seed \
    --task "tsp" \
    --diffusion_type "categorical" \
    --storage_path "eval/tsp100_large/$(date +%Y-%m-%d)/$MODEL_NAME" \
    --ckpt_path  $SUBOPT_CKPT \
    --training_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
    --validation_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
    --test_split "../data/shared/huiyuan/tsp100/tsp100_test_1023.txt" \
    --test_examples 1280 \
    --val_batch_size 128 \
    --inference_schedule "cosine" \
    --inference_diffusion_steps 50 \
    --decoding_strategy "greedy"\
    --do_test
done
  
  