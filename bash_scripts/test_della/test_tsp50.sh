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
#small
# OPT_CKPT='../outputs/tsp50/2024-09-21/22-36-47/models/tsp50_opt_baseline/8l9n2pnn/checkpoints/epoch=49-step=5000.ckpt'
# SUBOPT_CKPT='../outputs/tsp50/2024-09-21/23-54-41/models/tsp50_subopt_baseline/qbvh3fst/checkpoints/epoch=46-step=4700.ckpt'
#large
# OPT_CKPT='../outputs/tsp50/2024-09-15/22-31-24/models/RWD_DifusCO_TSP50_large/pg0u77gz/checkpoints/epoch=49-step=73300.ckpt'
# SUBOPT_CKPT='../outputs/tsp50/2024-09-16/04-17-52/models/RWD_DifusCO_TSP50_large/fiw3y75c/checkpoints/epoch=49-step=73300.ckpt'
#mid
OPT_CKPT='../data/shared/huiyuan/tsp50mid_ckpts/tsp50mid_opt_baseline_epoch=49-step=30000.ckpt'
SUBOPT_CKPT='../data/shared/huiyuan/tsp50mid_ckpts/tsp50mid_subopt_baseline_epoch=49-step=30000.ckpt'


for seed in $(seq 1023 1 1027); do
  python difusco/train.py \
    --seed $seed \
    --task "tsp" \
    --diffusion_type "categorical" \
    --storage_path "eval/tsp50_mid/$(date +%Y-%m-%d)/$MODEL_NAME" \
    --ckpt_path  $SUBOPT_CKPT \
    --training_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
    --validation_split "../data/shared/huiyuan/tsp50/tsp50_val_1235.txt" \
    --test_split "../data/shared/huiyuan/tsp50/tsp50_test_1023.txt" \
    --test_examples 1280 \
    --val_batch_size 256 \
    --inference_schedule "cosine" \
    --inference_diffusion_steps 50 \
    --decoding_strategy "greedy"\
    --do_test
done
  
  