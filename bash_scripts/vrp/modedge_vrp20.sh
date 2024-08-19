#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/vrp/modedge_vrp20.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='online'

python difusco/train.py \
  --seed 1023 \
  --task "rwd_vrp" \
  --project_name "RWD_DifusCO_VRP" \
  --wandb_logger_name "vrp20_modedge1.2" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/tsp20/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/vrp/vrp20/train_mod_edge.pkl" \
  --validation_split "/data/shared/huiyuan/vrp/vrp20/test_seed1023.pkl" \
  --test_split "/data/shared/huiyuan/vrp/vrp20/test_seed1023.pkl" \
  --batch_size 128 \
  --num_epochs 100 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "heuristic"\
  --vrp_decode_factor 5 \
  --do_train \
  --do_val \
  #--debug
  #--refine \
  #--do_test \
  
  
  