#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4

#submit by $sbatch bash_scripts/rwd_dag34_single.sh
source activate Digress

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
#export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='disabled'

python -u difusco/train.py \
  --seed 1023 \
  --task "dag" \
  --model "gnn" \
  --project_name "RWD_DifusCO_DAG" \
  --wandb_logger_name "dag5_combo34_tar0.5+add0.2_epc300" \
  --dag_target_factor 0.5 \
  --dag_decode_factor 0.2 \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/dag_combo34/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "/data/shared/huiyuan/dag34_new/train.pkl" \
  --validation_split "/data/shared/huiyuan/dag34_new/val.pkl" \
  --test_split "/data/shared/huiyuan/dag34_new/test.pkl" \
  --batch_size 128 \
  --num_epochs 300 \
  --validation_examples 32 \
  --test_examples 320 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --decoding_strategy "greedy"\
  --refine \
  --do_train \
  --do_val \
  --debug
  #--do_test \
  
  