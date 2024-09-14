
#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

#submit by $sbatch bash_scripts/mis/mis_er.sh
module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"
export WANDB_MODE='offline'

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "wmis100_mis_baseline" \
  --project_name "RWD_MIS_ER100_0.15" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "../outputs/mis_er/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  --training_split "../data/shared/huiyuan/mis100/weighted_ER_train_1234/*gpickle" \
  --validation_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
  --test_split "../data/shared/huiyuan/mis100/weighted_ER_val_1023/*gpickle" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --do_train \
  --do_val \
  --weighted \
  #--debug \
  #--use_activation_checkpoint \
  #--do_test \
