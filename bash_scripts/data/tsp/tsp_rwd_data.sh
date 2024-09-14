#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH -A mengdigroup
#SBATCH -p pli


#submit by $sbatch bash_scripts/data/tsp/tsp_rwd_data.sh

module purge
module load anaconda3/2023.9
conda activate DIFFOPT

SEED=$((SEED + 1234))

python -m data.tsp.generate_tsp_data_rwd\
        --min_nodes 50 \
        --max_nodes 50 \
        --num_samples 1502000 \
        --batch_size 1000 \
        --filename "../data/shared/huiyuan/tsp50_large/tsp50_train_subopt_farthest.txt"\
        --seed $SEED\
        --reward_labelled\
        