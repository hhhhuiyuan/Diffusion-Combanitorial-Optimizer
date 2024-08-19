#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32

#submit by $sbatch bash_scripts/data/cvrp/vrp_data.sh
SEED=$((SEED + 1234))

python data/attention-learn-to-route/generate_vrp_data.py\
        --data_dir "/data/shared/huiyuan/vrp"\
        --name test \
        --graph_sizes 20 \
        --problem vrp \
        --method lkh \
        --dataset_size 128 \
        --seed 1023 \
        -f