#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#submit by $sbatch bash_scripts/data/tsp/tsp_rwd_data.sh
#source activate Digress

SEED=$((SEED + 1234))

python -m data.tsp.generate_tsp_data_rwd\
        --min_nodes 50 \
        --max_nodes 50 \
        --num_samples 12800 \
        --batch_size 128 \
        --filename "/data/shared/huiyuan/tsp50/tsp50_nearopt_train.txt"\
        --seed $SEED\
        --reward_labelled\
        --near_opt
        #--seed 1023\
        