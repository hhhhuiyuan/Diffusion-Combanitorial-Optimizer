#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --nodelist=a100-st-p4d24xlarge-3

#submit by $sbatch bash_scripts/data/flp/flp_data.sh

export GRB_LICENSE_FILE=/data/home/huiyuan23/gurobi_lic/gurobi.lic

python data/flp/generate_flp.py\
        --num_loc 100 \
        --num_fac 20 \
        --filename "/data/shared/huiyuan/flp100_new/flp_test.txt"\
        --seed 1024 \
        --num_samples 1280 \
        --label opt \
        --batch_size 1 \
