#!/bin/bash
#run $sbatch bash_scripts/data/mis/mis_ER_graph.sh
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=9:00:00

export GRB_LICENSE_FILE= GRB_LICENSE_FILE=/home/huiyuan/gurobi_license/gurobi.lic
SEED=1023
#$((SEED + 1234))

python -u data/mis-benchmark-framework/generate_satlib_graph.py gendata \
    random \
    None \
    ../data/shared/huiyuan/mis100/ER_kamis_train\
    --model er \
    --num_graphs 2560 \
    --min_n 700 \
    --max_n 800 \
    --er_p 0.15 \
    --gen_labels \
    --seed $SEED \
    --label_solver kamis \
    --num_workers 2\
    #--weighted