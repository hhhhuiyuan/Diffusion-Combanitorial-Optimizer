#!/bin/bash
#run $sbatch bash_scripts/data/mis/mis_ER_graph.sh
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=9:00:00

export GRB_LICENSE_FILE=/data/home/huiyuan23/gurobi_lic/gurobi.lic
#export PATH=/data/home/$USER/miniconda3/bin:$PATH
#echo $PATH

SEED=1023
#$((SEED + 1234))

python -u data/mis-benchmark-framework/generate_satlib_graph.py gendata \
    random \
    None \
    /data/shared/huiyuan/mis700-800/ER_gurobi_val\
    --model er \
    --num_graphs 2560 \
    --min_n 700 \
    --max_n 800 \
    --er_p 0.15 \
    --gen_labels \
    --seed $SEED \
    --label_solver gurobi \
    --num_workers 32\
    #--weighted