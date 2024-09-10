#!/bin/bash
#run $sbatch bash_scripts/data/mis/mis_ER_graph.sh
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH -A mengdigroup
#SBATCH -p pli

module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export GRB_LICENSE_FILE= GRB_LICENSE_FILE=/home/huiyuan/gurobi_license/gurobi.lic
SEED=1023
#$((SEED + 1234))

# python -u data/mis-benchmark-framework/generate_satlib_graph.py gendata \
#     random \
#     None \
#     ../data/shared/huiyuan/mis700-800/ER_kamis_train_$SEED\
#     --model er \
#     --num_graphs 2560 \
#     --min_n 700 \
#     --max_n 800 \
#     --er_p 0.15 \
#     --gen_labels \
#     --seed $SEED \
#     --label_solver kamis \
#     --num_workers 32 \
#     --time_limit 60 \
#     #--weighted

python -u data/mis-benchmark-framework/generate_satlib_graph.py gendata \
    random \
    None \
    ../data/shared/huiyuan/mis100/weighted_ER_val_$SEED\
    --model er \
    --num_graphs 2560 \
    --min_n 100 \
    --max_n 100 \
    --er_p 0.15 \
    --gen_labels \
    --seed $SEED \
    --label_solver gurobi \
    --num_workers 32 \
    --time_limit 60 \
    --weighted