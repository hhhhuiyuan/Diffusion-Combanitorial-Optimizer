#!/bin/bash
#run $bash bash_scripts/data/mis/mis_satlib_graph.sh
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --nodelist=della-k18g3
#SBATCH -A mengdigroup
#SBATCH -p pli


module purge
module load anaconda3/2023.9
conda activate DIFFOPT

export GRB_LICENSE_FILE=/home/huiyuan/gurobi_della_k18g3/gurobi.lic
#GRB_LICENSE_FILE=/home/huiyuan/gurobi_license/gurobi.lic

python -u data/mis-benchmark-framework/generate_satlib_graph.py \
                gendata \
                sat \
                ../data/shared/huiyuan/mis/mis_satlib/${FOLDER} \
                ../data/shared/huiyuan/mis/mis_satlib_train_kamis_${FOLDER} \
                --gen_labels \
                --label_solver gurobi \
                #--weighted \ 
