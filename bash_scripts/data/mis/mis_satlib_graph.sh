#!/bin/bash
#run $bash bash_scripts/data/mis/generate_misgraph.sh
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=6:00:00
source activate Digress

# m_values=("m403" "m411" "m418" "m423" "m429" "m435" "m441" "m449")
# b_values=("b10" "b30" "b50" "b70" "b90")
# seed=1023

# for m in "${m_values[@]}"; do
#     for b in "${b_values[@]}"; do
    
#         folder_name="CBS_k3_n100_${m}_${b}"
#         python -u data/mis-benchmark-framework/generate_satlib_graph.py \
#                 gendata \
#                 sat \
#                 /data/shared/huiyuan/mis/mis_satlib/${folder_name} \
#                 /data/shared/huiyuan/mis/mwis_satlib_train_kamis\
#                 --gen_labels \
#                 --label_solver kamis \
#                 --weighted \
#                 --seed ${seed}
        
#         seed=$((seed+1))
#     done
# done

python -u data/mis-benchmark-framework/generate_satlib_graph.py \
        gendata \
        sat \
        /data/shared/huiyuan/mis/mis_satlib/CBS_k3_n100_m403_b10 \
        /data/shared/huiyuan/mis/mwis_satlib_train_kamis\
        --gen_labels \
        --label_solver kamis \
        --seed 1023 \
        #--weighted

