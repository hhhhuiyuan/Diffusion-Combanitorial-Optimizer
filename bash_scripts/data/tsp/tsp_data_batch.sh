#!/bin/bash

# # Set N to the number of jobs you want to submit
# N=99

# for ((i=1;i<=N;i++)); do
#     echo "Submitting batch $i"
#     sbatch --export=ALL,SEED=$i bash_scripts/data/tsp/tsp_data.sh
# done

# # Set the output file
# output_file="/data/shared/huiyuan/tsp500/tsp500_train_subopt.txt"

# # Loop over the seed values
# for ((seed=1234;seed<1334;seed++)); do
#     # Set the input file based on the seed
#     input_file="/data/shared/huiyuan/tsp500/tsp500_train_subopt_${seed}.txt"

#     # Append the contents of the input file to the output file
#     cat "$input_file" >> "$output_file"
# done

for ((i=39;i<=99;i++)); do
    scancel $((i+10142))
done