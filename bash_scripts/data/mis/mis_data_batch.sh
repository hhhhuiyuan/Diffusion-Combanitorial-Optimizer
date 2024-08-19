#!/bin/bash

# Set N to the number of jobs you want to submit
N=18

# Submit the first 3 jobs without dependencies
for ((i=6;i<=8;i++)); do
    echo "Submitting batch $i"
    jobid=$(sbatch --parsable --export=ALL,SEED=$i bash_scripts/data/mis/mis_ER_graph.sh)
    jobids[i]=$jobid
done

# Submit the remaining jobs with dependencies
for ((i=9;i<=N;i++)); do
    echo "Submitting batch $i"
    # Calculate the index of the job that this job should depend on
    dep_index=$((i - 3))
    jobid=$(sbatch --parsable --dependency=afterany:${jobids[dep_index]} --export=ALL,SEED=$i bash_scripts/data/mis/mis_ER_graph.sh)
    jobids[i]=$jobid
done

# # Set the output file
# output_file="/data/shared/huiyuan/tsp500/tsp500_train_subopt.txt"

# # Loop over the seed values
# for ((seed=1234;seed<1334;seed++)); do
#     # Set the input file based on the seed
#     input_file="/data/shared/huiyuan/tsp500/tsp500_train_subopt_${seed}.txt"

#     # Append the contents of the input file to the output file
#     cat "$input_file" >> "$output_file"
# done

# for ((i=39;i<=99;i++)); do
#     scancel $((i+10142))
# done