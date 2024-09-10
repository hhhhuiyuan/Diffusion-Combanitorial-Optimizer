#!/bin/bash

# # Set N to the number of jobs you want to submit
# N=63

# # Submit the first 3 jobs without dependencies
# for ((i=1;i<=N;i++)); do
#     echo "Submitting batch $i"
#     jobid=$(sbatch --parsable --export=ALL,SEED=$i bash_scripts/data/mis/mis_ER_graph.sh)
#     jobids[i]=$jobid
# done

# # for gurobi
# # Submit the remaining jobs with dependencies
# for ((i=27;i<=N;i++)); do
#     echo "Submitting batch $i"
#     # Calculate the index of the job that this job should depend on
#     dep_index=$((i - 3))
#     jobid=$(sbatch --parsable --dependency=afterany:${jobids[dep_index]} --export=ALL,SEED=$i bash_scripts/data/mis/mis_ER_graph.sh)
#     jobids[i]=$jobid
# done

# Set the output file
new_dir="../data/shared/huiyuan/mis700-800/ER_kamis_train"

# Loop over the file range
for ((i=1234;i<=1297;i++)); do
    # Set the directory based on the index
    dir="../data/shared/huiyuan/mis700-800/ER_kamis_train_${i}"

    # Check if the number of files is less than 2560
    num_files=$(ls -l "$dir"/*.gpickle | wc -l)

    echo "Number of files in $dir: $num_files"

    if (( num_files != 2560 )); then
        echo "Skipping directory $dir due to insufficient number of files"
        continue
    fi

     # Loop over all .gpickle files in the directory
    for file in "$dir"/*.gpickle; do
        # Extract the filename and extension
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        basename="${filename%.*}"

        basename=${basename#ER_700_800_0.15_}

        # Set the new filename
        new_filename="${basename}_seed${i}.${extension}"

        # Copy the file to the new directory with the new filename
        cp "$file" "$new_dir/$new_filename"
    done
done