#!/bin/bash

N=99

for ((i=0;i<=N;i++)); do
    echo "Submitting batch $i"
    sbatch --export=ALL,SEED=$i bash_scripts/data/cvrp/vrp_data.sh
done