#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#submit by $sbatch bash_scripts/data/tsp/tsp_data.sh

# source activate Digress

SEED=$((SEED + 1234))

# seed for test --seed 1023


python -u data/tsp/generate_tsp_data.py \
  --min_nodes 1000 \
  --max_nodes 1000 \
  --num_samples 1600 \
  --batch_size 100 \
  --solver "lkh"\
  --filename "/data/shared/huiyuan/tsp1000/tsp1000_train_$SEED.txt" \
  --seed $SEED
  

