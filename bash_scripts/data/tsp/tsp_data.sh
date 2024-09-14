#!/bin/bash
#SBATCH --output=slurm/main_%j.out
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10
#SBATCH -A mengdigroup
#SBATCH -p pli

#submit by $sbatch bash_scripts/data/tsp/tsp_data.sh

module purge
module load anaconda3/2023.9
conda activate DIFFOPT

SEED=$((SEED + 1234))

python -m data.tsp.generate_tsp_data \
  --min_nodes 50 \
  --max_nodes 50 \
  --num_samples 150m2000 \
  --batch_size 1000 \
  --solver "concorde"\
  --filename "../data/shared/huiyuan/tsp50_large/tsp50_train_opt.txt" \
  --seed $SEED
  

