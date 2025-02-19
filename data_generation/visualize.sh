#run by $bash data/visualize.sh
source activate Digress

python -u data/vis_data_subopt.py \
  --subopt_file "./data/tsp20_rwd/tsp20_train.txt" \
  --opt_file ".txt"