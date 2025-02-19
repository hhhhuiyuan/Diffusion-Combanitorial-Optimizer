#run by $bash data/flp/subopt_gap_label.sh
source activate Digress

python -u data/flp/subopt_gap_label.py \
  --subopt_file "/data/shared/huiyuan/flp/flp_train_subopt.txt" \
  --opt_file "/data/shared/huiyuan/flp/flp_train_opt.txt" \
  --new_file "/data/shared/huiyuan/flp/flp_train_subopt_gaplabel.txt" \
  --task flp