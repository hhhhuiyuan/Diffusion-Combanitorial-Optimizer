#run by $bash data/flp/mix_data.sh
#source activate Digress

python -u data/flp/mix_data.py \
  --subopt_file "/data/shared/huiyuan/flp/flp_train_subopt.txt" \
  --opt_file "/data/shared/huiyuan/flp/flp_train_opt.txt" \
  --new_file "/data/shared/huiyuan/flp/flp_train_subopt_mixed_0.05.txt" \
  --task flp \
  --ratio 0.05