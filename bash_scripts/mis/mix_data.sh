#run by $bash bash_scripts/mis/mix_data.sh

python -u bash_scripts/mis/mis_mix_data.py \
  --subopt_file "../data/shared/huiyuan/mis100/weighted_ER_subbopt_train_1234/*gpickle" \
  --opt_file "../data/shared/huiyuan/mis100/weighted_ER_train_1234/*gpickle" \
  --new_file "../data/shared/huiyuan/mis100/weighted_ER_mix_0.20" \
  --task mis \
  --weighted \
  --ratio 0.20