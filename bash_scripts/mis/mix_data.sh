#run by $bash bash_scripts/mis/mix_data.sh

python -u bash_scripts/mis/mis_mix_data.py \
  --subopt_file "/data/shared/huiyuan/mis100/weighted_ER100_subopt_train/*gpickle" \
  --opt_file "/data/shared/huiyuan/mis100/weighted_ER100_train/*gpickle" \
  --new_file "/data/shared/huiyuan/mis100/weighted_ER100_mixed_0.05_new" \
  --task mis \
  --weighted \
  --ratio 0.05