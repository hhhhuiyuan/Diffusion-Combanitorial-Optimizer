#run by $python data/tsp/subopt_data_process.py
input_filename = "/data/shared/huiyuan/tsp50/tsp50_nearopt_train.txt"
output_filename = "/data/shared/huiyuan/tsp50/tsp50_uncond_nearopt_train.txt"

with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
    for line in infile:
        line = line[:line.index(" time_cost")] + "\n"
        outfile.write(line)