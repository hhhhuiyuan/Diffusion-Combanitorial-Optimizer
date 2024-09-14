import time
import argparse
import pprint as pp
import os
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

import numpy as np
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde
from difusco.heuristics import solve_w_heuristics
from scipy.spatial.distance import cdist
from difusco.utils.tsp_algorithms import get_lower_adj_matrix
from difusco.utils.tsp_utils import TSPEvaluator
from tqdm import tqdm
from functools import partial

#run by $python -m data.generate_tsp_data_small --num_samples=128 --filename "./data/tsp20_rwd/tsp20_train_copy.txt" --reward_labelled

def solve_tsp_heuristic(nodes_coord, nearopt_flag = False):
    edge_matrix = cdist(nodes_coord, nodes_coord, 'euclidean')
    available_solvers = ['furthest']
                         #'nn', 'lkh-fast']
    key = random.choice(available_solvers)
    
    # if nearopt_flag and key != 'lkh-fast':
    #     return
        
    heuristic_tours, heuristic_costs = solve_w_heuristics(get_lower_adj_matrix(edge_matrix), key)  
    tour = heuristic_tours[key][1:]
    time_cost = heuristic_costs[key]
    print(f"heuristic time: {time_cost}")

    return (tour, time_cost)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=76800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument('--reward_labelled', action='store_true')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--near_opt", action='store_true')

    opts = parser.parse_args()
    
    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
    
    np.random.seed(opts.seed)
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"

    # Pretty print the run args
    pp.pprint(vars(opts))
    
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm(range(opts.num_samples//opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes+1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes
            
            batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])
                
            with Pool(opts.batch_size) as p:
                results = p.map(partial(solve_tsp_heuristic, nearopt_flag=opts.near_opt), [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

            for idx, result in enumerate(results):
                if result is None:
                    continue
                tour = result[0]
                time_cost = result[1]
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(str(" ") + str('output') + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write( str(" ") + str('time_cost') + str(" ") + str(time_cost) )
                    f.write("\n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")

    # plt.figure()
    # plt.hist(heurestic_time, bins=10, alpha=0.5, label='heuristic')
    # plt.hist(optimal_time, bins=10, alpha=0.5, label='optimal')
    # plt.legend(loc='upper right')
    # plt.savefig('./heuristic_vs_optimal.png')
    # plt.show()