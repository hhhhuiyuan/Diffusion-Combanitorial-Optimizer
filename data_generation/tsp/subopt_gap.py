import time
import argparse
import pprint as pp
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from rodis.utils.tsp_utils import TSPEvaluator

# python -m data.tsp.subopt_gap --subopt_file "../data/shared/huiyuan/tsp100/tsp50_test_1023_supopt_test.txt" --opt_file "../data/shared/huiyuan/tsp100/tsp50_test_1023.txt"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subopt_file", type=str)
    parser.add_argument("--opt_file", type=str)
    #parser.add_argument("--new_opt_file", type=str)
    opts = parser.parse_args()

    subopt_file_lines = open(opts.subopt_file).read().splitlines()
    opt_file_lines = open(opts.opt_file).read().splitlines()
    subopt_gap = []
    opt_cost = []
    subopt_cost = []
    
    for subopt_data, opt_data in zip(subopt_file_lines, opt_file_lines):
        subopt_data = subopt_data.strip()
        opt_data = opt_data.strip()

        points_subopt, rest_line_subopt= subopt_data.split(' output ')
        points_subopt = points_subopt.split(' ')
        _, cost_subopt= rest_line_subopt.split(' time_cost ')
        points_subopt = np.array([[float(points_subopt[i]), float(points_subopt[i + 1])] for i in range(0, len(points_subopt), 2)])
        
        points_opt, tour = opt_data.split(' output ')
        points_opt = points_opt.split(' ')
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        tour -= 1
        points_opt = np.array([[float(points_opt[i]), float(points_opt[i + 1])] for i in range(0, len(points_opt), 2)])
        tsp_solver = TSPEvaluator(points_opt)
        cost_opt = tsp_solver.evaluate(tour)
        
        if np.array_equal(points_subopt, points_opt):
            opt_cost.append(cost_opt)
            subopt_cost.append(float(cost_subopt))
            opt_gap = (float(cost_subopt) - float(cost_opt))/float(cost_opt)
            subopt_gap.append(opt_gap)

    print(f"subopt gap is: {sum(subopt_gap)/len(subopt_gap)*100}% total samples: {len(subopt_gap)}") 
    print(f"opt cost is: {sum(opt_cost)/len(opt_cost)}, subopt cost is: {sum(subopt_cost)/len(subopt_cost)}") 
