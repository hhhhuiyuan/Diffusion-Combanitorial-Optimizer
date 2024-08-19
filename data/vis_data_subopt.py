import time
import argparse
import pprint as pp
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from difusco.utils.tsp_utils import TSPEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subopt_file", type=str)
    parser.add_argument("--opt_file", type=str)
    parser.add_argument("--task", type=str, choices=["flp", "tsp"])
    opts = parser.parse_args()

    subopt_file_lines = open(opts.subopt_file).read().splitlines()
    opt_file_lines = open(opts.opt_file).read().splitlines()

    subopt_gap = []
    
    if opts.task == "flp":
        for subopt_data, opt_data in zip(subopt_file_lines, opt_file_lines):
            subopt_data = subopt_data.strip()
            opt_data = opt_data.strip()

            points_subopt, rest_line_subopt= subopt_data.split(' output ')
            points_subopt = points_subopt.split(' ')
            _, cost_subopt= rest_line_subopt.split(' cost ')
            points_subopt = np.array([[float(points_subopt[i]), float(points_subopt[i + 1])] for i in range(0, len(points_subopt), 2)])
            
            points_opt, rest_line_opt= opt_data.split(' output ')
            points_opt = points_opt.split(' ')
            _, cost_opt= rest_line_opt.split(' cost ')
            points_opt = np.array([[float(points_opt[i]), float(points_opt[i + 1])] for i in range(0, len(points_opt), 2)])
            
            if np.array_equal(points_subopt, points_opt):
                subopt_gap.append((float(cost_subopt) - float(cost_opt))/float(cost_opt))

    elif opts.task == "tsp":
        for subopt_data, opt_data in zip(subopt_file_lines, opt_file_lines):
            subopt_data = subopt_data.strip()
            opt_data = opt_data.strip()

            points_subopt, rest_line_subopt= subopt_data.split(' output ')
            points_subopt = points_subopt.split(' ')
            _, cost_subopt= rest_line_subopt.split(' time_cost ')
            points_subopt = np.array([[float(points_subopt[i]), float(points_subopt[i + 1])] for i in range(0, len(points_subopt), 2)])
            
            points_opt, tour = opt_data.split(' output ')
            points_opt = points_opt.split(' ')
            points_opt = np.array([[float(points_opt[i]), float(points_opt[i + 1])] for i in range(0, len(points_opt), 2)])
            tour = tour.split(' ')
            tour = np.array([int(t) for t in tour])
            tour -= 1
            tsp_solver = TSPEvaluator(points_opt)
            cost_opt = tsp_solver.evaluate(tour)

            if np.array_equal(points_subopt, points_opt):
                subopt_gap.append((float(cost_subopt) - float(cost_opt))/float(cost_opt))

    plt.figure()
    plt.hist(subopt_gap, bins=80, alpha=0.5,  range=(0, 2.0))
    #plt.legend(loc='upper right')
    plt.savefig(f'./{opts.task}_suboptimality_gap.png')
    plt.show()