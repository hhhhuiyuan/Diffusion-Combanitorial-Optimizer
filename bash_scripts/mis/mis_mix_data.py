import time
import argparse
import pprint as pp
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle5 as pickle
import networkx as nx
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subopt_file", type=str)
    parser.add_argument("--opt_file", type=str)
    parser.add_argument("--new_file", type=str)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--task", type=str, choices=["flp", "tsp", "mis"], required=True)
    parser.add_argument("--weighted", action="store_true")
    
    opts = parser.parse_args()
    
    os.makedirs(opts.new_file, exist_ok=True)

    if opts.task == "mis":
        subopt_file_lines = sorted(glob.glob(opts.subopt_file))
        print(len(subopt_file_lines))
        opt_file_lines = sorted(glob.glob(opts.opt_file))
        print(len(opt_file_lines))
        subopt_gap = []
        mixed_lines = []

        for subopt_data, opt_data in zip(subopt_file_lines, opt_file_lines):
            # now the problem graph in opt and subopt datasets are not the same
            with open(subopt_data, "rb") as f:
                subopt_graph = pickle.load(f)
            with open(opt_data, "rb") as f:    
                opt_graph = pickle.load(f)

            num_nodes_subopt = subopt_graph.number_of_nodes()
            num_nodes_opt = opt_graph.number_of_nodes()

            if not num_nodes_subopt == num_nodes_opt:
                print("nodes check failed")
                continue

            edges_subopt = np.array(subopt_graph.edges, dtype=np.int64)
            edges_opt = np.array(opt_graph.edges, dtype=np.int64)
            
            # print(edges_subopt)
            # print(edges_opt)
            if not (edges_subopt == edges_opt).all():
                print("edges check failed")
                continue

            weights_subopt = nx.get_node_attributes(subopt_graph, 'weight')
            weights_opt = nx.get_node_attributes(opt_graph, 'weight')
            
            # print(weights_subopt)
            # print(weights_opt)
            if not weights_subopt == weights_opt:
                print("weights check failed")
                continue
            
            choice = np.random.binomial(1, opts.ratio)
            if choice:
                print(opt_data)
                shutil.copy2(opt_data, opts.new_file)
            else:
                print(subopt_data)
                shutil.copy2(subopt_data, opts.new_file)
