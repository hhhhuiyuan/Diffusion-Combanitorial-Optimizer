# test by $ python difusco/heuristics.py
import torch
import random
import os
import glob
import re
from copy import deepcopy
import time
import numpy as np
import tsplib95
from rodis.utils.tsp_algorithms import calc_furthest_insertion_tour_len, calc_lkh_tour_len, calc_nearest_neighbor_tour_len,\
    get_lower_matrix, solveFarthestInsertion, get_lower_adj_matrix

def process_dataset():
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")
        dirpath = '/data/home/huiyuan23/DIFUSCO/data/hcp_data'
        tspfiles = []
        min_size = 10
        max_size = 50

        def get_size(elem):
            prob_size = re.findall('[0-9]+', elem)
            if len(prob_size) != 1:
                prob_size = prob_size[1]
            else:
                prob_size = prob_size[0]
            return int(prob_size)

        for fp in glob.iglob(os.path.join(dirpath, "*.hcp")):
            if min_size <= get_size(fp) <= max_size:
                tspfiles.append(fp)
        tspfiles.sort(key=get_size)
        return tspfiles

def parse_tsp(list_m, dim=None, name='unknown'):
    if dim is None:
        dim = len(list_m)
    outstr = ''
    outstr += 'NAME: %s\n' % name #problem.name
    outstr += 'TYPE: TSP\n'
    outstr += 'COMMENT: %s\n' % name
    outstr += 'DIMENSION: %d\n' % dim #problem.dimension
    outstr += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
    outstr += 'EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW\n'
    outstr += 'EDGE_WEIGHT_SECTION:\n'
    for l in list_m:
        listToStr = ' '.join([str(elem) for elem in l])
        outstr += ' %s\n' % listToStr
    #outstr += 'EDGE_DATA_FORMAT: EDGE_LIST\n'
    #outstr += 'EDGE_DATA_SECTION:\n'
    #for edge_idx, weight in edges_dict.items():
    #    outstr += f' {edge_idx[0]+1} {edge_idx[1]+1} {weight}\n'
    #outstr += '-1\n'

    return outstr

def solve_feasible_tsp(lower_left_matrix, solver_type):
    prev_time = time.time()
    tsp_inst = tsplib95.parse(parse_tsp(lower_left_matrix))
    if solver_type == 'nn':
        tour, length = calc_nearest_neighbor_tour_len(tsp_inst)
    elif solver_type == 'furthest':
        tour, length = solveFarthestInsertion(tsp_inst)
    elif solver_type == 'lkh-fast':
        tour, length = calc_lkh_tour_len(tsp_inst, move_type=5, runs=10)
    else:
        raise ValueError(f'{solver_type} is not implemented.')
    comp_time = time.time() - prev_time
    return tour, length, comp_time


def solve_w_heuristics(lower_left_matrix, solver=None):
    tsp_solutions = {}
    tsp_tours = {}
    tsp_times = {}
    #problem_dim = len(lower_left_matrix)
    if solver is None:
        available_solvers = ['nn','furthest','lkh-fast']
        for key in available_solvers:
            tour, sol, sec = solve_feasible_tsp(lower_left_matrix, key)
            tsp_tours[key] = tour
            tsp_solutions[key] = sol
            tsp_times[key] = sec
    else:
        tour, sol, sec = solve_feasible_tsp(lower_left_matrix, solver)
        tsp_tours[solver] = tour
        tsp_solutions[solver] = sol
        tsp_times[solver] = sec
    return (tsp_tours, tsp_solutions)

if __name__ == '__main__':
    tspfiles = process_dataset()
    tsp_path = tspfiles[0]
    problem = tsplib95.load(tsp_path)
    
    feasible_edge = 1
    infeasible_edge = 2
    lower_left_matrix = get_lower_matrix(problem, feasible_edge, infeasible_edge)
    
    available_solvers = ['nn','furthest', 'lkh-fast','lkh-accu']
    tsp_solutions = {}
    tsp_times = {}
    
    for key in available_solvers:
        tour, sol, sec = solve_feasible_tsp(lower_left_matrix, key)
        tsp_solutions[key] = sol - feasible_edge * problem.dimension
        tsp_times[key] = sec
        # if key == self.solver_type:
        #     edge_candidates = self.edge_candidate_from_tour(tour, problem.dimension)
    print(f'probelm dim: {int(problem.dimension)} '
        f'{"; ".join([f"{x} tour={tsp_solutions[x]:.2f} time={tsp_times[x]:.2f}" for x in available_solvers])}')
    