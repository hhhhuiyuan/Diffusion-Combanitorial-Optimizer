import time
import argparse
import pprint as pp
import os
import random
#import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

import numpy as np

def solve_w_gurobi(n, m, xc, yc):
    I = [i for i in range(0, n)] # locations
    A = [(i, j) for i in I for j in I] # 2-D cartesian product
    c = {(i, j): 1*np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for (i, j) in A} # Cost to reach customer from Facility

    mdl = Model('UFLP')
    x = mdl.addVars(I, vtype = GRB.BINARY)
    y = mdl.addVars(A, vtype = GRB.BINARY)

    mdl.ModelSense = GRB.MINIMIZE # Minimisation model
    mdl.setObjective(quicksum(c[i,j]*y[i,j] for i,j in A)) # Cost Function
    mdl.addConstrs(quicksum(y[i,j] for j in I) == 1 for i in I)
    mdl.addConstr(quicksum(x[i] for i in I) == m)
    mdl.addConstrs(y[i,j] <= x[j] for i,j in A)
                    
    mdl.optimize()
    return mdl

def solve_w_sampling(n, m, xc, yc):
    solu = random.sample(range(n), m)
    points = np.stack((xc, yc), axis =1)
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    nearest_distances = np.min(distances[:, solu], axis=1)
    solu_obj = np.sum(nearest_distances)
    
    return solu, solu_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_loc", type=int, default=20)
    parser.add_argument("--num_fac", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument('--label', type=str, choices=['opt', 'sub_opt'], default='opt')
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()
    
    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
    
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Pretty print the run args
    pp.pprint(vars(opts))

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in range(opts.num_samples//opts.batch_size):
            n = opts.num_loc # Number of Customer
            m = opts.num_fac  # Numebr of Facility
            xc = np.random.random(n) * 10 # x-coordinate
            yc = np.random.random(n) * 10 # y-coordinate

            idx = 0
            while idx < opts.batch_size:
                if opts.label == 'opt':
                    mdl = solve_w_gurobi(n, m, xc, yc)
                    if mdl.status == 2:
                        solu_label = np.array(mdl.X[:n])
                        solu = np.nonzero(solu_label)[0]
                        solu_obj = mdl.ObjVal

                        f.write( " ".join( str(cord_x)+str(" ")+str(cord_y) for cord_x,cord_y in zip(xc, yc)) )
                        f.write( str(" ") + str('output') + str(" ") )
                        f.write( str(" ").join( str(node_idx) for node_idx in solu) )
                        f.write( str(" ") + str('cost') + str(" ") + str(solu_obj) )
                        f.write( "\n" )

                        idx += 1

                elif opts.label == 'sub_opt':
                    solu, solu_obj = solve_w_sampling(n, m, xc, yc)

                    f.write( " ".join( str(cord_x)+str(" ")+str(cord_y) for cord_x,cord_y in zip(xc, yc)) )
                    f.write( str(" ") + str('output') + str(" ") )
                    f.write( str(" ").join( str(node_idx) for node_idx in solu) )
                    f.write( str(" ") + str('cost') + str(" ") + str(solu_obj) )
                    f.write( "\n" )

                    idx += 1
     
            assert idx == opts.batch_size
            
        end_time = time.time() - start_time
        
        assert b_idx == opts.num_samples//opts.batch_size - 1
    
    # plt.figure()
    # plt.hist(heurestic_time, bins=10, alpha=0.5, label='heuristic')
    # plt.hist(optimal_time, bins=10, alpha=0.5, label='optimal')
    # plt.legend(loc='upper right')
    # plt.savefig('./heuristic_vs_optimal.png')
    # plt.show()
  
    # print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    # print(f"Total time: {end_time/60:.1f}m")
    # print(f"Average time: {end_time/opts.num_samples:.1f}s")