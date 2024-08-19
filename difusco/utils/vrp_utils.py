import networkx as nx
import numpy as np
from vrpy import VehicleRoutingProblem
import logging

def solve_vrp_masked(edges, demand, solution = None, num_mask = None, mask_idx = None):
    logging.getLogger('vrpy').setLevel(logging.WARNING)
    G = nx.DiGraph() 
    num_nodes = edges.shape[0]-1
    
    for i in range(1, num_nodes+1):
        G.add_edge("Source", i, cost=edges[0, i])
        G.add_edge(i, "Sink", cost=edges[i, 0])
        
        for j in range(i+1, num_nodes+1):
            G.add_edge(i, j, cost=edges[i, j])
            G.add_edge(j, i, cost=edges[j, i])
        
        G.nodes[i]["demand"] = demand[i]
    
    # prob = VehicleRoutingProblem(G, load_capacity=30)
    # prob.solve(heuristic_only=True)
    # print("heuristic cost_before_masking: ", prob.best_value)
    
    if mask_idx is not None:
        if  len(mask_idx) != 0:
            for i, j in mask_idx:
                G.add_edge(i, j, cost=-10)
                #G.add_edge(j, i, cost=-10)
    elif num_mask != 0:
        flattened = solution.flatten()
        top_indices_flat = np.argpartition(flattened, -num_mask)[-num_mask:]
        top_indices = np.unravel_index(top_indices_flat, solution.shape)
        top_indices = list(zip(top_indices[0], top_indices[1]))

        for i, j in top_indices:
            if i*j != 0:
                G.add_edge(i, j, cost=-10)
                #G.add_edge(j, i, cost=-10)

    prob = VehicleRoutingProblem(G, load_capacity=30)
    prob.solve(heuristic_only=True)
    #print("heuristic cost_after_masking: ", prob.best_value)

    for i in range(len(prob.best_routes)):
        prob.best_routes[i] = [loc if loc != "Source" and loc != "Sink" else 0 for loc in prob.best_routes[i]]
    
    return prob.best_routes

def vrp_evaluate(edges, demand, tour):
    # validate capacity constraints
    full_route = []
    for i in range(len(tour)):
        route = tour[i]
        full_route.append(route)
        assert sum(demand[node] for node in route) <= 30, "Capacity constraint violated!"
    num_nodes = edges.shape[0]-1
    full_route = np.concatenate(full_route)
    assert (np.sort(full_route)[-num_nodes:] == np.arange(num_nodes) + 1).all(), "All nodes must be visited once!"
    visited_edge = edges[(full_route[:-1], full_route[1:])]
    
    return visited_edge.sum()


def solve_by_heuristic(edge_matrix, demands, solution_mat, num_mask):
    bs = solution_mat.shape[0]
    vrp_tours = []
    vrp_costs = []
    for i in range(bs):
        edges = edge_matrix[i]
        demand = demands[i]
        solution = solution_mat[i]
        #construct probelm graph
        tour = solve_vrp_masked(edges, demand, solution, num_mask)
        cost = vrp_evaluate(edges, demand, tour)
        
        vrp_tours.append(tour)
        vrp_costs.append(cost)
       
    return vrp_tours, vrp_costs