from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import networkx as nx
import numpy as np
import random
import torch
import copy
import math

from queue import PriorityQueue, Queue
from collections import deque

def arrays_to_digraph(nodes, dep_mat):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(len(nodes)):
        G.add_node(i, features=nodes[i].tolist())

    # Add edges to the graph based on the dependency matrix
    for i in range(len(dep_mat)):
        for j in range(len(dep_mat[i])):
            if np.isclose(dep_mat[i][j], 1.0):
                G.add_edge(i, j)

    return G

def transitive_closure(dep_mat):
        '''
        return the transitive closure that has e_ij = 1 if i is an ancestor of j
        '''
        n_nodes = dep_mat.shape[0]
        dep_mat = dep_mat.astype(bool)
        identity_like_adj_mat = np.identity(n_nodes, dtype=bool)
        closure = identity_like_adj_mat | dep_mat
        # Floyd-Warshall Algorithm
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    closure[j, i] = closure[j, i] or (closure[k, i] and closure[j, k])
        closure = ~identity_like_adj_mat & closure
        return closure

def update_transitive_closure(trans_closure, i_node, j_node):
        n_nodes = trans_closure.shape[0]
        trans_closure[i_node, j_node] = True
        trans_closure[:, j_node] |= trans_closure[:, i_node]
        for k in range(n_nodes):
            if trans_closure[j_node, k]:
                trans_closure[:, k] |= trans_closure[:, i_node]
                trans_closure[i_node, k] = True

class DAG_Evaluator(object):
    def __init__(self, resource_dim, node_feature_dim):
        self.evaluator = DAGraph(resource_dim=resource_dim,
                        feature_dim=node_feature_dim,
                    )             
    def evaluate(self, nodes, dep_mat, order):
        """evaluate a schedle ordering given a dag
        nodes: np.array, shape=(num_nodes, node_feature_dim)
        dep_mat: np.array, shape=(num_nodes, num_nodes)
        order: list, shape=(num_nodes,)
        """
        dag_graph = arrays_to_digraph(nodes, dep_mat)
        return self.evaluator.scheduling_evaluator(dag_graph, order)
    
    def decode_w_heuristic(self, nodes, dep_mat, slu_mat, scheduler, decode_factor):
        """using heuristc to decode a schedule ordering given a dag and slu matrix
        nodes: np.array, shape=(num_nodes, node_feature_dim)
        dep_mat: np.array, shape=(num_nodes, num_nodes)
        slu_mat: np.array, shape=(num_nodes, num_nodes), entries in [0,1]
        order: np.array, shape=(num_nodes,)
        """

        flat_slu_mat = slu_mat.flatten()
        sorted_indices = np.argsort(flat_slu_mat)[::-1]
        rows, cols = np.unravel_index(sorted_indices, slu_mat.shape)
        indices = list(zip(rows, cols))

        # avoid adding edges that voilate the original dependency in dag
        # self loop edge is not allowed
        # edges in the opposite directions of the original dep_mat are not allowed
        
        #tuning parameter max_add
        max_add = math.floor((nodes.shape[0] - 1)* decode_factor)
        
        edge_mat = copy.deepcopy(dep_mat)
        n_nodes = dep_mat.shape[0]
        identity_mat = np.identity(n_nodes, dtype=bool)
        trans_closure = transitive_closure(edge_mat)
        redundant_mask = ~trans_closure
        cyclic_mask = ~(trans_closure.T | identity_mat)
        mask = redundant_mask & cyclic_mask
        
        for idx in indices:
            if max_add == 0:
                break
            if mask[idx[0], idx[1]]:
                edge_mat[idx[0], idx[1]] = 1
                update_transitive_closure(trans_closure, idx[0], idx[1])
                redundant_mask = ~trans_closure
                cyclic_mask = ~(trans_closure.T | identity_mat)
                mask = redundant_mask & cyclic_mask
                max_add -= 1
        
        dag_graph = arrays_to_digraph(nodes, edge_mat)
        
        if scheduler == [1,0,0]:
            time_cost, order = self.evaluator.shortest_first_scheduling(dag_graph)
        elif scheduler == [0,1,0]:
            time_cost, order = self.evaluator.critical_path_scheduling(dag_graph)
        elif scheduler == [0,0,1]:
            time_cost, order = self.evaluator.tetris_scheduling(dag_graph)
        makespan = self.evaluate(nodes, dep_mat, order)
        return order, makespan
    
    def decode_greedy(self, nodes, dep_mat, slu_mat, scheduler, decode_factor):
        """using heuristc to decode a schedule ordering given a dag and slu matrix
        nodes: np.array, shape=(num_nodes, node_feature_dim)
        dep_mat: np.array, shape=(num_nodes, num_nodes)
        slu_mat: np.array, shape=(num_nodes, num_nodes), entries in [0,1]
        order: np.array, shape=(num_nodes,)
        """

        flat_slu_mat = slu_mat.flatten()
        sorted_indices = np.argsort(flat_slu_mat)[::-1]
        rows, cols = np.unravel_index(sorted_indices, slu_mat.shape)
        indices = list(zip(rows, cols))

        # avoid adding edges that voilate the original dependency in dag
        # self loop edge is not allowed
        # edges in the opposite directions of the original dep_mat are not allowed
        
        #tuning parameter max_add
        if decode_factor > 1:
            max_add = int(decode_factor)
        else:
            max_add = math.floor((nodes.shape[0] - 1)* decode_factor)
        
        edge_mat = copy.deepcopy(dep_mat)
        ref_cost, ref_order = self.evaluator.critical_path_scheduling(arrays_to_digraph(nodes, edge_mat))
        
        n_nodes = dep_mat.shape[0]
        identity_mat = np.identity(n_nodes, dtype=bool)
        trans_closure = transitive_closure(edge_mat)
        redundant_mask = ~trans_closure
        cyclic_mask = ~(trans_closure.T | identity_mat)
        mask = redundant_mask & cyclic_mask
        
        for idx in indices:
            if max_add == 0:
                break
            if mask[idx[0], idx[1]]:
                edge_mat[idx[0], idx[1]] = 1
                update_transitive_closure(trans_closure, idx[0], idx[1])
                redundant_mask = ~trans_closure
                cyclic_mask = ~(trans_closure.T | identity_mat)
                mask = redundant_mask & cyclic_mask
                max_add -= 1
        
        dag_graph = arrays_to_digraph(nodes, edge_mat)
        
        # if scheduler == [1,0,0]:
        #     time_cost, order = self.evaluator.shortest_first_scheduling(dag_graph)
        # elif scheduler == [0,1,0]:
        #     time_cost, order = self.evaluator.critical_path_scheduling(dag_graph)
        # elif scheduler == [0,0,1]:
        #     time_cost, order = self.evaluator.tetris_scheduling(dag_graph)
        time_cost, order = self.evaluator.critical_path_scheduling(dag_graph)
        makespan = self.evaluate(nodes, dep_mat, order)
        return order, makespan, ref_cost
    
    def batch_decode_greedy(self, nodes_batach, dep_mat_batch, slu_mat_batch, scheduler_batch, decode_factor):
        all_order = []
        all_makespan = []
        all_ref_cost = []
        bs = nodes_batach.shape[0]
        for i in range(bs):
            order, makespan, ref_cost = self.decode_greedy(nodes_batach[i], dep_mat_batch[i], slu_mat_batch[i], scheduler_batch[i], decode_factor)
            all_order.append(order)
            all_makespan.append(makespan)
            all_ref_cost.append(ref_cost)
        makespan = sum(all_makespan) / bs
        ref_cost = sum(all_ref_cost) / bs
        return all_order, makespan, ref_cost
    
class DAGraph(object):
    def __init__(self, resource_dim, feature_dim, scheduler_type='sjf'):
        self.feature_dim = feature_dim
        self.resource_dim = resource_dim
        self.scheduler_type = scheduler_type
        self.resource_limits = np.array([1.0] * resource_dim)

    def step(self, input_graph, act, prev_greedy):
        new_graph = input_graph.copy()
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        new_graph.add_edge(act[1], act[0],
                           features=[0.0] * self.feature_dim)
        assert nx.is_directed_acyclic_graph(new_graph)
        new_greedy = self.makespan_time(new_graph, self.scheduler_type)
        reward = prev_greedy - new_greedy
        edge_candidates = self.get_edge_candidates(new_graph)
        done = all([len(x) == 0 for x in edge_candidates.values()])
        return reward, new_graph, new_greedy, edge_candidates, done

    def step_e2e(self, input_graph, act, prev_makespan):
        new_graph = input_graph.copy()
        if act == -1:
            running_nodes = self.get_running_nodes(new_graph)
            if len(running_nodes) > 0:
                proceed_time = min([-new_graph.node[running_id]['features'][0] for running_id in running_nodes])
                assert proceed_time > 0
                for running_id in running_nodes:
                    new_graph.node[running_id]['features'] = [
                        new_graph.node[running_id]['features'][0] + proceed_time,
                        new_graph.node[running_id]['features'][1]
                    ]
            else:
                proceed_time = 0.1  # penalty if there is no running nodes but act=-1
        else:
            new_graph.node[act]['features'] = [-new_graph.node[act]['features'][0], new_graph.node[act]['features'][1]]
            proceed_time = 0

        node_candidates, finished_nodes = self.get_node_candidates(new_graph, return_finished=True)

        reward = -proceed_time

        return reward, new_graph, prev_makespan + proceed_time, node_candidates, len(finished_nodes) == len(new_graph)

    def get_running_nodes(self, graph):
        running = set()
        for node, data in graph.nodes(data=True):
            duration, resource = data['features']
            if duration < 0:
                running.add(node)
        return running

    def get_node_candidates(self, graph, return_finished=False):
        duration = {}
        resource = {}
        running = set()
        finished = set()
        for node, data in graph.nodes(data=True):
            duration[node], resource[node] = data['features']
            if duration[node] < 0:
                running.add(node)
            elif duration[node] == 0:
                finished.add(node)
        available_resource = 1
        for running_node in running:
            available_resource -= resource[running_node]
        if available_resource < -1e-4:
            raise ValueError('Exceed maximum resource limit:', available_resource)

        dep_map, _, __ = self.get_dependency_nodes(graph)
        for finished_node in finished:
            dep_map = self.remove_dependency(dep_map, finished_node)
        node_candidates = set()
        for candidate in self.get_ready_nodes(dep_map):
            if candidate in running or candidate in finished:
                pass
            elif available_resource >= resource[candidate]:
                node_candidates.add(candidate)
        if return_finished:
            return node_candidates, finished
        else:
            return node_candidates

    def add_features(self):
        running_time = float(random.randint(1, 10))
        resources = [float(random.randint(1, 10)) / 10.0
                     for _ in range(self.resource_dim)]
        features = [running_time]
        features.extend(resources)
        return features

    def generate_graph_tupsles(self, num_samples=1, num_graph_range=(1,1)):
        tuples = []
        for i in range(num_samples):
            n_graphs = random.randint(num_graph_range[0], num_graph_range[1])
            input_graph, num_nodes = self.generate_graphs(num_graphs=n_graphs)
            orig_sft = self.shortest_first_time(input_graph)
            tuples.append((input_graph, orig_sft))
        return tuples

    def generate_graphs(self, num_graphs=1):
        num_levels_min_max = (4, 6)
        num_nodes_per_level = (4, 8)
        num_conn_min_max = (1, 3)
        empty_features = np.array([0.0] * self.feature_dim, dtype=float)

        # generate graph
        root_index = 0
        graph = nx.DiGraph()
        graph.graph["features"] = empty_features
        graph.add_node(root_index, features=empty_features)

        # add nodes and edges
        node_index = 1
        for _ in range(num_graphs):
            # add nodes
            level_nodes = [[root_index]]
            num_levels = random.randint(*num_levels_min_max)
            for _ in range(num_levels):
                level_nodes.append([])
                num_nodes = random.randint(*num_nodes_per_level)
                for _ in range(num_nodes):
                    level_nodes[-1].append(node_index)
                    graph.add_node(node_index, features=self.add_features())
                    node_index += 1

            # add edges
            for level, nodes in enumerate(level_nodes):
                if level == 0:
                    continue
                for node2 in nodes:
                    num_conn = random.randint(*num_conn_min_max)
                    num_conn = min(num_conn, len(level_nodes[level - 1]))
                    conn_nodes = np.random.choice(level_nodes[level - 1],
                                                  num_conn, replace=False)
                    for node1 in conn_nodes:
                        graph.add_edge(node2, node1, features=empty_features)
        #graph.add_node(node_index, features=empty_features)
        #graph.add_node(node_index + 1, features=empty_features)
        return graph, node_index + 2

    def generate_fixed_graph1(self):
        # 6-node graph
        graph = nx.DiGraph()
        graph.add_node(0, features=[1.0, 0.1, 0.1])
        graph.add_node(1, features=[5.0, 0.3, 0.06])
        graph.add_node(2, features=[4.0, 0.2, 0.05])
        graph.add_node(3, features=[3.0, 0.4, 0.4/3])
        graph.add_node(4, features=[2.0, 0.3, 0.15])
        graph.add_node(5, features=[1.0, 0.1, 0.1])
        empty_features = [0.0, 0.0, 0.0]
        graph.add_edge(1, 0, features=empty_features)
        graph.add_edge(2, 0, features=empty_features)
        graph.add_edge(3, 0, features=empty_features)
        graph.add_edge(4, 0, features=empty_features)
        graph.add_edge(5, 1, features=empty_features)
        graph.add_edge(5, 2, features=empty_features)
        graph.add_edge(5, 3, features=empty_features)
        graph.add_edge(5, 4, features=empty_features)
        graph.graph["features"] = empty_features
        return graph, 6

    def generate_fixed_graph2(self):
        graph = nx.DiGraph()
        graph.add_node(0, features=[2.0, 0.1, 0.1])
        graph.add_node(1, features=[2.0, 0.5, 0.06])
        graph.add_node(2, features=[4.0, 0.2, 0.05])
        graph.add_node(3, features=[3.0, 0.4, 0.4/3])
        graph.add_node(4, features=[2.0, 0.3, 0.15])
        graph.add_node(5, features=[1.0, 0.1, 0.1])
        graph.add_node(6, features=[2.0, 0.1, 0.1])
        empty_features = [0.0, 0.0, 0.0]
        # 0 -> 1, 2, 3
        graph.add_edge(1, 0, features=empty_features)
        graph.add_edge(2, 0, features=empty_features)
        graph.add_edge(3, 0, features=empty_features)
        # 1 -> 4
        graph.add_edge(4, 1, features=empty_features)
        # 2 -> 5
        graph.add_edge(5, 2, features=empty_features)
        # 3 -> 5, 6
        graph.add_edge(5, 3, features=empty_features)
        graph.add_edge(6, 3, features=empty_features)
        # 4 -> 3
        graph.add_edge(3, 4, features=empty_features)
        graph.graph["features"] = empty_features
        return graph, 7

    def generate_fixed_graph(self):
        graph = nx.DiGraph()
        node_feature = [0.0] * self.feature_dim
        graph.add_node(0, features=node_feature)
        graph.add_node(1, features=node_feature)
        graph.add_edge(0, 1, features=node_feature)
        graph.graph["features"] = node_feature
        return graph, 2

    def get_dependency_nodes(self, graph):
        parents = {}
        children = {}
        features = {}
        for node, data in graph.nodes(data=True):
            parents[node] = set()
            children[node] = set()
            features[node] = data
        for from_node, to_node in graph.edges(data=False):
            parents[from_node].add(to_node)
            children[to_node].add(from_node)
        return parents, children, features

    def get_relations(self, relation_map, node, relations):
        relates = relation_map[node]
        while relates:
            next_list = set()
            for relate in relates:
                relations.add(relate)
                next_list = next_list.union(relation_map[relate])
            relates = next_list

    def get_edge_candidates(self, graph):
        """Candidates are node pairs that are not on any paths.
        This is not a bottle neck yet, but there is no need to repetitively call
        this whole process after adding one single edge.
        """
        relations_map = {}
        parents, children, _ = self.get_dependency_nodes(graph)
        for node in graph.nodes(data=False):
            relations = set()
            self.get_relations(parents, node, relations)
            self.get_relations(children, node, relations)
            relations_map[node] = relations

        edge_candidates = {}
        num_nodes = len(graph.nodes())
        for i in range(num_nodes):
            edge_candidates[i] = set(range(num_nodes)) - \
                                 set([i]) - relations_map[i]
        return edge_candidates

    def get_ready_nodes(self, dependency_nodes):
        ready_nodes = []
        for node, dep_nodes in dependency_nodes.items():
            if len(dep_nodes) == 0:
                ready_nodes.append(node)
        return ready_nodes

    def remove_dependency(self, dependency_map, remove_node):
        dependency_map.pop(remove_node)
        for node, dep_nodes in dependency_map.items():
            if remove_node in dep_nodes:
                dep_nodes.remove(remove_node)
        return dependency_map

    def add_edges(self, graph, added_edges):
        for edge in added_edges:
            graph.add_edge(edge[1], edge[0], features=[0.0] * self.feature_dim)
        return graph

    def scheduling_evaluator(self, graph, order):
        """
        Evaluate the makespan time of the given order.
            order: list node ids.
            return: makespan time if the order is valid, otherwise None.
        """
        if sorted(order) != list(range(graph.number_of_nodes())):
            return None
        resource_dim = self.resource_limits.shape[0]
        resource_limit = self.resource_limits
        dep_map, _, features_map = self.get_dependency_nodes(graph)
        run_queue = PriorityQueue()
        used_resource = np.zeros(self.resource_dim)
        wallclock = 0.0
        job_queue = deque()
        for node in order:
            job_queue.append(node)
        while not wallclock or job_queue or not run_queue.empty():
            #import pdb; pdb.set_trace()
            # peek finished jobs from run_queue)
            current_time = None
            processed = []
            while not run_queue.empty():
                completion_time, spend, resource, node = run_queue.get()
                resource = np.array(resource)
                if current_time is None:
                    current_time = completion_time
                    assert wallclock <= current_time
                    wallclock = current_time
                if completion_time == current_time:
                    used_resource = used_resource - resource
                    dep_map = self.remove_dependency(dep_map, node)
                    processed.append(node)
                elif completion_time > current_time:
                    run_queue.put((completion_time, spend, resource.tolist(), node))
                    break
                else:
                    raise Exception('finish_time less than current_time')

            ready_nodes = self.get_ready_nodes(dep_map)
            ready_in_job_queue = 0
            
            if job_queue:
                while job_queue and job_queue[0] in ready_nodes:
                    ready_in_job_queue += 1
                    cur_job = job_queue[0]
                    values = features_map[cur_job]['features']
                    spend = values[0]
                    resource =  np.array(values[1:resource_dim + 1])
                    if np.all(used_resource + resource <= resource_limit):
                        used_resource += resource
                        run_queue.put((wallclock + spend, spend, resource.tolist(), cur_job))
                        job_queue.popleft()
                    else:
                        break
                if run_queue.empty() and ready_in_job_queue == 0:
                    return None
        return wallclock
    
    def makespan_time(self, dag_graph, scheduler_type='sft'):
        if scheduler_type == 'sft':
            return self.shortest_first_time(dag_graph)
        elif scheduler_type == 'cp':
            return self.critical_path_scheduling(dag_graph)
        elif scheduler_type == 'ts':
            return self.tetris_scheduling(dag_graph)

    
    def shortest_first_time(self, graph, print_solution=False):
        order = []
        dep_map, _, features_map = self.get_dependency_nodes(graph)
        run_queue = PriorityQueue()
        used_resource = [0.0] * self.resource_dim
        wallclock = 0.0
        while not wallclock or not run_queue.empty():
            # peek finished jobs from run_queue
            current_time = None
            processed = []
            while not run_queue.empty():
                completion_time, spend, resource, node = run_queue.get()
                if current_time is None:
                    current_time = completion_time
                    assert wallclock <= current_time
                    wallclock = current_time
                if completion_time == current_time:
                    used_resource = np.subtract(used_resource, resource)
                    dep_map = self.remove_dependency(dep_map, node)
                    processed.append(node)
                elif completion_time > current_time:
                    run_queue.put((completion_time, spend, resource, node))
                    break
                else:
                    raise Exception('SJF: finish_time less than current_time')

            # schedule ready nodes to run in shortest-first order
            ready_nodes = self.get_ready_nodes(dep_map)
            combos = sorted([(features_map[node]['features'], node)
                             for node in ready_nodes])
            run_nodes = set([q[-1] for q in run_queue.queue])
            for features, node in combos:
                spend, resource = features[0], features[1:(1 + self.resource_dim)]
                if node not in run_nodes:
                    to_resource = np.add(used_resource, resource)
                    if np.all(to_resource <= self.resource_limits):
                        used_resource = to_resource
                        completion_time = wallclock + spend
                        run_queue.put((completion_time, spend, resource, node))
                        order.append(node)
            if print_solution:
                print('wallclock {} processed {} queued {}'.format(
                    current_time, processed, run_queue.queue))
        return wallclock, order

    def ranker_based_scheduling(self, graph, ranker, resource_limit):
        order = []
        dep_map, _, features_map = self.get_dependency_nodes(graph)
        run_queue = PriorityQueue()
        used_resource = np.zeros(self.resource_dim)
        wallclock = 0.0
        while not wallclock or not run_queue.empty():
            # peek finished jobs from run_queue
            current_time = None
            processed = []
            while not run_queue.empty():
                completion_time, spend, resource, node = run_queue.get()
                resource = np.array(resource)
                if current_time is None:
                    current_time = completion_time
                    assert wallclock <= current_time
                    wallclock = current_time
                if completion_time == current_time:
                    used_resource = used_resource - resource
                    dep_map = self.remove_dependency(dep_map, node)
                    processed.append(node)
                elif completion_time > current_time:
                    run_queue.put((completion_time, spend, resource.tolist(), node))
                    break
                else:
                    raise Exception('finish_time less than current_time')

            # schedule ready nodes to run in shortest-first order
            ready_nodes = self.get_ready_nodes(dep_map)
            # The first of 'features' happen to be running_time
            ranked_nodes = ranker(
                ready_nodes,
                features_map,
                resource_limit,
                resource_limit - used_resource)
            combos = [(features_map[node]['features'], node)
                for node in ranked_nodes]
            run_nodes = set([q[3] for q in run_queue.queue])
            for values, node in combos:
                spend = values[0]
                resource =  np.array(values[1:resource_limit.shape[0] + 1])
                if node not in run_nodes and \
                        np.all(used_resource + resource <= resource_limit):
                    used_resource += resource
                    run_queue.put((wallclock + spend, spend, resource.tolist(), node))
                    order.append(node)
        return wallclock, order

    def shortest_first_scheduling(self, graph, resource_limit=None):
        if resource_limit is None:
            resource_limit = self.resource_limits
        def ranker(ready_nodes, feature_map, resource_limit,
                   remaining_resource):
            run_time_ind = 0
            ordered = sorted(
                [(feature_map[node]['features'][run_time_ind], node)
                    for node in ready_nodes])
            return [node for _, node in ordered]
        return self.ranker_based_scheduling(graph, ranker, resource_limit)

    def longest_path_to_any_leaf(self, graph, normalize=False):
        # find all leaves.
        running_time_index = 0
        stack = []  # append == push; pop(-1) == pop
        scores = {}  # longest path from a node to ANY leave, inclusive.
        for node, data in graph.nodes(data=True):
            children = []
            children.extend(graph.predecessors(node))
            if not children:
                stack.append(node)
                scores[node] = data['features'][running_time_index]
        if not stack:
            raise Exception('Critical path: cycles found in graph.')
        # compute critical path scores. Worse case O(n^2)
        while stack:
            node = stack.pop(-1)
            assert node in scores
            parents = graph.successors(node)
            for p in parents:
                parent_time = graph.node[p]['features'][running_time_index]
                parent_score = parent_time + scores[node]
                if (p not in scores) or (scores[p] < parent_score):
                    scores[p] = parent_score
                    stack.append(p)  # recompute from here
        if normalize:
            largest = max(scores.values())
            for k, v in scores.items():
                scores[k] = v / largest
        return scores

    def critical_path_scheduling(self, graph, resource_limit=None):
        if resource_limit is None:
            resource_limit = self.resource_limits
        scores = self.longest_path_to_any_leaf(graph)
        def ranker(ready_nodes, feature_map, resource_limit,
                   remaining_resource, scores):
            ordered = sorted(
                [(scores[node], node) for node in ready_nodes], reverse=True)
            return [node for _, node in ordered]
        ranker2 = functools.partial(ranker, scores=scores)
        return self.ranker_based_scheduling(graph, ranker2, resource_limit)

    def tetris_scheduling(self, graph, resource_limit=None):
        """To be extended to multiple dimensions."""
        if resource_limit is None:
            resource_limit = self.resource_limits
        def ranker(ready_nodes, feature_map, resource_limit,
                   remaining_resource):
            assert isinstance(resource_limit, np.ndarray)
            assert isinstance(remaining_resource, np.ndarray)
            assert remaining_resource.shape == resource_limit.shape
            index_start = 1
            index_end = index_start + len(resource_limit)
            scores_and_nodes = []
            for node in ready_nodes:
                features = feature_map[node]['features']
                demand = features[index_start: index_end]
                score = 0.0
                for d, r, l in zip(demand, remaining_resource, resource_limit):
                    assert d <= l + 1e-5, 'd={} l={}'.format(d, l)
                    assert r <= l + 1e-5, 'r={} l={}'.format(r, l)
                    score += (d / l * r / l)
                scores_and_nodes.append((score, node))
            return [node for _, node in sorted(scores_and_nodes, reverse=True)]
        return self.ranker_based_scheduling(graph, ranker, resource_limit)
