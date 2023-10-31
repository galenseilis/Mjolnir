from itertools import product
import random

import networkx as nx
import numpy as np
import pandas as pd


def decycle(d):
    """
    Pseudorandomly remove cycles from a directed graph.

    Changes the graph inplace.

    PARAMETERS
    ----------
    d : networkx.DiGraph
        Directed graph.

    RETURNS
    -------
    None
    """
    while True:
        if nx.is_directed_acyclic_graph(d):
            break
        first_cycle = nx.cycles.find_cycle(d)
        target_edge = random.sample(first_cycle, 1)[0]
        d.remove_edge(*target_edge)


def random_dag(n=100, m=1000, node_prefix="X"):
    possible_edges = [
        (f"{node_prefix}{pair[0]}", f"{node_prefix}{pair[1]}")
        for pair in product(range(n), repeat=2)
    ]
    d = nx.DiGraph()
    edges = random.sample(possible_edges, n)
    d.add_edges_from(edges)
    decycle(d)
    return d


def make_dag_regression(n=100, m=1000, node_prefix="X"):
    d = random_dag(n, m, node_prefix)
    data = np.random.normal(size=m * n).reshape(m, n)
    betas = {edge: np.random.normal(scale=10) for edge in d.edges()}
    for node in nx.topological_sort(d):
        for edge in d.out_edges(node):
            data[:, int(node[1:])] += betas[edge] * data[:, int(edge[1][1:])]
    return d, pd.DataFrame(data, columns=[f"{node_prefix}{i}" for i in range(n)])
