from itertools import product
import random
from typing import NoReturn

import networkx as nx
import numpy as np
import pandas as pd


def decycle(d: nx.DiGraph) -> NoReturn:
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


def random_dag(n: int = 100, m: int = 1000, node_prefix: str = "X") -> nx.DiGraph:
    """Generate a random DAG with ``n`` nodes and up to ``m`` edges.

    Self-loops are excluded and cycles are removed, so the returned graph is
    guaranteed to be a directed acyclic graph.
    """
    nodes = [f"{node_prefix}{i}" for i in range(n)]
    possible_edges = [(u, v) for u, v in product(nodes, repeat=2) if u != v]
    d = nx.DiGraph()
    d.add_edges_from(random.sample(possible_edges, min(m, len(possible_edges))))
    decycle(d)
    return d


def make_dag_regression(n=100, m=1000, node_prefix="X"):
    """Generate a DAG and ``m`` samples from a linear structural causal model.

    Each edge ``(u, v)`` contributes ``beta * u`` to ``v``, so descendants are
    noisy linear functions of their ancestors. Models fit to this data should
    recover the parent relationships encoded in the DAG.
    """
    d = random_dag(n, n, node_prefix)
    data = np.random.normal(size=m * n).reshape(m, n)
    index = {f"{node_prefix}{i}": i for i in range(n)}
    betas = {edge: np.random.normal(scale=10) for edge in d.edges()}
    for parent in nx.topological_sort(d):
        for _, child in d.out_edges(parent):
            data[:, index[child]] += betas[(parent, child)] * data[:, index[parent]]
    return d, pd.DataFrame(data, columns=[f"{node_prefix}{i}" for i in range(n)])
