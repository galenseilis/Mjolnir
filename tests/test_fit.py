"""Regression tests verifying that Mjolnir fits data correctly.

Run with:  python -m pytest tests/ -q
"""
import numpy as np
import networkx as nx
import pandas as pd

from thorhammer import Mjolnir, datasets


def _linear_dag_data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    dag = nx.DiGraph()
    dag.add_edge("X0", "X1")
    dag.add_edge("X1", "X2")

    x0 = rng.normal(size=n)
    x1 = 3.0 * x0 + rng.normal(scale=0.1, size=n)
    x2 = -2.0 * x1 + rng.normal(scale=0.1, size=n)
    data = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
    return dag, data


def _fast_gp():
    return {
        "generations": 10,
        "population_size": 500,
        "parsimony_coefficient": 0.001,
        "random_state": 0,
    }


def test_package_exports_mjolnir():
    from thorhammer import Mjolnir as Exported
    from thorhammer.mjolnir import Mjolnir as Direct

    assert Exported is Direct


def test_recovers_linear_relationships():
    dag, data = _linear_dag_data()
    model = Mjolnir(dag, gp_params=_fast_gp())
    model.fit(data)

    preds = model.predict(data)
    for node in ["X1", "X2"]:
        p = np.asarray(preds[node], dtype=float)
        t = np.asarray(data[node], dtype=float)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        assert rmse < 0.5, f"{node} RMSE too high: {rmse}"


def test_conformal_intervals_have_nominal_coverage():
    dag, data = _linear_dag_data()
    model = Mjolnir(dag, gp_params=_fast_gp())
    model.fit(data)
    model.conformal_fit(data)

    _, intervals = model.conformal_predict(data)
    for node, iv in intervals.items():
        iv = np.asarray(iv)
        assert iv.ndim == 3 and iv.shape[1] == 2, "unexpected interval shape"
        lower, upper = iv[:, 0, 0], iv[:, 1, 0]
        truth = np.asarray(data[node], dtype=float)
        coverage = np.mean((truth >= lower) & (truth <= upper))
        assert coverage >= 0.8, f"{node} coverage too low: {coverage}"


def test_dataset_generation_respects_edge_direction():
    np.random.seed(0)
    dag, data = datasets.make_dag_regression(n=5, m=1000)

    for parent in dag.nodes():
        parents = list(dag.predecessors(parent))
        if not parents:
            # Exogenous roots must be unit-variance noise.
            assert abs(np.std(data[parent]) - 1.0) < 0.2
