from typing import Any, Dict, NoReturn

import graphviz
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
import networkx as nx

import warnings

# TODO: Implement Markov misclassification class for classifiers.


# TODO: Decide if I want to add measurement error terms.
# TODO: Decide if/how to deal with missing data.
# TODO: Latent variables
# TODO: Add preformations and  postformations as transformations before/after prediction respectively.
class DAGModel(BaseEstimator, TransformerMixin):
    """Directed acylic graph of predictive models."""

    def __init__(
        self,
        dag: nx.DiGraph,
        models: dict,
        transforms: dict = None,
        verbose: bool = False,
    ) -> NoReturn:
        """
        Initialize a DAGModel.

        PARAMETERS
        ----------
            dag (networkx.DiGraph): A directed acyclic graph specifying variable relationships.
            models (dict): A dictionary of Scikit-Learn models to be used for each variable.
            transforms (dict): Transformations to variables.

        Example models dictionary:
        models = {
            'X0': LinearRegression(),
            'X1': RandomForestRegressor(),
            'X2': SVR()
        }
        """

        self.verbose = verbose

        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("DiGraph must be acyclic.")

        self.dag = dag
        self.models = models

        for node in self.dag:
            if not list(self.dag.predecessors(node)) and node in self.models:
                warn_str = f"Variable {node} was assigned a model but does not have any predecessors."
                warnings.warn(warn_str)

        self.ordered_nodes = list(nx.topological_sort(self.dag))
        self.fitted_predictions = {}

    def fit(self, X: NDArray, y: NDArray = None):
        """
        Fit the DAGModel to the data.

        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

        Returns:
            self
        """
        # Iterate through the nodes in topological order and fit the models
        for node in self.ordered_nodes:
            # Get the input nodes for this variable
            input_nodes = list(self.dag.predecessors(node))

            if self.verbose and input_nodes:
                print(f"Fitting {node} as a function of {input_nodes}.")

            if input_nodes:
                input_nodes = sorted(
                    set(self.ordered_nodes) & set(input_nodes),
                    key=self.ordered_nodes.index,
                )
                # Collect predictors from input data and earlier predictions in DAG.
                input_data = np.column_stack(
                    [
                        self.fitted_predictions[in_node]
                        if in_node in self.fitted_predictions
                        else X[in_node]
                        for in_node in input_nodes
                    ]
                )

                # Fit model
                self.models[node].fit(input_data, X[node])

                # Store predictions
                self.fitted_predictions[node] = self.models[node].predict(input_data)

        return self

    def transform(self, X: NDArray) -> NoReturn:
        """
        Transform the data using the fitted models.

        Parameters:
            X (pd.DataFrame): The input features.

        Returns:
            output (pd.DataFrame): The transformed data.
        """
        # check_is_fitted(self, 'fitted_models')

        transformed_data = {}
        for node in self.ordered_nodes:
            if node in self.models:
                input_nodes = list(self.dag.predecessors(node))
                if input_nodes:
                    input_nodes = sorted(
                        set(self.ordered_nodes) & set(input_nodes),
                        key=self.ordered_nodes.index,
                    )
                    input_data = np.column_stack(
                        [
                            transformed_data[in_node]
                            if in_node in transformed_data
                            else X[in_node]
                            for in_node in input_nodes
                        ]
                    )
                    transformed_data[node] = self.models[node].predict(input_data)

        return transformed_data

    def predict(self, X: NDArray) -> dict:
        return self.transform(X)

    def do(self, query):
        """Compute a do-calculus query.

        PARAMETERS
        ----------
        query:
            Causal query

        RETURNS
        -------
        DAGModel:
            Modified model for query.
        """
        raise NotImplementedError("Do-calculus not implemented yet.")

    def export_graphviz(self) -> str:
        """Export to graphviz."""
        dot = graphviz.Digraph()
        for node in self.ordered_nodes:
            dot.node(node, shape="rectangle")

        for edge in self.dag.edges():
            dot.edge(*edge)

        return dot


class AdditiveSciPyDistError(BaseEstimator, TransformerMixin):
    """Additive observation error using SciPy distribution."""

    def __init__(self, dist):
        """
        PARAMETERS
        ----------
        dist: scipy.stats.rv_<any>
            SciPy distribution.
        """
        self.dist = dist

    def fit(self, X, y):
        # Compute residuals
        residuals = y - X

        # Fit distribuion on errors
        self.dist.fit(residuals)

        return self

    def transform(self, X):
        return X + self.dist.rvs(size=X.size)


class MultiplicativeSciPyDistError(BaseEstimator, TransformerMixin):
    """Multiplicative observation error using SciPy distribution."""

    def __init__(self, dist):
        """
        PARAMETERS
        ----------
        dist: scipy.stats.rv_<any>
            SciPy distribution.

        RETURNS
        -------
        self: object
            Self
        """
        self.dist = dist

    def fit(self, X: NDArray, y: NDArray):
        # Compute quotient
        quotient = y / X

        # Fit distribuion on quotients
        self.dist.fit(quotient)

        return self

    def transform(self, X: NDArray):
        return X * self.dist.rvs(size=X.size)


class StatsmodelsAPI(BaseEstimator):
    """Wrapper around Statsmodels API."""

    def __init__(self, sm_model, sm_params=None) -> NoReturn:
        """
        PARAMETERS
        ----------
        sm_model:
            statsmodels model class.
        sm_params:
            Parameters for SM model.
        """
        self.sm_model = sm_model
        self.sm_params = sm_params

    def fit(self, X: NDArray, y: NDArray):
        if sm_params is None:
            self._model = sm_model(y, X)
        else:
            self._model = sm_model(y, X, **self.sm_params)

        self.model_results = self._model.fit()

        return self

    def predict(self, X: NDArray) -> Any:
        return self.model_results.predict(X)
