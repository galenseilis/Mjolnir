from gplearn.genetic import SymbolicRegressor
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sympy

from . import transforms

try:
    # MAPIE < 1.0
    from mapie.regression import MapieRegressor as _LegacyMapieRegressor
except ImportError:  # pragma: no cover - depends on installed MAPIE
    _LegacyMapieRegressor = None
    from mapie.regression import CrossConformalRegressor as _CrossConformalRegressor


def _make_conformal_model(estimator):
    """Build a conformal regressor for the installed MAPIE version.

    Supports both the legacy ``MapieRegressor`` API (MAPIE < 1.0) and the
    current ``CrossConformalRegressor`` API (MAPIE >= 1.0).
    """
    if _LegacyMapieRegressor is not None:
        return _LegacyMapieRegressor(estimator=estimator, method="plus", cv=5)
    return _CrossConformalRegressor(
        estimator=estimator,
        method="plus",
        cv=5,
        confidence_level=[0.95, 0.05],
    )


def _fit_conformal(conformal_model, X, y):
    """Fit a conformal regressor across MAPIE versions."""
    if hasattr(conformal_model, "fit_conformalize"):
        return conformal_model.fit_conformalize(X, y)
    return conformal_model.fit(X, y)


def _predict_conformal(conformal_model, X):
    """Return ``(point_predictions, prediction_intervals)`` across MAPIE versions."""
    if hasattr(conformal_model, "predict_interval"):
        return conformal_model.predict_interval(X)
    return conformal_model.predict(X, alpha=[0.05, 0.95])



# TODO: Default multiple imputation of input data for missing data. Follow DAG structure
# TODO: Parameter to use Optuna to hyperparameter tune.
# TODO: SymPy-based methods for mathematical analysis of model.
# TODO: Compatible conformal prediction.
class Mjolnir(transforms.DAGModel):
    """
    Mjolnir is a useful default model for interpretable causal ML.

    "Let me tell you about my grandfather's hammer, which is rightfully mine." - Bodger

    Parameters
    ----------
    dag : networkx.DiGraph
        The directed acyclic graph (DAG) representing the causal relationships.
    dagm_params : dict, optional
        Parameters to pass to the parent DAGModel, if any (default is None).
    gp_params : dict, optional
        Additional parameters for the SymbolicRegressor models (default is None).
    sympy_converter : dict, optional
        A dictionary that maps mathematical operation strings to SymPy functions (default is None).

    Attributes
    ----------
    ordered_nodes : list
        A list of nodes in topological order, indicating the order in which nodes should be processed.
    sympy_converter : dict
        A dictionary that maps mathematical operation strings to SymPy functions.

    Methods
    -------
    fit(X)
        Fit the Mjolnir model to the input data X.
    _get_sympy_exprs()
        Extract SymPy expressions from GPLearn instances.
    derivative(method='analytic')
        Compute the derivative of the model.
    gradient()
        Compute the gradient of the model.
    divergence()
        Compute the divergence of the model.
    jacobian()
        Compute the Jacobian matrix of the model.
    hessian()
        Compute the Hessian matrix of the model.
    inv(X)
        Compute the inverse using the Jacobian.
    pinv(X)
        Compute the Moore-Penrose pseudoinverse using the Jacobian.
    fit_approx_inverse(X)
        Fit a Mjolnir model on the reverse DAG to create an approximated inverse function.
    conformal_fit(X)
        Fit conformal models for each node in the DAG.
    conformal_predict(X)
        Make predictions using the fitted conformal models.

    See Also
    --------
    transforms.DAGModel : Parent class for directed acyclic graph models.
    """

    def __init__(self, dag, dagm_params=None, gp_params=None, sympy_converter=None):
        self.ordered_nodes = list(nx.topological_sort(dag))

        # Prepare models with or without additional parameters.
        models = {}
        for node in self.ordered_nodes:
            input_nodes = list(dag.predecessors(node))

            if input_nodes:
                input_nodes = sorted(
                    set(self.ordered_nodes) & set(input_nodes),
                    key=self.ordered_nodes.index,
                )

                if gp_params is None:
                    models[node] = SymbolicRegressor(feature_names=input_nodes)
                else:
                    models[node] = SymbolicRegressor(
                        feature_names=input_nodes, **gp_params
                    )

        # Pass along any params to DAGModel
        if dagm_params is None:
            super().__init__(dag, models)
        else:
            super().__init__(dag, models, **dagm_params)

        if sympy_converter is None:
            self.sympy_converter = {
                "sub": lambda x, y: x - y,
                "div": lambda x, y: x / y,
                "mul": lambda x, y: x * y,
                "add": lambda x, y: x + y,
                "neg": lambda x: -x,
                "pow": lambda x, y: x**y,
            }
        else:
            self.sympy_converter = sympy_converter

    def fit(self, X):
        """
        Fit the Mjolnir model to the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for training the model.
        """
        super().fit(X)
        self._get_sympy_exprs()

    def _get_sympy_exprs(self):
        """Extract SymPy expressions from GPLearn instances.

        Notes
        -----
        This method extracts SymPy expressions from GPLearn instances and stores them in `symb_model_exprs`.

        https://stackoverflow.com/questions/48404263/how-to-export-the-output-of-gplearn-as-a-sympy-expression-or-some-other-readable
        """
        self.symb_model_exprs = {}
        for node in self.ordered_nodes:
            if list(self.dag.predecessors(node)):
                self.symb_model_exprs[node] = sympy.sympify(
                    str(self.models[node]), locals=self.sympy_converter
                )

        # TODO: Handle MAPIE regression instances

    def derivative(self, method="analytic"):
        """
        Compute the derivative of the model.

        Parameters
        ----------
        method : {'analytic', 'spectral'}, optional
            The method to use for computing the derivative (default is 'analytic').
        """
        raise NotImplementedError

        if method == "analytic":
            ...
        elif method == "spectral":
            # https://www.youtube.com/watch?v=reievpVoSsY
            # https://www.youtube.com/watch?v=SBYQ3bprKy0
            ...
        else:
            raise NotImplementedError(
                f'Method {method} is not supported. Only available methods are "analytic" and "spectral."'
            )

    def gradient(self):
        """
        Compute the gradient of the model.
        """
        raise NotImplementedError

    def divergence(self):
        """
        Compute the divergence of the model.
        """
        raise NotImplementedError

    def jacobian(self):
        """
        Compute the Jacobian matrix of the model.
        """
        raise NotImplementedError

    def hessian(self):
        """
        Compute the Hessian matrix of the model.
        """
        raise NotImplementedError

    def inv(self, X):
        """
        Compute the inverse using the Jacobian.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for computing the inverse.

        Notes
        -----
        This function is not guaranteed to work and depends on the properties of the DAG and learned functions.
        """
        ...

    def pinv(self, X):
        """
        Compute the Moore-Penrose pseudoinverse using the Jacobian.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for computing the pseudoinverse.
        """
        raise NotImplementedError

    def fit_approx_inverse(self, X):
        """
        Fit a Mjolnir model on the reverse DAG to create an approximated inverse function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for fitting the approximated inverse.

        Notes
        -----
        Creates `self.approx_inverse`, which acts as an approximated inverse function.
        """
        dag_inv = nx.DiGraph.reverse(self.dag)
        self.inv_mjolnir = Mjolnir(dag_inv)
        self.inv_mjolnir.fit(X)
        self.approx_inverse = self.inv_mjolnir.predict

    def conformal_fit(self, X):
        """
        Fit conformal models for each node in the DAG.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for fitting conformal models.

        Returns
        -------
        self : Mjolnir
            The updated Mjolnir model with fitted conformal models.

        Notes
        -----
        Overwrites symbolic expressions.

        See Also
        --------
        MapieRegressor : The estimator used for fitting conformal models.
        """
        self.conformal_models = {}
        fitted_conformal_predictions = {}

        for node in self.ordered_nodes:
            input_nodes = list(self.dag.predecessors(node))

            if self.verbose and input_nodes:
                print(f"Conformal fitting {node} as a function of {input_nodes}.")

            if input_nodes:
                input_nodes = sorted(
                    set(self.ordered_nodes) & set(input_nodes),
                    key=self.ordered_nodes.index,
                )
                self.conformal_models[node] = _make_conformal_model(self.models[node])

                input_data = np.column_stack(
                    [
                        fitted_conformal_predictions[in_node]
                        if in_node in fitted_conformal_predictions
                        else X[in_node]
                        for in_node in input_nodes
                    ]
                )
                # Fit model
                _fit_conformal(self.conformal_models[node], input_data, X[node])

                # Store predictions
                fitted_conformal_predictions[node], _ = _predict_conformal(
                    self.conformal_models[node], input_data
                )

        # TODO: Overwrite symbolic expressions
        return self

    def conformal_predict(self, X):
        """
        Make predictions using the fitted conformal models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for making predictions.

        Returns
        -------
        predictions : dict
            Predictions for each node in the DAG.
        conformal_predictions : dict
            Conformal predictions for each node in the DAG.

        Notes
        -----
        Nicely formats the output with pandas.

        See Also
        --------
        MapieRegressor : The estimator used for fitting conformal models.
        """
        conformal_predictions = {}
        predictions = {}

        for node in self.ordered_nodes:
            input_nodes = list(self.dag.predecessors(node))

            if input_nodes:
                input_nodes = sorted(
                    set(self.ordered_nodes) & set(input_nodes),
                    key=self.ordered_nodes.index,
                )
                input_data = np.column_stack(
                    [
                        predictions[in_node] if in_node in predictions else X[in_node]
                        for in_node in input_nodes
                    ]
                )

                predictions[node], conformal_predictions[node] = _predict_conformal(
                    self.conformal_models[node], input_data
                )

        return (
            predictions,
            conformal_predictions,
        )  # TODO: Nicely format output with pandas

