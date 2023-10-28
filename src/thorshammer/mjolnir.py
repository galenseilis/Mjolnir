from gplearn.genetic import SymbolicRegressor
from mapie.regression import MapieRegressor
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sympy

import transforms

# TODO: Default multiple imputation of input data for missing data. Follow DAG structure
# TODO: Parameter to use Optuna to hyperparameter tune.
# TODO: SymPy-based methods for mathematical analysis of model.
# TODO: Compatible conformal prediction.
class Mjolnir(transforms.DAGModel):
    '''Mjolnir is a useful default model for interpretable causal ML.

    "Let me tell you about my grandfather's hammer, which is rightfully mine." - Bodger
    '''

    def __init__(self, dag, dagm_params=None, gp_params=None, sympy_converter=None):

        self.ordered_nodes = list(nx.topological_sort(dag))

        # Prepare models with or without additional parameters.
        models = {}
        for node in self.ordered_nodes:
            input_nodes = list(dag.predecessors(node))

            if input_nodes:
                input_nodes = sorted(set(self.ordered_nodes) & set(input_nodes), key = self.ordered_nodes.index)

                if gp_params is None:
                    models[node] = SymbolicRegressor(feature_names=input_nodes)
                else:
                    models[node] = SymbolicRegressor(feature_names=input_nodes, **gp_params)

        # Pass along any params to DAGModel
        if dagm_params is None:
            super().__init__(dag, models)
        else:
            super().__init__(dag, models, **dagm_params)

        if sympy_converter is None:
            self.sympy_converter = {
                'sub': lambda x, y : x - y,
                'div': lambda x, y : x/y,
                'mul': lambda x, y : x*y,
                'add': lambda x, y : x + y,
                'neg': lambda x    : -x,
                'pow': lambda x, y : x**y
                }
        else:
            self.sympy_converter = sympy_converter

    def fit(self, X):
        super().fit(X)
        self._get_sympy_exprs()

    def _get_sympy_exprs(self):
        '''Extract SymPy expressions from GPLearn instances.
        https://stackoverflow.com/questions/48404263/how-to-export-the-output-of-gplearn-as-a-sympy-expression-or-some-other-readable
        '''
        self.symb_model_exprs = {}
        for node in self.ordered_nodes:
            if list(self.dag.predecessors(node)):
                self.symb_model_exprs[node] = sympy.sympify(
                    str(self.models[node]),
                    locals=self.sympy_converter
                    )

        # TODO: Handle MAPIE regression instances

    def derivative(self, method='analytic'):
        raise NotImplementedError

        if method == 'analytic':
            ...
        elif method == 'spectral':
            # https://www.youtube.com/watch?v=reievpVoSsY
            # https://www.youtube.com/watch?v=SBYQ3bprKy0
            ...
        else:
            raise NotImplementedError(f'Method {method} is not supported. Only available methods are "analytic" and "spectral."')


    def gradient(self):
        raise NotImplementedError

    def divergence(self):
        raise NotImplementedError

    def jacobian(self):
        raise NotImplementedError

    def hessian(self):
        raise NotImplementedError

    def inv(self, X):
        '''Compute inverse using the Jacobian.

        This function is not guaranteed to work. It depends
        on the properties of the DAG and learned functions. In
        some cases the Jacobian may not be defined. In other cases
        the Jacobian may not be invertible.
        '''
        ...

    def pinv(self, X):
        '''Compute Moore-Penrose pseudoinverse using the Jacobian.'''
        raise NotImplementedError

    def fit_approx_inverse(self, X):
        '''Fit a Mjolnir model on the reverse DAG.

        Creates `self.approx_inverse` which acts as an
        approximated inverse function.
        '''
        dag_inv = nx.DigGraph.reverse(self.dag)
        self.inv_mjolnir = Mjolnir(dag_inv, X)
        self.approx_inverse = self.inv_mjolnir.predict

    def conformal_fit(self, X):

        self.conformal_models = {}
        fitted_conformal_predictions = {}

        for node in self.ordered_nodes:
            input_nodes = list(self.dag.predecessors(node))

            if self.verbose and input_nodes:
                
                print(f'Conformal fitting {node} as a function of {input_nodes}.')

            if input_nodes:
                self.conformal_models[node] = MapieRegressor(
                    estimator=self.models[node], method='plus', cv=5
                    )

                input_data = np.column_stack(
                    [
                        fitted_conformal_predictions[in_node] if in_node in fitted_conformal_predictions
                        else X[in_node]
                        for in_node in input_nodes
                        ]
                        )
                # Fit model
                self.conformal_models[node].fit(input_data, X[node])

                # Store predictions
                fitted_conformal_predictions[node], _  = self.conformal_models[node].predict(input_data, alpha=[0.05, 0.95])

        # TODO: Overwrite symbolic expressions
        return self

    def conformal_predict(self, X):
        conformal_predictions = {}
        predictions = {}

        for node in self.ordered_nodes:
            input_nodes = list(self.dag.predecessors(node))

            if input_nodes:

                input_data = np.column_stack(
                    [
                        predictions[in_node] if in_node in predictions
                        else X[in_node]
                        for in_node in input_nodes
                        ]
                        )

                predictions[node], conformal_predictions[node]  = self.conformal_models[node].predict(input_data, alpha=[0.05, 0.95])

        return predictions, conformal_predictions # TODO: Nicely format output with pandas

import examples
dag, data = examples.make_dag_regression(n=10)
model = Mjolnir(
    dag,
    dagm_params={'verbose':1},
    gp_params={'generations':2}
    )
model.conformal_fit(data)
##model._get_sympy_exprs()
