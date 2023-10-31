
# Defining Your Causal Assumptions

The first step you should take to use Mjölnir effectively is to encode your causal assumptions into a DAG. This is done with a networkx `DiGraph`. Here is an example of a collider $X_1 \rightarrow X_2 \leftarrow X_3$. 

```python
import networkx as nx

dag = nx.DiGraph()

dag.add_edge('X1', 'X2')
dag.add_edge('X3', 'X2') 
```

# Training a Model

If you're familiar with the Scikit-Learn API, this should be familiar. Simply import the class `Mjolnir`. For the sake of example I am going to ignore the DAG we defined earlier, and generate one randomly from scratch.

```python
from mjolnir import datasets
from mjolnir import Mjolnir

dag, data = datasets.make_dag_regression(n=10)

model = Mjolnir(dag)

model.conformal_fit(data)
```

# Symbolically Analyzing the Model

Mjölnir provides some built-in funtionality for performing symbolic math on the model discovered in the symbolic regression.

For starters, let's display the symbolic expressions learned by model.

```python
print(model._get_sympy_exprs())
```

If you would like to further analyze the model but you're unfamiliar with SymPy or computer algebra, I recommend looking at the [Introductory Tutorial](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html) for SymPy to get you started.
