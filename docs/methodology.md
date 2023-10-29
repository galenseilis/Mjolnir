
# Causal Assumptions

Mjolnir is designed with the assignment of causal assumptions in mind. A causal directed acyclic graph is provided by the use which is used to constrain the space of models that can be learned from the data.

# Symbolic Regression

Symbolic regression involves estimating a symbolic expression that approximates the data well. Mjolnir uses genetic programming to balance accuracy and simplicity of the models estimated. Thus Mjolnir provides a means of model discovery.

# Confomal Prediction

Conformal prediction is a form of uncertainty quantification that assumes relatively little about the data generating process.

# Computer Algebra System

The symbolic expressions estimated in through symbolic regression become automatically available for mathematical analysis with SymPy. Mjolnir also comes with some methods for common mathematical queries such as locating optima and inferring asymptotic behaviour.
