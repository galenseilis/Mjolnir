
# Causal Assumptions

Machine learning algorithms are like a black box. Things go in, things go out, but it is not usually possible to understand what is happening within the box. A neural network is an example of a machine learning model. It can use something similar to fuzzy logic to weight different paths of causality in a computation graph, but the topology of the computation graph does not have to match the causal assumptions in the causal model.

Mjölnir is designed with the assignment of causal assumptions in mind. A causal directed acyclic graph is provided by the use which is used to constrain the space of models that can be learned from the data.

# Symbolic Regression

Symbolic regression involves estimating a symbolic expression that approximates the data well. Mjölnir uses genetic programming to balance accuracy and simplicity of the models estimated. Thus Mjölnir provides a means of model discovery.

# Confomal Prediction

Conformal prediction is a form of uncertainty quantification that assumes relatively little about the data generating process. Anastasios N. Angelopoulos and Stephen Bates have produce some resources for learning conformal prediction. A paper they authored [*A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*](https://arxiv.org/abs/2107.07511). They also provide the following videos:

<iframe width="560" height="315" src="https://www.youtube.com/embed/nql000Lu_iE?si=iWCiWH3XG1asGhiZ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/TRx4a2u-j7M?si=pgyI02bltC_vqtlN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/37HKrmA5gJE?si=-xeRsOQEhDtTcbKJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

# Computer Algebra System

The symbolic expressions estimated in through symbolic regression become automatically available for mathematical analysis with SymPy. Mjölnir also comes with some methods for common mathematical queries such as locating optima and inferring asymptotic behaviour.
