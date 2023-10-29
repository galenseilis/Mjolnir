# Introduction

Mjolnir is a machine learning model enhanced by causal assumptions and uncertainty quantification.

![](assets/thor.jpg)

# Quick Start

```python
from mjolnir import Mjolnir
import networkx as nx

data = ...
dag = nx.DiGraph(...)

model = Mjolnir(dag, data)
model.fit(data)
```
