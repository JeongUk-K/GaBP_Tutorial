# GaBP Tutorial

A **Gaussian Belief Propagation (GaBP)** Factor Graph Optimization (FGO) Tutorial for
state estimation, control and planning.

## Install

Install the package (editable) from the repo root. This pulls in `numpy` and
`scipy` and puts `gbp` on the import path so the examples run from anywhere:

```
pip install -e .
```

Optional — the JAX autodiff Jacobian backend:

```
pip install -e .[jax]
```

## Usage

Build a graph declaratively, provide the measurement function of each factor, and
solve:

```python
import numpy as np
from gbp import FactorGraph

graph = FactorGraph(nonlinear_factors=False)

x0 = graph.add_variable(dofs=2, prior_mean=[0, 0], prior_cov=0.01 * np.eye(2))
x1 = graph.add_variable(dofs=2, prior_mean=[1, 1])

# h(x) = x1 - x0, observed displacement z, isotropic noise std 0.1
graph.add_factor([x0, x1],
                 meas_fn=lambda x: x[2:] - x[:2],
                 measurement=np.array([1.0, 0.0]),
                 noise_std=0.1,
                 jacobian='numerical')

converged, iters = graph.solve(max_iters=200, tol=1e-6)
print(x1.mean, x1.cov)
```

### Jacobians

Each factor linearises its measurement function; the Jacobian is obtained one of
three ways via the `jacobian` argument of `add_factor`:

| mode | how the Jacobian is computed |
|------|------------------------------|
| `'analytic'`  | user-supplied `jac_fn(x, *args)` |
| `'numerical'` | finite differences (no extra dependency) |
| `'auto'`      | automatic differentiation with JAX |

## Core API

- `FactorGraph.add_variable(dofs, prior_mean=None, prior_cov=None)` — add a variable node.
- `FactorGraph.add_factor(variables, meas_fn, measurement, noise_std, jacobian=..., jac_fn=None, loss=None)` — add a factor and wire its edges.
- `FactorGraph.solve(max_iters, tol)` — run synchronous GBP until convergence; returns `(converged, n_iters)`.
- `VariableNode.mean` / `VariableNode.cov` — current belief estimate.

## Test

A self-contained graph-operation test builds an N-dimensional pose graph and checks
that GBP converges to the exact batch MAP solution:

```
python examples/pose_graph.py
```
