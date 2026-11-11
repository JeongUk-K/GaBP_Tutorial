import numpy as np
import jax.numpy as jnp

from my_gbp import FactorGraph
from my_gbp.manifold import Euclidean


import matplotlib.pyplot as plt
if __package__:
    from .utils.plotter import plot_history
else:
    from examples.utils.plotter import plot_history
    

# measurement function and its Jacobian
def distance(x):
    return np.array([np.linalg.norm(x[2:] - x[:2])])


def distance_ad(x):
    return jnp.array([jnp.linalg.norm(x[2:] - x[:2])])


def distance_jacobian(x):
    d = x[2:] - x[:2]
    direction = d / np.linalg.norm(d)
    return np.concatenate((-direction, direction)).reshape(1, 4)


# diffrenent ways to build the factor graph by mesurement function and Jacobian
def build_graph(meas_fn, jacobian, jac_fn=None):
    factor_graph = FactorGraph()

    x1 = factor_graph.add_variable(
        dofs=2,
        prior_mean=np.array([1.0, 2.0]),
        prior_cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        manifold=Euclidean(2)
    )
    x2 = factor_graph.add_variable(
        dofs=2,
        prior_mean=np.array([0.5, 1.5]),
        prior_cov=np.array([[0.5, 0.2], [0.2, 1.0]]),
        manifold=Euclidean(2)
    )
    x3 = factor_graph.add_variable(
        dofs=2,
        prior_mean=np.array([1.5, 2.5]),
        prior_cov=np.array([[1.5, 0.7], [0.7, 3.0]]),
        manifold=Euclidean(2)
    )

    meas_x1_x2 = 2.0
    meas_x2_x3 = 1.0
    meas_cov = 0.1 * np.eye(1)

    factor_graph.add_factor(
        [x1, x2], meas_fn=meas_fn,
        measurement=np.array([meas_x1_x2]), noise_cov=meas_cov,
        jacobian=jacobian, jac_fn=jac_fn,
    )
    factor_graph.add_factor(
        [x2, x3], meas_fn=meas_fn,
        measurement=np.array([meas_x2_x3]), noise_cov=meas_cov,
        jacobian=jacobian, jac_fn=jac_fn,
    )
    return factor_graph


graph_ad = build_graph(distance_ad, jacobian='auto')
converged_ad, n_iters_ad, history_ad = graph_ad.solve(history=True)

graph_numerical = build_graph(distance, jacobian='numerical')
converged_numerical, n_iters_numerical, history_numerical = graph_numerical.solve(history=True)

graph_analytic = build_graph(distance, jacobian='analytic', jac_fn=distance_jacobian)
converged_analytic, n_iters_analytic, history_analytic = graph_analytic.solve(history=True)



fig, animation = plot_history(history_analytic, 'Analytic')
plt.show()
