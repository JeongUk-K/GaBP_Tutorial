import numpy as np

from my_gbp import FactorGraph
from my_gbp.manifold import Euclidean


N = 30
DIM = 3
M = 5
NOISE_STD = 1.0


def displacement(x):
    return x[DIM:] - x[:DIM]


def displacement_jacobian(x):
    return np.hstack((-np.eye(DIM), np.eye(DIM)))


def build_graph(jacobian, jac_fn=None):
    # Use the same positions, priors and measurements for every Jacobian mode.
    rng = np.random.RandomState(0)
    ground_truth = rng.rand(N, DIM) * 10
    prior_cov = 3 * np.eye(DIM)
    measurement_cov = NOISE_STD ** 2 * np.eye(DIM)

    factor_graph = FactorGraph(nonlinear_factors=False)

    # Add one Euclidean variable for each position.
    nodes = []
    for position in ground_truth:
        node = factor_graph.add_variable(
            dofs=DIM,
            prior_mean=position + rng.randn(DIM),
            prior_cov=prior_cov,
            manifold=Euclidean(DIM),
        )
        nodes.append(node)

    # Add each nearest-neighbour displacement factor once.
    edges = set()
    for i in range(N):
        neighbours = np.argsort(np.linalg.norm(ground_truth - ground_truth[i], axis=1))
        for j in neighbours[1:M + 1]:
            edge = tuple(sorted((i, int(j))))
            if edge in edges:
                continue
            edges.add(edge)
            a, b = edge
            measurement = (ground_truth[b] - ground_truth[a]
                           + rng.normal(0, NOISE_STD, DIM))
            factor_graph.add_factor(
                [nodes[a], nodes[b]],
                meas_fn=displacement,
                measurement=measurement,
                noise_cov=measurement_cov,
                jacobian=jacobian,
                jac_fn=jac_fn,
            )

    return factor_graph


if __name__ == '__main__':
    graph_ad = build_graph(jacobian='auto')
    converged_ad, n_iters_ad, history_ad = graph_ad.solve()

    graph_numerical = build_graph(jacobian='numerical')
    converged_numerical, n_iters_numerical, history_numerical = graph_numerical.solve()

    graph_analytic = build_graph(jacobian='analytic', jac_fn=displacement_jacobian)
    converged_analytic, n_iters_analytic, history_analytic = graph_analytic.solve()

    print('AD:', converged_ad, 'iters:', n_iters_ad)
    print('Numerical:', converged_numerical, 'iters:', n_iters_numerical)
    print('Analytic:', converged_analytic, 'iters:', n_iters_analytic)
