"""
Simple graph-operation test for the general-purpose GBP solver.

Builds an N-dimensional pose graph: variable nodes sit at random locations and are
connected to their nearest neighbours by linear displacement factors
    h(x_i, x_j) = x_j - x_i.
GBP is run with the declarative API and the resulting belief means are checked
against the exact batch MAP solution. This exercises the whole pipeline
(add_variable / add_factor / message passing / solve) on a purely general problem
with no SLAM-specific code.

Run with the repo root on the path:
    PYTHONPATH=. python examples/pose_graph.py
"""

import numpy as np

from gbp import FactorGraph

np.random.seed(0)

N = 30          # number of variable nodes
DIM = 3         # dimensionality of each node
M = 5           # nearest neighbours each node connects to
NOISE_STD = 1.0  # std of the Gaussian measurement noise


def displacement(x):
    """Measurement function h([x_i, x_j]) = x_j - x_i."""
    d = len(x) // 2
    return x[d:] - x[:d]


def displacement_jac(x, *args):
    """Analytic Jacobian of the displacement measurement function."""
    d = len(x) // 2
    return np.hstack((-np.eye(d), np.eye(d)))


def main():
    gt = np.random.rand(N, DIM) * 10  # ground-truth node positions

    graph = FactorGraph(nonlinear_factors=False)

    # One variable per node, initialised near (but not at) ground truth.
    nodes = [
        graph.add_variable(DIM,
                           prior_mean=gt[i] + np.random.randn(DIM),
                           prior_cov=3 * np.eye(DIM))
        for i in range(N)
    ]

    # Connect each node to its M nearest neighbours with a displacement factor.
    edges = set()
    for i in range(N):
        order = np.argsort(np.linalg.norm(gt - gt[i], axis=1))
        for j in order[1:M + 1]:          # order[0] is the node itself
            edge = tuple(sorted((i, int(j))))
            if edge in edges:
                continue
            edges.add(edge)
            a, b = edge
            z = gt[b] - gt[a] + np.random.normal(0, NOISE_STD, DIM)
            graph.add_factor([nodes[a], nodes[b]], displacement, z, NOISE_STD,
                             jacobian='analytic', jac_fn=displacement_jac)

    print(f'nodes={N}  dim={DIM}  factors={len(graph.factors)}')

    # Exact reference solution (linear problem => GBP should reach it).
    mu_batch, _ = graph.joint_distribution_cov()

    converged, iters = graph.solve(max_iters=200, tol=1e-6)
    err = np.linalg.norm(graph.get_means() - mu_batch)

    print(f'converged={converged} in {iters} iters  //  '
          f'||GBP means - batch MAP|| = {err:.3e}')
    assert err < 1e-3, 'GBP means did not match the batch MAP solution'
    print('OK: graph converged to the batch MAP solution.')


if __name__ == '__main__':
    main()
