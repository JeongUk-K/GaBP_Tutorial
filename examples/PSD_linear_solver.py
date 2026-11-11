"""Solve a symmetric, strictly diagonally dominant SPD system Ax = b.

Each scalar x_i has prior mean b_i/r_i and variance 1/r_i, where
r_i = A_ii - sum_{j != i} |A_ij|. Pairwise factors supply the remaining
quadratic terms, so the graph objective equals 0.5*x.T@A@x - b.T@x + const.
"""

from time import perf_counter

import numpy as np

from my_gbp import FactorGraph
from my_gbp.manifold import Euclidean


N = 30


def coupling(x, weight, sign):
    # Slicing preserves a one-element output for both NumPy and JAX inputs.
    return weight * (x[:1] + sign * x[1:2])

def coupling_jac(x, weight, sign):
    return weight * np.array([[1.0, sign]])




def build_graph(A, b, jacobian='analytic', jac_fn=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    off_diagonal = A.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    remainder = np.diag(A) - np.sum(np.abs(off_diagonal), axis=1)

    factor_graph = FactorGraph(nonlinear_factors=False)
    if jacobian == 'analytic' and jac_fn is None:
        jac_fn = coupling_jac

    # Add one scalar Euclidean variable per unknown.
    nodes = []
    for i in range(len(b)):
        node = factor_graph.add_variable(
            dofs=1,
            prior_mean=np.array([b[i] / remainder[i]]),
            prior_cov=np.array([[1.0 / remainder[i]]]),
            manifold=Euclidean(1),
        )
        nodes.append(node)

    # Add one pairwise factor for each nonzero off-diagonal entry.
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            if A[i, j] == 0:
                continue
            factor_graph.add_factor(
                [nodes[i], nodes[j]],
                meas_fn=coupling,
                measurement=np.array([0.0]),
                noise_cov=np.eye(1),
                jacobian=jacobian,
                jac_fn=jac_fn,
                args=(np.sqrt(abs(A[i, j])), np.sign(A[i, j])),
            )

    return factor_graph


if __name__ == '__main__':
    import jax

    rng = np.random.default_rng(0)
    upper = np.triu(rng.uniform(-0.1, 0.1, size=(N, N)), k=1)
    A = upper + upper.T
    # Positive diagonal remainder makes this symmetric matrix positive definite.
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1.0)
    b = rng.standard_normal(N)

    # Match NumPy precision for the AD Jacobian.
    with jax.enable_x64():
        graph_ad = build_graph(A, b, jacobian='auto')
        start = perf_counter()
        converged_ad, n_iters_ad, history_ad = graph_ad.solve(max_iters=200, tol=1e-10)
        elapsed_ad = perf_counter() - start

    graph_numerical = build_graph(A, b, jacobian='numerical')
    start = perf_counter()
    converged_numerical, n_iters_numerical, history_numerical = graph_numerical.solve(max_iters=200, tol=1e-10)
    elapsed_numerical = perf_counter() - start

    graph_analytic = build_graph(A, b, jacobian='analytic', jac_fn=coupling_jac)
    start = perf_counter()
    converged_analytic, n_iters_analytic, history_analytic = graph_analytic.solve(max_iters=200, tol=1e-10)
    elapsed_analytic = perf_counter() - start

    # NumPy is only the reference; GBP computes x_gbp through message passing.
    start = perf_counter()
    x_reference = np.linalg.solve(A, b)
    numpy_elapsed = perf_counter() - start

    print(f'A shape={A.shape}  nodes={N}  dofs per node=1')
    print(f'pairwise factors={len(graph_analytic.factors)}')
    print(f'NumPy solve time = {numpy_elapsed * 1000:.6f} ms')
    for name, graph, converged, iters, elapsed in (
        ('AD', graph_ad, converged_ad, n_iters_ad, elapsed_ad),
        ('Numerical', graph_numerical, converged_numerical, n_iters_numerical, elapsed_numerical),
        ('Analytic', graph_analytic, converged_analytic, n_iters_analytic, elapsed_analytic),
    ):
        x_gbp = graph.get_means()
        error = np.linalg.norm(x_gbp - x_reference)
        residual = np.linalg.norm(A @ x_gbp - b)
        print(f'{name}: converged={converged}  iters={iters}')
        print(f'  solve time = {elapsed * 1000:.6f} ms (excluding graph construction)')
        print(f'  ||x_GBP - x_NumPy|| = {error:.3e}  ||A @ x_GBP - b|| = {residual:.3e}')
        assert converged, 'GBP did not converge within the iteration limit.'
        assert error < 1e-8, 'GBP solution did not match the NumPy reference.'
        assert residual < 1e-8, 'GBP solution did not satisfy Ax = b.'
