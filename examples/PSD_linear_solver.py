"""Solve a 30-dimensional SPD system Ax = b using 30 scalar GBP nodes.

The example generates a symmetric, strictly diagonally dominant matrix. Its
quadratic objective is represented exactly by scalar priors and pairwise factors:

    0.5 * x.T @ A @ x - b.T @ x
    = 0.5 * sum_i r_i * (x_i - b_i / r_i)**2
      + 0.5 * sum_{i<j} |A_ij| * (x_i + sign(A_ij) * x_j)**2
      + constant,

where r_i = A_ii - sum_{j != i} |A_ij| > 0. No extra prior or regularizer is added.
This construction is for strictly diagonally dominant systems, not arbitrary SPD
or singular PSD matrices.

Run after installing the package with `pip install -e .`:
    python examples/PSD_linear_solver.py
"""

from time import perf_counter

import numpy as np

from gbp import FactorGraph


N = 30


def coupling(x, weight, sign):
    """A scalar measurement coupling two scalar variable nodes."""
    return np.array([weight * (x[0] + sign * x[1])])


def coupling_jac(x, weight, sign):
    return weight * np.array([[1.0, sign]])


def build_graph(A, b):
    """Represent a symmetric strictly diagonally dominant Ax = b as GBP."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    off_diagonal = A.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    remainder = np.diag(A) - np.sum(np.abs(off_diagonal), axis=1)

    graph = FactorGraph(nonlinear_factors=False)
    
    nodes = [
        graph.add_variable(
            dofs=1,
            prior_mean=np.array([b[i] / remainder[i]]),
            prior_cov=np.array([[1.0 / remainder[i]]]),
        )
        for i in range(len(b))
    ]

    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            if A[i, j] == 0:
                continue
            graph.add_factor(
                [nodes[i], nodes[j]],
                meas_fn=coupling,
                measurement=np.array([0.0]),
                noise_std=1.0,
                jacobian='analytic',
                jac_fn=coupling_jac,
                args=(np.sqrt(abs(A[i, j])), np.sign(A[i, j])),
            )

    return graph


def main():
    rng = np.random.default_rng(0)
    upper = np.triu(rng.uniform(-0.1, 0.1, size=(N, N)), k=1)
    A = upper + upper.T
    # Positive diagonal remainder makes this symmetric matrix positive definite.
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1.0)
    b = rng.standard_normal(N)

    graph = build_graph(A, b)
    start = perf_counter()
    converged, iters = graph.solve(max_iters=200, tol=1e-10)
    gbp_elapsed = perf_counter() - start
    x_gbp = graph.get_means()

    # NumPy is only the reference; GBP computes x_gbp through message passing.
    start = perf_counter()
    x_reference = np.linalg.solve(A, b)
    numpy_elapsed = perf_counter() - start
    error = np.linalg.norm(x_gbp - x_reference)
    residual = np.linalg.norm(A @ x_gbp - b)

    print(f'A shape={A.shape}  nodes={len(graph.var_nodes)}  dofs per node=1')
    print(f'pairwise factors={len(graph.factors)}')
    print(f'converged={converged} in {iters} iters')
    print(f'GBP solve time = {gbp_elapsed * 1000:.6f} ms (excluding graph construction)')
    print(f'NumPy solve time = {numpy_elapsed * 1000:.6f} ms')
    print(f'||x_GBP - x_NumPy|| = {error:.3e}')
    print(f'||A @ x_GBP - b|| = {residual:.3e}')
    assert converged, 'GBP did not converge within the iteration limit.'
    assert error < 1e-8, 'GBP solution did not match the NumPy reference.'
    assert residual < 1e-8, 'GBP solution did not satisfy Ax = b.'
    print('OK: 30 scalar GBP nodes solved the SPD linear system.')


if __name__ == '__main__':
    main()
