"""Read-only numerical audit of the current UTIAS configuration.

Run from the repository root. Does not call main() or overwrite history.csv.
"""
import argparse
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cooperative localization' / 'UTIAS dataset'))
from my_gbp import FactorGraph
from my_gbp.gaussian import NdimGaussian


def load_configuration():
    path = ROOT / 'cooperative localization' / 'UTIAS dataset' / 'UTIAS dataset.py'
    spec = importlib.util.spec_from_file_location('utias_configuration', path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    data = config.prepare_data(config.load_data(config.DATASET_DIR), config.hz, config.duration,
                               start_offset=config.start_offset)
    graph, _ = config.build_graph(data, prior_cov=config.prior_cov,
                                 odometry_cov=config.odometry_cov,
                                 measurement_cov=config.measurement_cov)
    graph.update_all_beliefs()
    return config, graph


def jacobian(factor, states):
    return np.asarray(factor.jac_fn(states, *factor.args))


def audit_derivatives(graph):
    result = {}
    for step in (1e-4, 1e-5, 1e-6):
        worst = {}
        for factor in graph.factors:
            states = tuple(v.mean.copy() for v in factor.adj_var_nodes)
            analytic = jacobian(factor, states)
            numerical = np.column_stack([
                (factor._residual_delta(d, states) - factor._residual_delta(-d, states)) / (2 * step)
                for d in np.eye(factor.dofs_conditional_vars) * step
            ])
            name = factor.meas_fn.__name__
            error = float(np.max(np.abs(analytic - numerical)))
            if error > worst.get(name, {}).get('max_abs_error', -1):
                worst[name] = {'max_abs_error': error, 'factor_id': factor.factorID}
        result[str(step)] = worst
    return result


def energy_gradient_check(graph):
    """Compare the assembled -b with finite differences of the actual energy."""
    _, b = assemble(graph)
    states = graph.get_states()
    directions = np.random.default_rng(0).normal(size=(5, len(b)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    errors = []
    step = 1e-6
    for direction in directions:
        energies = []
        for sign in (1, -1):
            for v, state, d in zip(graph.var_nodes, states, direction.reshape(-1, 3)):
                v.mu = v.manifold.retract(state, sign * step * d)
            energies.append(graph.energy())
        errors.append(abs((energies[0] - energies[1]) / (2 * step) + b @ direction))
    for v, state in zip(graph.var_nodes, states):
        v.mu = state
    return {'max_directional_absolute_error': float(max(errors)), 'step': step}


def assemble(graph):
    size = sum(v.dofs for v in graph.var_nodes)
    H, b = np.zeros((size, size)), np.zeros(size)
    for v in graph.var_nodes:
        g = v.prior.rebase(v.mean)
        ix = slice(3 * v.variableID, 3 * v.variableID + 3)
        H[ix, ix] += g.lam
        b[ix] += g.eta.ravel()
    for f in graph.factors:
        eta, lam = f.compute_factor(update_self=False)
        ix = np.concatenate([np.arange(3 * v.variableID, 3 * v.variableID + 3)
                             for v in f.adj_var_nodes])
        H[np.ix_(ix, ix)] += lam
        b[ix] += eta.ravel()
    return H, b


def frozen_euclidean(graph):
    """An exactly linear graph in ONE fixed initial chart, with copied messages."""
    out = FactorGraph(nonlinear_factors=False)
    for v in graph.var_nodes:
        node = out.add_variable(3, np.zeros(3), np.eye(3))
        g = v.prior.rebase(v.mean)
        node.prior = NdimGaussian(3, g.eta.copy(), g.lam.copy())
    for f in graph.factors:
        states = tuple(v.mean.copy() for v in f.adj_var_nodes)
        J = jacobian(f, states)
        residual = f._residual_delta(np.zeros(f.dofs_conditional_vars), states)
        new = out.add_factor([out.var_nodes[v.variableID] for v in f.adj_var_nodes],
                             lambda x, A: A @ x, -residual, f.gauss_noise_var,
                             jacobian='analytic', jac_fn=lambda x, A: A, args=(J,))
        new.messages = [NdimGaussian(v.dofs, m.rebase(v.mean).eta.copy(),
                                     m.rebase(v.mean).lam.copy())
                        for v, m in zip(f.adj_var_nodes, f.messages)]
    return out


def iterate(graph, count, checkpoints, *, local_relin=True):
    rows = [{'iter': 0, 'energy': graph.energy(), 'step': 0.}]
    best, best_states = rows[0]['energy'], graph.get_states()
    for i in range(1, count + 1):
        previous = graph.get_states()
        graph.synchronous_iteration(local_relin=local_relin)
        _, step = graph.converged_mean_change(previous, 0.)
        energy = graph.energy()
        if energy < best:
            best, best_states = energy, graph.get_states()
        rows.append({'iter': i, 'energy': energy, 'step': step})
        if i in checkpoints:
            print(json.dumps(rows[-1]), flush=True)
        if not np.isfinite(energy) or energy > 1e12:
            break
    minimum = min(rows, key=lambda r: r['energy'])
    return {'minimum': minimum, 'last': rows[-1], 'trace': rows}, best_states


def centralized_gn(graph, count=100):
    rows = []
    for i in range(count):
        H, b = assemble(graph)
        delta = np.linalg.solve(H, b).reshape(-1, 3)
        states, energy = graph.get_states(), graph.energy()
        alpha = 1.
        for _ in range(25):
            for v, state, d in zip(graph.var_nodes, states, delta):
                v.mu = v.manifold.retract(state, alpha * d)
            new_energy = graph.energy()
            if new_energy <= energy:
                break
            alpha *= .5
        else:
            for v, state in zip(graph.var_nodes, states):
                v.mu = state
            break
        rows.append({'iter': i + 1, 'energy': new_energy, 'alpha': alpha,
                     'gradient_inf': float(np.max(np.abs(b)))})
        if np.max(np.abs(b)) < 1e-7:
            break
    _, b = assemble(graph)
    return {'trace': rows, 'energy': graph.energy(),
            'gradient_inf': float(np.max(np.abs(b)))}


def message_spectrum(graph):
    """Exact eta iteration matrix at the frozen graph's current precisions.

    The UTIAS graph has only unary and binary factors, all of dimension 3.
    A spectral radius above one proves local instability of this eta iteration
    once precisions have settled; SPD of the joint precision alone is insufficient.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import eigs
    slots = {(f.factorID, i): k for k, (f, i) in enumerate(
        (f, i) for f in graph.factors for i in range(len(f.adj_var_nodes)))}
    A = lil_matrix((3 * len(slots), 3 * len(slots)))
    for f in graph.factors:
        if len(f.adj_var_nodes) == 1:
            continue
        if len(f.adj_var_nodes) != 2:
            raise ValueError('Spectrum audit supports unary and binary factors only')
        for i in (0, 1):
            j = 1 - i
            other = f.adj_var_nodes[j]
            cavity = other.compute_message(f, reference=np.zeros((3, 1)))
            L = f.factor.lam
            row = slice(3 * slots[f.factorID, i], 3 * slots[f.factorID, i] + 3)
            B = -np.linalg.solve((L[3*j:3*j+3, 3*j:3*j+3] + cavity.lam).T,
                                 L[3*i:3*i+3, 3*j:3*j+3].T).T
            for incoming in other.adj_factors:
                if incoming is f:
                    continue
                k = slots[incoming.factorID, incoming.adj_vIDs.index(other.variableID)]
                A[row, 3*k:3*k+3] = B
    eigenvalues = eigs(A.tocsr(), k=6, which='LM', return_eigenvectors=False,
                      v0=np.ones(A.shape[0]), tol=1e-10)
    return {'spectral_radius': float(np.max(np.abs(eigenvalues))),
            'leading_eigenvalues': [[float(z.real), float(z.imag)] for z in eigenvalues]}


def damping_schedule_check():
    graph = FactorGraph(eta_damping=.8, num_undamped_iters=1)
    v = graph.add_variable(1, [1.], np.eye(1))
    f = graph.add_factor([v], lambda x: x ** 2, [4.], np.eye(1),
                         jacobian='analytic', jac_fn=lambda x: np.array([[2 * x[0]]]))
    rows = []
    for i in range(4):
        graph.synchronous_iteration()
        rows.append({'iter': i + 1, 'iters_since_relin': f.iters_since_relin,
                     'effective_eta_damping': f.eta_damping})
    return {'requested_eta_damping': .8, 'num_undamped_iters': 1, 'trace': rows}


def checker_check():
    from my_gbp.differentiation import check_jac
    output = io.StringIO()
    with redirect_stdout(output):
        check_jac(lambda x: np.array([[-1.]]), np.array([1.]), lambda x: x)
    return {'supplied_jacobian': -1., 'correct_jacobian': 1.,
            'checker_output': output.getvalue().strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT / 'utias_solver_audit.json')
    args = parser.parse_args()
    config, graph = load_configuration()
    results = {'configuration': {'hz': config.hz, 'duration': config.duration,
                                'start_offset': config.start_offset,
                                'variables': graph.n_var_nodes, 'factors': graph.n_factor_nodes,
                                'prior_cov': config.prior_cov.tolist(),
                                'odometry_cov': config.odometry_cov.tolist(),
                                'measurement_cov': config.measurement_cov.tolist()},
               'initial_derivatives': audit_derivatives(graph),
               'initial_energy_gradient': energy_gradient_check(graph),
               'damping_schedule_probe': damping_schedule_check(),
               'jacobian_checker_probe': checker_check()}
    print(json.dumps(results), flush=True)
    H, b = assemble(graph)
    exact = np.linalg.solve(H, b)
    results['initial_linear_system'] = {'condition': float(np.linalg.cond(H)),
                                       'min_eigenvalue': float(np.linalg.eigvalsh(H)[0]),
                                       'solve_residual_inf': float(np.max(np.abs(H @ exact - b)))}
    frozen = frozen_euclidean(graph)
    print('BASELINE', flush=True)
    baseline = copy.deepcopy(graph, {id(np): np})
    results['baseline'], best_states = iterate(baseline, args.iters, {1, 5, 10, 20, 40, 60, 100, 200})
    best_graph = copy.deepcopy(graph, {id(np): np})
    for v, state in zip(best_graph.var_nodes, best_states):
        v.mu = state
    results['best_derivatives'] = audit_derivatives(best_graph)
    print('FROZEN EUCLIDEAN', flush=True)
    results['frozen'], _ = iterate(frozen, args.iters, {1, 5, 10, 20, 40, 60, 100, 200})
    results['frozen']['solution_error_inf'] = float(np.max(np.abs(frozen.get_means() - exact)))
    results['frozen']['eta_iteration_spectrum'] = message_spectrum(frozen)
    print('FIXED MANIFOLD FACTORS', flush=True)
    results['fixed_manifold'], _ = iterate(
        copy.deepcopy(graph, {id(np): np}), min(args.iters, 100),
        {1, 20, 60, 100}, local_relin=False,
    )
    print('CENTRAL GN', flush=True)
    central = copy.deepcopy(graph, {id(np): np})
    results['central_gn'] = centralized_gn(central)
    results['central_derivatives'] = audit_derivatives(central)
    results['central_energy_gradient'] = energy_gradient_check(central)
    summary = {k: v for k, v in results.items()
               if k not in ('baseline', 'frozen', 'fixed_manifold', 'central_gn')}
    summary['central_gn'] = {k: v for k, v in results['central_gn'].items() if k != 'trace'}
    print(json.dumps(summary), flush=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
