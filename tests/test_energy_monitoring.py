import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from my_gbp import FactorGraph
from my_gbp.manifold import SE2


class EnergyMonitoringTests(unittest.TestCase):
    def test_energy_uses_current_state_and_correlated_covariances(self):
        graph = FactorGraph()
        prior_cov = np.array([[2., .5], [.5, 1.]])
        noise_cov = np.array([[.5, .1], [.1, .8]])
        variable = graph.add_variable(2, [1., -1.], prior_cov)
        factor = graph.add_factor(
            [variable], lambda x: np.array([x[0] ** 2, x[0] + x[1]]),
            [1., 1.], noise_cov, jacobian='analytic',
            jac_fn=lambda x: np.array([[2 * x[0], 0.], [1., 1.]]),
        )
        reference = factor.linpoint.copy()
        parameters = (factor.factor.eta.copy(), factor.factor.lam.copy())
        variable.mu = np.array([[2.], [.5]])

        residual = np.array([3., 1.5])
        prior_error = np.array([1., 1.5])
        whitened_factor = np.linalg.solve(np.linalg.cholesky(noise_cov), residual)
        whitened_prior = np.linalg.solve(np.linalg.cholesky(prior_cov), prior_error)
        expected = .5 * (whitened_factor @ whitened_factor + whitened_prior @ whitened_prior)
        np.testing.assert_array_equal(factor.compute_residual(), residual[:, None])
        self.assertIsInstance(graph.energy(), float)
        self.assertAlmostEqual(graph.energy(), expected)

        variable.prior.eta = np.array([[.2], [-.4]])
        self.assertAlmostEqual(graph.energy(), expected - np.dot([.2, -.4], prior_error))
        np.testing.assert_array_equal(factor.linpoint, reference)
        np.testing.assert_array_equal(factor.factor.eta, parameters[0])
        np.testing.assert_array_equal(factor.factor.lam, parameters[1])

    def test_se2_prior_uses_local_angle_across_wrap(self):
        space = SE2()
        graph = FactorGraph()
        variable = graph.add_variable(
            3, space.exp([0., 0., np.pi - .02]), np.diag([.4, .6, .1]),
            manifold=space,
        )
        graph.add_factor(
            [variable], lambda states: states[0][:2, 2], [1., -2.], np.diag([.5, .8]),
            jacobian='analytic',
            jac_fn=lambda states: np.column_stack((states[0][:2, :2], np.zeros(2))),
        )
        variable.mu = space.exp([0., 0., -np.pi + .03])
        self.assertAlmostEqual(graph.energy(), 3.5 + .5 * .05 ** 2 / .1)

    def linear_graph(self):
        graph = FactorGraph(nonlinear_factors=False)
        variable = graph.add_variable(1, [0.], np.eye(1))
        graph.add_factor(
            [variable], lambda x: x, [3.], np.eye(1), jacobian='analytic',
            jac_fn=lambda x: np.eye(1),
        )
        return graph

    def test_monitoring_prints_actual_energy_and_step_separately(self):
        graph = self.linear_graph()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(graph, 'energy', wraps=graph.energy) as energy, \
                patch('my_gbp.gbp.perf_counter', side_effect=[100., 101.25]):
            graph.solve(max_iters=1, monitoring=True)
        energy.assert_called_once_with()
        self.assertIn('Elapsed time: 1.250 seconds.', output.getvalue())
        self.assertIn('Step norm: 1.500000', output.getvalue())
        self.assertIn('Energy: 2.250000', output.getvalue())

    def test_monitoring_off_does_not_evaluate_energy(self):
        graph = self.linear_graph()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(graph, 'energy', side_effect=AssertionError('Unexpected energy evaluation')):
            graph.solve(max_iters=2, monitoring=False)
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
