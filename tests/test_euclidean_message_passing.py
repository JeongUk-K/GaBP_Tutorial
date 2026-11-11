import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from my_gbp import FactorGraph


def batch_posterior(means, covs, measurements):
    sizes = [len(mean) for mean in means]
    starts = np.cumsum([0] + sizes)
    precision = np.zeros((starts[-1], starts[-1]))
    information = np.zeros(starts[-1])
    for i, (mean, cov) in enumerate(zip(means, covs)):
        block = slice(starts[i], starts[i + 1])
        prior_precision = np.linalg.inv(cov)
        precision[block, block] = prior_precision
        information[block] = prior_precision @ mean
    for indices, model, observation, noise in measurements:
        columns = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in indices])
        matrix = np.zeros((len(observation), starts[-1]))
        matrix[:, columns] = model
        precision += matrix.T @ np.linalg.solve(noise, matrix)
        information += matrix.T @ np.linalg.solve(noise, observation)
    return np.linalg.solve(precision, information), np.linalg.inv(precision), starts


class EuclideanMessagePassingTests(unittest.TestCase):
    def setUp(self):
        output = patch('my_gbp.gbp.save_history')
        output.start()
        self.addCleanup(output.stop)

    def assert_batch_solution(self, graph, means, covs, measurements):
        expected_mean, expected_cov, starts = batch_posterior(means, covs, measurements)
        converged, _, _ = graph.solve(tol=1e-12)
        self.assertTrue(converged)
        np.testing.assert_allclose(graph.get_means(), expected_mean, atol=1e-11)
        for i, var in enumerate(graph.var_nodes):
            block = slice(starts[i], starts[i + 1])
            np.testing.assert_allclose(var.cov, expected_cov[block, block], atol=1e-11)
            np.testing.assert_array_equal(var.belief.reference, var.mean)
            np.testing.assert_array_equal(var.prior.reference.ravel(), means[i])
            np.testing.assert_array_equal(var.prior.eta, np.zeros((var.dofs, 1)))

    def test_mixed_dimension_tree_and_diagnostic_linearisation(self):
        means = [np.array([0.3]), np.array([-0.2, 0.7]), np.array([1.1])]
        covs = [np.array([[1.2]]), np.array([[1.3, 0.2], [0.2, 0.8]]), np.array([[0.9]])]
        measurements = [
            ([0, 1], np.array([[1., -1., 0.], [0., 0.5, 1.]]),
             np.array([0.4, -0.8]), np.array([[0.7, 0.1], [0.1, 1.4]])),
            ([1, 2], np.array([[1., -0.5, -1.]]), np.array([1.2]), np.array([[0.8]])),
        ]
        graph = FactorGraph(nonlinear_factors=False)
        variables = [graph.add_variable(len(m), m, c) for m, c in zip(means, covs)]
        for indices, model, observation, noise in measurements:
            graph.add_factor(
                [variables[i] for i in indices], lambda x, matrix: matrix @ x,
                observation, noise, jacobian='analytic',
                jac_fn=lambda x, matrix: matrix, args=(model,),
            )
        self.assert_batch_solution(graph, means, covs, measurements)

        factor = graph.factors[0]
        previous = (factor.linpoint.copy(), factor.factor.eta.copy(), factor.factor.lam.copy())
        factor.compute_factor(factor.linpoint + 0.1, update_self=False)
        for actual, expected in zip((factor.linpoint, factor.factor.eta, factor.factor.lam), previous):
            np.testing.assert_array_equal(actual, expected)

    def test_three_variables_one_factor_with_rank_deficient_messages(self):
        means = [np.array([1., 2.]), np.array([0.5, 1.5]), np.array([1.5, 2.5])]
        covs = [np.array([[1., 0.5], [0.5, 2.]]), np.array([[0.5, 0.2], [0.2, 1.]]),
                np.array([[1.5, 0.7], [0.7, 3.]])]
        model = np.array([[1., 0., 2., 0., -1., 0.]])
        measurements = [([0, 1, 2], model, np.array([1.7]), np.array([[0.4]]))]
        graph = FactorGraph(nonlinear_factors=False)
        variables = [graph.add_variable(2, m, c) for m, c in zip(means, covs)]
        graph.add_factor(variables, lambda x: model @ x, np.array([1.7]), np.array([[0.4]]),
                         jacobian='analytic', jac_fn=lambda x: model)
        self.assert_batch_solution(graph, means, covs, measurements)
        for message in graph.factors[0].messages:
            self.assertEqual(np.linalg.matrix_rank(message.lam, tol=1e-10), 1)

    def test_synchronous_order_and_message_reference_invariance(self):
        def build(damping):
            graph = FactorGraph(nonlinear_factors=False, eta_damping=damping)
            variables = [graph.add_variable(1, [m], np.array([[1.]])) for m in (0.2, -0.4, 1.1)]
            for i, j, z in ((0, 1, 1.), (1, 2, -0.3), (2, 0, 0.4)):
                graph.add_factor(
                    [variables[i], variables[j]], lambda x: np.array([x[1] - x[0]]),
                    np.array([z]), np.array([[2.]]), jacobian='analytic',
                    jac_fn=lambda x: np.array([[-1., 1.]]),
                )
            return graph

        for damping in (0., 0.4):
            with self.subTest(damping=damping):
                first, second = build(damping), build(damping)
                second.factors.reverse()
                for i, var in enumerate(second.var_nodes):
                    var.prior = var.prior.rebase(np.array([[10. + i]]))
                for factor in second.factors:
                    factor.messages = [message.rebase(np.array([[-5.]]))
                                       for message in factor.messages]
                first.solve(max_iters=12, tol=0., history=True)
                second.solve(max_iters=12, tol=0., history=True)
                for key in ('means', 'covs'):
                    np.testing.assert_allclose(
                        np.array([step[key] for step in first.history]),
                        np.array([step[key] for step in second.history]), atol=5e-12,
                    )

    def test_nonlinear_jacobian_modes_and_independent_history(self):
        modes = ['numerical', 'analytic']
        if importlib.util.find_spec('jax') is not None:
            modes.append('auto')
        expected = np.max(np.roots([2., 0., -7., -1.]).real)
        for mode in modes:
            with self.subTest(mode=mode):
                graph = FactorGraph()
                var = graph.add_variable(1, [1.], np.array([[1.]]))
                if mode == 'auto':
                    import jax.numpy as jnp
                    measurement = lambda x: jnp.square(x[0])
                else:
                    measurement = lambda x: float(x[0] ** 2)
                kwargs = {'jac_fn': lambda x: np.array([[2 * x[0]]])} if mode == 'analytic' else {}
                graph.add_factor([var], measurement, 4., np.array([[1.]]), jacobian=mode, **kwargs)
                converged, iterations, history = graph.solve(history=True)
                self.assertTrue(converged)
                np.testing.assert_allclose(var.mean.item(), expected, atol=2e-6)
                self.assertEqual(len(history), iterations)
                self.assertEqual([step['iter'] for step in history], list(range(1, iterations + 1)))
                self.assertFalse(np.shares_memory(history[-1]['means'][0], var.mean))
                self.assertFalse(np.shares_memory(history[-1]['covs'][0], var.cov))
                snapshot = history[-1]['means'][0].copy()
                graph.solve(max_iters=0)
                self.assertEqual(graph.history, [])
                np.testing.assert_array_equal(history[-1]['means'][0], snapshot)

    def test_tuple_callback_preserves_matrix_state_shape(self):
        # A test chart separates state shape from tangent dimension without SE2.
        class DiagonalChart:
            tangent_dim = 2

            def retract(self, state, delta):
                return np.diag(np.diag(state) + np.asarray(delta).ravel())

            def local(self, reference, state):
                return np.diag(state) - np.diag(reference)

            def retract_jacobians(self, state, delta):
                return np.eye(2), np.eye(2)

            def local_jacobians(self, reference, state):
                return -np.eye(2), np.eye(2)

        for mode in ('numerical', 'analytic'):
            with self.subTest(mode=mode):
                graph = FactorGraph()
                var = graph.add_variable(2, np.diag([1., 2.]), np.eye(2), manifold=DiagonalChart())
                kwargs = {'jac_fn': lambda states: np.ones((1, 2))} if mode == 'analytic' else {}
                graph.add_factor(
                    [var], lambda states: np.array([np.trace(states[0])]),
                    np.array([5.]), np.array([[1.]]), jacobian=mode, **kwargs,
                )
                self.assertTrue(graph.solve(history=True)[0])
                self.assertIsInstance(graph.get_means(), tuple)
                self.assertEqual(var.mean.shape, (2, 2))
                np.testing.assert_allclose(np.diag(var.mean), [5 / 3, 8 / 3], atol=1e-8)
                np.testing.assert_allclose(var.cov, np.eye(2) - np.ones((2, 2)) / 3, atol=1e-8)
                self.assertEqual(graph.history[-1]['means'][0].shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
