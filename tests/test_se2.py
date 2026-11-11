import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from my_gbp import FactorGraph, NdimGaussian
from my_gbp.manifold import SE2


def finite_jacobian(function, size=3, step=1e-6):
    return np.column_stack([
        (np.asarray(function(direction)) - np.asarray(function(-direction))) / (2 * step)
        for direction in np.eye(size) * step
    ])


class SE2GeometryTests(unittest.TestCase):
    def setUp(self):
        self.space = SE2()
        output = patch('my_gbp.gbp.save_history')
        output.start()
        self.addCleanup(output.stop)

    def test_exp_log_against_matrix_exponential(self):
        rng = np.random.default_rng(13)
        angles = [0., 1e-14, -1e-14, 1e-8, -1e-8, 0.0999999, -0.1000001,
                  0.7, -1.4, np.pi - 1e-9, -np.pi + 1e-9,
                  np.pi, -np.pi, 4.5, -5.5, 2 * np.pi]
        angles.extend(rng.uniform(-np.pi + 0.01, np.pi - 0.01, 80))
        for theta in angles:
            with self.subTest(theta=theta):
                delta = np.r_[rng.normal(size=2), theta]
                state = self.space.exp(delta)
                np.testing.assert_allclose(state, expm(self.space.hat(delta)), atol=1e-13)
                principal = self.space.log(state)
                self.assertLessEqual(abs(principal[2]), np.pi)
                np.testing.assert_allclose(self.space.exp(principal), state, atol=1e-13)
                if abs(theta) < np.pi - 1e-7:
                    np.testing.assert_allclose(principal, delta, atol=1e-13)
                np.testing.assert_array_equal(state[2], [0., 0., 1.])
                np.testing.assert_allclose(state[:2, :2].T @ state[:2, :2], np.eye(2), atol=1e-14)

    def test_group_operations_adjoint_and_vector_shapes(self):
        rng = np.random.default_rng(17)
        for _ in range(25):
            delta = rng.normal(size=3)
            state = self.space.exp(delta)
            other = self.space.exp(rng.normal(size=3))
            np.testing.assert_allclose(self.space.compose(state, self.space.inverse(state)), np.eye(3), atol=1e-13)
            np.testing.assert_allclose(self.space.inverse(state), np.linalg.inv(state), atol=1e-13)
            np.testing.assert_allclose(self.space.compose(state, other), state @ other, atol=1e-13)
            np.testing.assert_array_equal(self.space.vee(self.space.hat(delta.reshape(3, 1))), delta)
            np.testing.assert_allclose(self.space.retract(state, self.space.local(state, other)), other, atol=1e-13)
            np.testing.assert_allclose(
                self.space.hat(self.space.adjoint(state) @ delta),
                state @ self.space.hat(delta) @ self.space.inverse(state), atol=1e-13,
            )
            self.assertEqual(self.space.local(state, other).shape, (3,))
            self.assertEqual(self.space.retract(state, delta.reshape(3, 1)).shape, (3, 3))
        np.testing.assert_array_equal(self.space.identity(), np.eye(3))

    def test_right_jacobian_against_finite_differences(self):
        rng = np.random.default_rng(19)
        increments = [np.zeros(3), np.array([1., -2., 0.]), np.array([0.2, 0.4, 1e-12])]
        increments.extend(np.r_[rng.normal(size=2), theta] for theta in rng.uniform(-3., 3., 30))
        for delta in increments:
            state = self.space.exp(delta)
            actual = finite_jacobian(lambda eps: self.space.local(state, self.space.exp(delta + eps)))
            np.testing.assert_allclose(self.space.right_jacobian(delta), actual, atol=2e-8)
            np.testing.assert_allclose(
                self.space.right_jacobian(delta) @ self.space.right_jacobian_inverse(delta),
                np.eye(3), atol=1e-12,
            )

    def test_retract_and_local_jacobians(self):
        state = self.space.exp(np.array([0.8, -0.3, 0.5]))
        for delta in (np.zeros(3), np.array([0.2, -0.4, 0.]), np.array([0.1, 0.3, -0.7])):
            result = self.space.retract(state, delta)
            state_jac, delta_jac = self.space.retract_jacobians(state, delta)
            np.testing.assert_allclose(state_jac, finite_jacobian(
                lambda eps: self.space.local(result, self.space.retract(self.space.retract(state, eps), delta))), atol=2e-8)
            np.testing.assert_allclose(delta_jac, finite_jacobian(
                lambda eps: self.space.local(result, self.space.retract(state, delta + eps))), atol=2e-8)
            reference_jac, result_jac = self.space.local_jacobians(state, result)
            np.testing.assert_allclose(reference_jac, finite_jacobian(
                lambda eps: self.space.local(self.space.retract(state, eps), result)), atol=2e-8)
            np.testing.assert_allclose(result_jac, finite_jacobian(
                lambda eps: self.space.local(state, self.space.retract(result, eps))), atol=2e-8)

    def test_gaussian_rebase_and_rank_deficient_messages(self):
        reference = self.space.exp(np.array([0.4, -0.2, 0.3]))
        other = self.space.retract(reference, np.array([0.05, 0.02, -0.04]))
        precision = np.array([[3., 0.2, 0.1], [0.2, 2., -0.1], [0.1, -0.1, 1.]])
        eta = np.array([[0.2], [-0.1], [0.05]])
        gaussian = NdimGaussian(3, eta, precision, manifold=self.space, reference=reference)
        converted = gaussian.rebase(other)
        offset = self.space.local(reference, other).reshape(3, 1)
        jac = finite_jacobian(lambda eps: self.space.local(reference, self.space.retract(other, eps)))
        np.testing.assert_allclose(converted.lam, jac.T @ precision @ jac, atol=2e-8)
        np.testing.assert_allclose(converted.eta, jac.T @ (eta - precision @ offset), atol=2e-8)
        np.testing.assert_array_equal(gaussian.reference, reference)
        np.testing.assert_array_equal(gaussian.eta, eta)
        np.testing.assert_allclose(gaussian.rebase(reference).lam, precision, atol=1e-13)
        np.testing.assert_allclose(gaussian.rebase(reference).eta, eta, atol=1e-13)
        for matrix, rank in ((np.zeros((3, 3)), 0), (np.diag([1., 0., 0.]), 1)):
            message = NdimGaussian(3, lam=matrix, manifold=self.space, reference=reference).rebase(other)
            self.assertEqual(np.linalg.matrix_rank(message.lam, tol=1e-10), rank)

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'JAX is optional')
    def test_jax_ad_jit_and_zero_angle_derivatives(self):
        import jax
        import jax.numpy as jnp

        with jax.enable_x64():
            space = SE2(xp=jnp)
            zero = jnp.zeros(3)
            for differentiation in (jax.jacfwd, jax.jacrev):
                actual = differentiation(lambda delta: space.log(space.exp(delta)))(zero)
                np.testing.assert_allclose(actual, np.eye(3), atol=1e-12)
            hessian = jax.hessian(lambda delta: jnp.sum(space.exp(delta) ** 2))(zero)
            np.testing.assert_allclose(hessian, np.diag([2., 2., 0.]), atol=1e-12)
            for delta in (zero, jnp.array([0.3, -0.2, 1e-12]), jnp.array([0.3, -0.2, 0.7])):
                state = space.exp(delta)
                actual = jax.jacfwd(lambda eps: space.local(state, space.exp(delta + eps)))(zero)
                np.testing.assert_allclose(actual, space.right_jacobian(delta), atol=1e-12)
                np.testing.assert_allclose(jax.jit(space.exp)(delta), self.space.exp(np.asarray(delta)), atol=1e-13)
                np.testing.assert_allclose(jax.jit(space.log)(state), delta, atol=1e-13)
                np.testing.assert_allclose(jax.jit(space.right_jacobian)(delta), space.right_jacobian(delta), atol=1e-13)
            state = space.exp(jnp.array([0.6, -0.3, 0.5]))
            delta = jnp.array([0.1, 0.2, -0.4])
            result = space.retract(state, delta)
            expected = space.retract_jacobians(state, delta)
            actual = (
                jax.jacfwd(lambda eps: space.local(result, space.retract(space.retract(state, eps), delta)))(zero),
                jax.jacfwd(lambda eps: space.local(result, space.retract(state, delta + eps)))(zero),
            )
            for value, target in zip(actual, expected):
                np.testing.assert_allclose(value, target, atol=1e-12)
            expected = space.local_jacobians(state, result)
            actual = (
                jax.jacfwd(lambda eps: space.local(space.retract(state, eps), result))(zero),
                jax.jacfwd(lambda eps: space.local(state, space.retract(result, eps)))(zero),
            )
            for value, target in zip(actual, expected):
                np.testing.assert_allclose(value, target, atol=1e-12)
            batch = jnp.array([[0., 0., 0.], [0.2, -0.1, 0.3]])
            np.testing.assert_allclose(jax.jit(jax.vmap(space.exp))(batch),
                                       np.array([self.space.exp(row) for row in np.asarray(batch)]), atol=1e-13)

    def test_variable_factor_connection(self):
        def solve(mode, xp=np):
            graph = FactorGraph()
            var = graph.add_variable(3, np.eye(3), np.eye(3), manifold=SE2(xp=xp))
            kwargs = {'jac_fn': lambda states: np.hstack((states[0][:2, :2], np.zeros((2, 1))))} if mode == 'analytic' else {}
            graph.add_factor([var], lambda states: states[0][:2, 2],
                             np.array([1., -0.5]), np.eye(2), jacobian=mode, **kwargs)
            converged, iterations, history = graph.solve(history=True)
            self.assertTrue(converged)
            self.assertEqual(len(history), iterations)
            expected = np.eye(3)
            expected[:2, 2] = [0.5, -0.25]
            np.testing.assert_allclose(var.mean, expected, atol=1e-8)
            np.testing.assert_array_equal(var.prior.reference, np.eye(3))
            np.testing.assert_array_equal(var.belief.reference, var.mean)
            self.assertEqual(np.linalg.matrix_rank(graph.factors[0].messages[0].lam, tol=1e-8), 2)
            self.assertEqual(history[-1]['means'][0].shape, (3, 3))
            self.assertEqual(history[-1]['covs'][0].shape, (3, 3))

        for mode in ('numerical', 'analytic'):
            with self.subTest(mode=mode):
                solve(mode)
        if importlib.util.find_spec('jax') is not None:
            import jax
            import jax.numpy as jnp
            with jax.enable_x64():
                solve('auto', xp=jnp)


if __name__ == '__main__':
    unittest.main()
