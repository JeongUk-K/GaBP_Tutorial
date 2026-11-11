import importlib
import importlib.util
import unittest

import numpy as np

from my_gbp.manifold import SE2


@unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'The example uses JAX')
class RangeBearingExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = importlib.import_module('examples.SE(2)_example')

    def test_body_frame_direction_and_wrapped_bearing(self):
        observer = self.example.pose(1., 2., np.pi / 2)
        target = self.example.pose(2., 2., 0.7)
        measurement = np.array([1., -np.pi / 2])
        np.testing.assert_allclose(self.example.range_bearing((observer, target), measurement),
                                   measurement, atol=1e-14)
        for predicted, observed in ((-np.pi + 0.01, np.pi - 0.01),
                                    (np.pi - 0.01, -np.pi + 0.01)):
            target = self.example.pose(np.cos(predicted), np.sin(predicted), 1.)
            measurement = np.array([1., observed])
            error = self.example.range_bearing((np.eye(3), target), measurement) - measurement
            expected = np.arctan2(np.sin(predicted - observed), np.cos(predicted - observed))
            np.testing.assert_allclose(error, [0., expected], atol=1e-14)

    def test_right_increment_jacobian_away_from_angle_cut(self):
        import jax
        import jax.numpy as jnp

        states = (self.example.pose(0.4, -0.2, 0.3), self.example.pose(1.2, 0.7, -0.4))
        measurement = np.array([1.1, 0.2])
        analytic = self.example.range_bearing_jacobian(states, measurement)
        space = SE2()

        def evaluate(delta):
            perturbed = (space.retract(states[0], delta[:3]), space.retract(states[1], delta[3:]))
            return self.example.range_bearing(perturbed, measurement)

        numerical = np.column_stack([(evaluate(eps) - evaluate(-eps)) / 2e-6
                                     for eps in np.eye(6) * 1e-6])
        np.testing.assert_allclose(numerical, analytic, atol=1e-8)
        np.testing.assert_allclose(analytic[:, 5], 0., atol=1e-14)
        with jax.enable_x64():
            space_ad = SE2(xp=jnp)

            def evaluate_ad(delta):
                perturbed = (space_ad.retract(states[0], delta[:3]),
                             space_ad.retract(states[1], delta[3:]))
                return self.example.range_bearing(perturbed, measurement, xp=jnp)

            np.testing.assert_allclose(jax.jacobian(evaluate_ad)(jnp.zeros(6)), analytic, atol=1e-12)

    def test_circular_difference_at_opposite_bearing(self):
        states = (self.example.pose(1., 2., 0.), self.example.pose(0.5, 1.5, np.pi / 4))
        measurement = np.array([2., np.pi / 4])
        numerical = self.example.range_bearing_numerical_jacobian(states, measurement)
        analytic = self.example.range_bearing_jacobian(states, measurement)
        np.testing.assert_allclose(numerical, analytic, atol=1e-8)
        self.assertLess(np.max(np.abs(numerical)), 2.)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None, 'Plotting is optional')
    def test_plot_world_covariance_and_euclidean_compatibility(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from examples.utils.plotter import plot_history

        for state, covariance, expected in (
            (self.example.pose(1., 2., np.pi / 2), np.diag([4., 1., 0.5]), np.diag([1., 4.])),
            (np.array([[1.], [2.]]), np.diag([4., 1.]), np.diag([4., 1.])),
        ):
            history = [{'iter': 1, 'means': [state], 'covs': [covariance]}]
            fig, animation = plot_history(history)
            fig.canvas.draw()
            ellipse = fig.axes[0].patches[0]
            angle = np.radians(ellipse.angle)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            actual = rotation @ np.diag([(ellipse.width / 2) ** 2, (ellipse.height / 2) ** 2]) @ rotation.T
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose(ellipse.center, [1., 2.], atol=1e-14)
            if state.shape == (3, 3):
                arrows = fig.axes[0].collections[0]
                np.testing.assert_allclose(arrows.U, 0., atol=1e-14)
                self.assertGreater(arrows.V[0], 0.)
            else:
                self.assertEqual(len(fig.axes[0].collections), 0)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
