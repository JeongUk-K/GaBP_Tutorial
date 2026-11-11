import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from my_gbp.manifold import SE2


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'cooperative localization' / 'UTIAS dataset' / 'build_graph.py'
)
SPEC = importlib.util.spec_from_file_location('utias_build_graph', MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def example_data():
    robots = {}
    for robot_id, x in ((1, 0.), (2, 2.)):
        robots[robot_id] = {
            'groundtruth': np.array([[0., x, 0., 0.], [1., x, 0., 0.]]),
            'odometry': np.array([
                [-.1, .2, 0.], [.3, .2, 0.], [.8, .2, 0.], [1.1, .2, 0.],
            ]),
            'measurements': np.empty((0, 4)),
        }
    bearing = np.arctan2(2., 1.9)
    distance = np.hypot(1.9, 2.)
    robots[1]['measurements'] = np.array([
        [.1, 14., 2., 0.],
        [.4, 72., distance, bearing],
        [.6, 72., distance, bearing],
    ])
    robots[2]['measurements'] = np.array([[.9, 5., 2., np.pi]])
    return {
        'node_times': np.array([0., .5, 1.]),
        'robots': robots,
        'landmarks': np.array([[6., 2., 2., .01, .02]]),
        'barcodes': np.array([[1, 5], [2, 14], [6, 72]]),
    }


class UTIASGraphTests(unittest.TestCase):
    def setUp(self):
        output = patch('my_gbp.gbp.save_history')
        output.start()
        self.addCleanup(output.stop)

    def build(self, data):
        # Covariances here are test fixtures, not experiment defaults.
        return builder.build_graph(
            data, prior_cov=np.diag([.2, .3, .1]),
            odometry_cov=np.diag([.02, .03, .01]),
            measurement_cov=np.diag([.1, .02]),
        )

    def test_piecewise_odometry_and_constant_turn(self):
        space = SE2()
        odometry = np.array([[0., 1., 0.], [1., 0., 1.], [2., 0., 1.]])
        expected = space.exp([.75, 0., 0.]) @ space.exp([0., 0., .75])
        np.testing.assert_allclose(builder.integrate_odometry(odometry, .25, 1.75), expected)
        odometry[:, 1:] = [1., .4]
        np.testing.assert_allclose(
            builder.integrate_odometry(odometry, .25, 1.75),
            space.exp(np.array([1., 0., .4]) * 1.5), atol=1e-14,
        )

    def test_initial_heading_interpolation_across_pi(self):
        groundtruth = np.array([
            [0., 0., 0., np.pi - .1], [1., 2., 4., -np.pi + .1],
        ])
        pose = builder.initial_pose(groundtruth, .5)
        np.testing.assert_allclose(pose[:2, 2], [1., 2.])
        np.testing.assert_allclose(pose[:2, :2], -np.eye(2), atol=1e-14)

    def test_odometry_jacobian_in_right_tangents(self):
        space = SE2()
        states = (space.exp([.3, -.2, .4]), space.exp([1., .8, -.5]))
        motion = space.exp([.4, .1, .2])
        numerical = np.column_stack([
            (builder.odometry_residual(
                (space.retract(states[0], eps[:3]), space.retract(states[1], eps[3:])), motion,
            ) - builder.odometry_residual(
                (space.retract(states[0], -eps[:3]), space.retract(states[1], -eps[3:])), motion,
            )) / 2e-6
            for eps in np.eye(6) * 1e-6
        ])
        np.testing.assert_allclose(builder.odometry_jacobian(states, motion), numerical, atol=2e-8)

    def test_pose_residual_and_global_frame_invariance(self):
        space = SE2()
        motion = space.exp([1., 0., 0.])
        states = (np.eye(3), space.exp([1., .7, 0.]))
        expected = np.array([0., .7, 0.])
        np.testing.assert_allclose(builder.odometry_residual(states, motion), expected)
        self.assertEqual(builder.odometry_jacobian(states, motion).shape, (3, 6))
        global_frame = space.exp([2., -3., 1.2])
        transformed = tuple(global_frame @ state for state in states)
        np.testing.assert_allclose(builder.odometry_residual(transformed, motion), expected, atol=1e-14)

    def test_pose_covariance_is_used_directly(self):
        data = example_data()
        data['node_times'] = np.array([0., .25, 1.])
        q = np.array([[.02, .003, 0.], [.003, .03, .001], [0., .001, .01]])
        graph, _ = builder.build_graph(
            data, prior_cov=.1 * np.eye(3), odometry_cov=q, measurement_cov=.1 * np.eye(2),
        )
        for factor in graph.factors[:4]:
            np.testing.assert_array_equal(factor.gauss_noise_var, q)

    def test_pose_covariance_must_be_positive_definite(self):
        for q in (None, np.eye(2), np.diag([.1, 0., .1])):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                builder.build_graph(
                    example_data(), prior_cov=.1 * np.eye(3),
                    odometry_cov=q, measurement_cov=.1 * np.eye(2),
                )

    def test_range_bearing_jacobians_for_robot_and_landmark(self):
        space = SE2()
        observer = space.exp([.2, -.4, .3])
        measurement = np.array([1., np.pi - .02])
        target = space.exp([1.4, .5, -.4])
        for landmark in (None, np.array([1.4, .5])):
            size = 6 if landmark is None else 3
            states = (observer, target) if landmark is None else (observer,)

            def evaluate(delta):
                moved = (space.retract(observer, delta[:3]),)
                if landmark is None:
                    moved += (space.retract(target, delta[3:]),)
                return builder.range_bearing(moved, measurement, landmark)

            columns = []
            for eps in np.eye(size) * 1e-6:
                difference = evaluate(eps) - evaluate(-eps)
                difference[1] = builder.wrap_angle(difference[1])
                columns.append(difference / 2e-6)
            np.testing.assert_allclose(
                builder.range_bearing_jacobian(states, measurement, landmark),
                np.column_stack(columns), atol=2e-8,
            )

    def test_priors_association_and_solve_initialization(self):
        graph, variables = self.build(example_data())
        self.assertEqual(graph.n_var_nodes, 6)
        self.assertEqual(set(variables), {'robots'})
        self.assertTrue(all(node.dofs == 3 for node in graph.var_nodes))
        self.assertEqual(graph.n_factor_nodes, 8)
        for factor in graph.factors[:4]:
            self.assertEqual(factor.measurement.shape, (3, 1))
            self.assertEqual(factor.gauss_noise_var.shape, (3, 3))
        for nodes in variables['robots'].values():
            np.testing.assert_allclose(nodes[0].prior.lam, np.diag([5., 1 / .3, 10.]))
            for node in nodes[1:]:
                np.testing.assert_array_equal(node.prior.lam, np.zeros((3, 3)))
            np.testing.assert_allclose(nodes[-1].mean[:2, 2] - nodes[0].mean[:2, 2], [.2, 0.])
        expected = [
            [variables['robots'][1][0], variables['robots'][2][0]],
            [variables['robots'][1][1]],
            [variables['robots'][1][1]],
            [variables['robots'][2][2], variables['robots'][1][2]],
        ]
        for factor, nodes in zip(graph.factors[4:], expected):
            self.assertEqual(factor.adj_var_nodes, nodes)
        for factor in graph.factors[5:7]:
            np.testing.assert_array_equal(factor.args[1], [2., 2.])
            self.assertEqual(factor.jac_fn(tuple(v.mean for v in factor.adj_var_nodes), *factor.args).shape,
                             (2, 3))
        _, n_iters, history = graph.solve(max_iters=2, history=True)
        self.assertGreaterEqual(n_iters, 1)
        for frame in history:
            for mean, covariance in zip(frame['means'], frame['covs']):
                self.assertTrue(np.all(np.isfinite(mean)))
                self.assertTrue(np.all(np.isfinite(covariance)))
                self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_fixed_landmarks_have_no_prior_or_mutable_state(self):
        data = example_data()
        data['landmarks'][:, 3:] = 0.
        graph, _ = self.build(data)
        landmark_factors = [f for f in graph.factors if len(f.args) == 2 and f.args[1] is not None]
        self.assertEqual(len(landmark_factors), 2)
        data['landmarks'][:, 1:3] = [100., 200.]
        graph.solve(max_iters=2)
        for factor in landmark_factors:
            np.testing.assert_array_equal(factor.args[1], [2., 2.])
            np.testing.assert_array_equal(factor.gauss_noise_var, np.diag([.1, .02]))

    def test_unknown_barcode_is_reported(self):
        data = example_data()
        data['robots'][1]['measurements'][0, 1] = 43
        with self.assertRaisesRegex(ValueError, 'Unknown measurement barcode 43'):
            self.build(data)

    def test_measurement_variants_partition_observations_and_keep_dynamics(self):
        data = example_data()
        full, full_variables = self.build(data)
        original_measurements = {
            rid: robot['measurements'].copy() for rid, robot in data['robots'].items()
        }
        kwargs = dict(prior_cov=np.diag([.2, .3, .1]),
                      odometry_cov=np.diag([.02, .03, .01]),
                      measurement_cov=np.diag([.1, .02]))
        for name, landmark_only, edges in (
            ('build_graph_landmark', True, 10),
            ('build_graph_robot_meas', False, 12),
        ):
            with self.subTest(module=name):
                spec = importlib.util.spec_from_file_location(name, MODULE_PATH.with_name(name + '.py'))
                module = importlib.util.module_from_spec(spec)
                with patch.object(sys, 'path', [str(MODULE_PATH.parent), *sys.path]):
                    spec.loader.exec_module(module)
                graph, variables = module.build_graph(data, **kwargs)
                self.assertEqual(graph.n_var_nodes, full.n_var_nodes)
                self.assertEqual(graph.n_factor_nodes, 6)
                self.assertEqual(graph.n_edges, edges)
                expected = [f for f in full.factors
                            if f.meas_fn.__name__ == 'odometry_residual'
                            or ((f.args[1] is not None) == landmark_only)]
                for actual, reference in zip(graph.factors, expected):
                    self.assertEqual(actual.adj_vIDs, reference.adj_vIDs)
                    self.assertEqual(actual.meas_fn.__name__, reference.meas_fn.__name__)
                    np.testing.assert_array_equal(actual.measurement, reference.measurement)
                    np.testing.assert_array_equal(actual.gauss_noise_var, reference.gauss_noise_var)
                    np.testing.assert_allclose(actual.factor.eta, reference.factor.eta)
                    np.testing.assert_allclose(actual.factor.lam, reference.factor.lam)
                for rid, nodes in variables['robots'].items():
                    for actual, reference in zip(nodes, full_variables['robots'][rid]):
                        np.testing.assert_array_equal(actual.mean, reference.mean)
                        np.testing.assert_array_equal(actual.prior.lam, reference.prior.lam)
                graph.solve(max_iters=2)
                for node in graph.var_nodes:
                    self.assertTrue(np.isfinite(node.mean).all())
                    self.assertGreater(np.linalg.eigvalsh(node.cov).min(), 0.)
        for rid, robot in data['robots'].items():
            np.testing.assert_array_equal(robot['measurements'], original_measurements[rid])

    def test_measurement_type_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'measurement_type'):
            builder.build_graph(example_data(), prior_cov=np.eye(3),
                                odometry_cov=np.eye(3), measurement_cov=np.eye(2),
                                measurement_type='invalid')

    def test_unknown_barcode_is_reported_for_each_measurement_type(self):
        data = example_data()
        data['robots'][1]['measurements'][0, 1] = 43
        for mode in ('landmark', 'robot'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Unknown measurement barcode 43'):
                builder.build_graph(data, prior_cov=np.eye(3), odometry_cov=np.eye(3),
                                    measurement_cov=np.eye(2), measurement_type=mode)


if __name__ == '__main__':
    unittest.main()
