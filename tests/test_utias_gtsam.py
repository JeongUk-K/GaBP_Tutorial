import csv
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import gtsam
except ImportError as error:
    raise unittest.SkipTest('GTSAM >= 4.3 is required for these tests.') from error
if tuple(int(part) for part in version('gtsam').split('.')[:2]) < (4, 3):
    raise unittest.SkipTest('GTSAM >= 4.3 is required for these tests.')

from test_utias_graph import builder as gbp_builder, example_data


ROOT = Path(__file__).resolve().parents[1]
GTSAM_DIR = ROOT / 'cooperative localization' / 'UTIAS dataset' / 'gtsam'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(GTSAM_DIR), str(GTSAM_DIR.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


builder = load_module('utias_gtsam_builder', GTSAM_DIR / 'build_gtsam_graph.py')


class UTIASGTSAMTests(unittest.TestCase):
    def setUp(self):
        self.data = example_data()
        self.covariances = {
            'prior_cov': np.diag([.2, .3, .1]),
            'odometry_cov': np.diag([.02, .03, .01]),
            'measurement_cov': np.array([[.1, .01], [.01, .02]]),
        }

    def test_same_initialization_factors_and_objective_as_gbp(self):
        for options, measurement_type in ((1, 'all'), (2, 'landmark'), (3, 'robot')):
            with self.subTest(options=options):
                graph, values, keys = builder.build_graph(
                    self.data, **self.covariances, options=options,
                )
                gbp, variables = gbp_builder.build_graph(
                    self.data, **self.covariances, measurement_type=measurement_type,
                )
                self.assertEqual(values.size(), len(gbp.var_nodes))
                self.assertEqual(graph.size(), len(gbp.factors) + len(self.data['robots']))
                for robot_id, nodes in variables['robots'].items():
                    for key, node in zip(keys[robot_id], nodes):
                        pose = values.atPose2(key)
                        np.testing.assert_allclose(pose.matrix(), node.mean, atol=1e-14)
                        # Test cost away from the initial estimate, including angle wrapping.
                        delta = np.array([.08 * (robot_id + 1), -.04, 2.9 * (-1)**robot_id])
                        pose = pose.compose(gtsam.Pose2.Expmap(delta))
                        values.update(key, pose)
                        node.mu = pose.matrix()
                self.assertAlmostEqual(graph.error(values), gbp.energy(), places=8)

    def test_factor_jacobians_in_gtsam_right_tangents(self):
        graph, values, robot_keys = builder.build_graph(self.data, **self.covariances)
        for robot_id, keys in robot_keys.items():
            for i, key in enumerate(keys):
                values.update(key, values.atPose2(key).retract(
                    np.array([.1 * (i + 1), -.15 * robot_id, .3 * (i + robot_id)]),
                ))
        for index in range(graph.size()):
            factor = graph.at(index)
            linear = factor.linearize(values)
            analytic, _ = linear.jacobian()
            columns = []
            for key in linear.keys():
                for delta in np.eye(3) * 1e-6:
                    plus, minus = gtsam.Values(values), gtsam.Values(values)
                    plus.update(key, values.atPose2(key).retract(delta))
                    minus.update(key, values.atPose2(key).retract(-delta))
                    columns.append(factor.noiseModel().whiten(
                        (factor.unwhitenedError(plus) - factor.unwhitenedError(minus)) / 2e-6,
                    ))
            np.testing.assert_allclose(analytic, np.column_stack(columns), atol=2e-7)

    def test_invalid_options_covariances_and_barcodes(self):
        with self.assertRaisesRegex(ValueError, 'options'):
            builder.build_graph(self.data, **self.covariances, options=4)
        for covariance in (np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                builder.build_graph(self.data, **dict(self.covariances, odometry_cov=covariance))
        self.data['robots'][1]['measurements'][0, 1] = 9999
        with self.assertRaisesRegex(ValueError, 'Unknown measurement barcode'):
            builder.build_graph(self.data, **self.covariances)

    def test_real_dataset_optimization_history_and_plot(self):
        main = load_module('utias_gtsam_main', GTSAM_DIR / 'UTIAS dataset.py')
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / 'history.csv'
            for options in (1, 2, 3):
                with self.subTest(options=options), patch.object(main, 'HISTORY_PATH', history_path):
                    graph, result, poses = main.main(
                        hz=1, duration=3, max_iters=30, start_offset=20,
                        prior_cov=main.prior_cov, odometry_cov=main.odometry_cov,
                        measurement_cov=main.measurement_cov, options=options, show_plot=False,
                    )
                    self.assertEqual(poses.shape, (5, 4, 3, 3))
                    self.assertTrue(np.isfinite(graph.error(result)))
                    with history_path.open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    final_iteration = int(rows[-1]['iteration'])
                    self.assertEqual(len(rows), (final_iteration + 1) * result.size())
                    initial = gtsam.Values()
                    keys = list(result.keys())
                    for key, row in zip(keys, rows[:result.size()]):
                        initial.insert(key, builder.as_pose2(np.array(json.loads(row['mean']))))
                    self.assertLessEqual(graph.error(result), graph.error(initial))
                    last_poses = np.array([
                        json.loads(row['mean']) for row in rows[-result.size():]
                    ]).reshape(poses.shape)
                    np.testing.assert_allclose(last_poses, poses)
                    raw = main.load_data(main.DATASET_DIR)
                    data = main.prepare_data(raw, 1, 3, start_offset=20)
                    fig, ax = main.plot_estimates(data, poses, options=options, solver_name='GTSAM LM')
                    self.assertIn('GTSAM LM', ax.get_title())
                    fig.savefig(Path(directory) / f'option_{options}.png')
                    import matplotlib.pyplot as plt
                    plt.close(fig)


if __name__ == '__main__':
    unittest.main()
