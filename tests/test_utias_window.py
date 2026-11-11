import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = (Path(__file__).resolve().parents[1]
        / 'cooperative localization' / 'UTIAS dataset' / 'load_data.py')
SPEC = importlib.util.spec_from_file_location('utias_window_loader', PATH)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


def raw_fixture():
    robots = {}
    for robot_id, gt_times, odom_times in (
        (1, np.arange(7.), np.arange(8.) - .5),
        (2, np.arange(7.) + 1., np.arange(6.) + 1.25),
    ):
        robots[robot_id] = {
            'groundtruth': np.column_stack([gt_times, gt_times, gt_times * 0, gt_times * 0]),
            'odometry': np.column_stack([odom_times, odom_times * 0 + 1, odom_times * 0]),
            'measurements': np.array([
                [3.24, 72., 1., 0.], [3.25, 72., 2., 0.],
                [3.75, 72., 3., 0.], [4.25, 72., 4., 0.], [4.26, 72., 5., 0.],
            ]),
        }
    return {'robots': robots, 'landmarks': np.array([[6., 1., 2., .1, .1]]),
            'barcodes': np.array([[6, 72]])}


class UTIASWindowTests(unittest.TestCase):
    def test_default_keeps_original_common_start(self):
        data = loader.prepare_data(raw_fixture(), hz=2, duration=1.)
        self.assertEqual(data['start_time'], 1.25)
        np.testing.assert_array_equal(data['node_times'], [1.25, 1.75, 2.25])

    def test_offset_shifts_window_and_preserves_boundary_support(self):
        raw = raw_fixture()
        data = loader.prepare_data(raw, hz=2, duration=1., start_offset=2.)
        np.testing.assert_array_equal(data['node_times'], [3.25, 3.75, 4.25])
        self.assertEqual(data['end_time'], 4.25)
        for robot in data['robots'].values():
            np.testing.assert_array_equal(robot['measurements'][:, 0], [3.25, 3.75, 4.25])
            for name in ('groundtruth', 'odometry'):
                self.assertLessEqual(robot[name][0, 0], data['start_time'])
                self.assertGreaterEqual(robot[name][-1, 0], data['end_time'])
        np.testing.assert_array_equal(raw['robots'][1]['groundtruth'][:, 0], np.arange(7.))
        data['robots'][1]['measurements'][0, 2] = 100.
        self.assertEqual(raw['robots'][1]['measurements'][1, 2], 2.)

    def test_window_may_end_at_last_common_sample(self):
        data = loader.prepare_data(raw_fixture(), hz=4, duration=.75, start_offset=4.)
        self.assertEqual(data['end_time'], 6.)
        np.testing.assert_array_equal(data['node_times'], [5.25, 5.5, 5.75, 6.])

    def test_invalid_offsets_and_out_of_range_windows(self):
        for offset in (-1., np.nan, np.inf, -np.inf):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'start_offset'):
                loader.prepare_data(raw_fixture(), hz=2, duration=1., start_offset=offset)
        for offset in (4., 5.):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'shared interval'):
                loader.prepare_data(raw_fixture(), hz=2, duration=1., start_offset=offset)


if __name__ == '__main__':
    unittest.main()
