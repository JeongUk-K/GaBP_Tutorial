import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from my_gbp import FactorGraph
from my_gbp.utils.save_history import save_history


class CSVHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / 'history.csv'

    def rows(self):
        with self.path.open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    def graph(self):
        graph = FactorGraph(nonlinear_factors=False)
        first = graph.add_variable(1, [0.], np.eye(1))
        second = graph.add_variable(1, [1.], np.eye(1))
        graph.add_factor(
            [first, second], lambda x: np.array([x[1]-x[0]]), [2.], np.eye(1),
            jacobian='analytic', jac_fn=lambda x: np.array([[-1., 1.]]),
        )
        return graph

    def test_array_shapes_and_all_iterations_round_trip(self):
        frame = {'iter': 1, 'means': [np.eye(3), np.array([[1.], [2.]])],
                 'covs': [np.diag([1., 2., 3.]), np.array([[2., .2], [.2, 3.]])]}
        save_history(None, [7, 3], self.path, append=False)
        save_history(frame, [7, 3], self.path)
        save_history({**frame, 'iter': 2}, [7, 3], self.path)
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows[0]), {'iteration', 'variable_id', 'mean', 'cov'})
        self.assertEqual([(r['iteration'], r['variable_id']) for r in rows],
                         [('1', '7'), ('1', '3'), ('2', '7'), ('2', '3')])
        for row, mean, cov in zip(rows[:2], frame['means'], frame['covs']):
            np.testing.assert_array_equal(json.loads(row['mean']), mean)
            np.testing.assert_array_equal(json.loads(row['cov']), cov)
        self.assertEqual(self.path.read_bytes().count(b'\xef\xbb\xbf'), 1)

    def test_completed_iteration_survives_solver_exception(self):
        graph = self.graph()
        original = graph.synchronous_iteration
        count = 0

        def fail_on_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise np.linalg.LinAlgError('Test failure')
            original(*args, **kwargs)

        with patch.object(graph, 'synchronous_iteration', side_effect=fail_on_second):
            with self.assertRaisesRegex(np.linalg.LinAlgError, 'Test failure'):
                graph.solve(max_iters=3, tol=0., history=True, history_path=self.path)
        self.assertEqual(len(graph.history), 1)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        for i, row in enumerate(rows):
            self.assertEqual(row['iteration'], '1')
            np.testing.assert_array_equal(json.loads(row['mean']), graph.history[0]['means'][i])

    def test_disabled_history_leaves_existing_file_untouched(self):
        self.path.write_text('keep this file', encoding='utf-8')
        graph = self.graph()
        _, _, history = graph.solve(max_iters=1, history=False, history_path=self.path)
        self.assertEqual(history, [])
        self.assertEqual(self.path.read_text(encoding='utf-8'), 'keep this file')

    def test_default_path_and_new_solve_reset_the_csv(self):
        graph = self.graph()
        with patch('my_gbp.utils.save_history.Path.cwd', return_value=self.directory):
            graph.solve(max_iters=2, tol=0., history=True)
        self.assertEqual(len(self.rows()), 4)
        graph.solve(max_iters=1, history=True, history_path=self.path)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['iteration'] == '1' for row in rows))


if __name__ == '__main__':
    unittest.main()
