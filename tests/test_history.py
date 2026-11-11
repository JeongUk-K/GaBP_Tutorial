import unittest
from unittest.mock import patch

import numpy as np

from my_gbp import FactorGraph


class HistoryTests(unittest.TestCase):
    def setUp(self):
        output = patch('my_gbp.gbp.save_history')
        output.start()
        self.addCleanup(output.stop)

    def graph(self):
        graph = FactorGraph(nonlinear_factors=False)
        first = graph.add_variable(1, [0.], np.eye(1))
        second = graph.add_variable(1, [1.], np.eye(1))
        graph.add_factor(
            [first, second], lambda x: np.array([x[1] - x[0]]), [2.], np.eye(1),
            jacobian='analytic', jac_fn=lambda x: np.array([[-1., 1.]]),
        )
        return graph

    def test_recording_does_not_change_solution(self):
        recorded, unrecorded = self.graph(), self.graph()
        status_on, iterations_on, history = recorded.solve(max_iters=3, tol=0., history=True)
        status_off, iterations_off, empty = unrecorded.solve(max_iters=3, tol=0., history=False)
        self.assertEqual((status_on, iterations_on), (status_off, iterations_off))
        self.assertEqual(empty, [])
        self.assertEqual(unrecorded.history, [])
        self.assertIs(history, recorded.history)
        self.assertEqual([frame['iter'] for frame in history], [1, 2, 3])
        for first, second in zip(recorded.var_nodes, unrecorded.var_nodes):
            np.testing.assert_array_equal(first.mean, second.mean)
            np.testing.assert_array_equal(first.cov, second.cov)
        snapshot = history[-1]['means'][0].copy()
        recorded.var_nodes[0].mu[:] = 999.
        np.testing.assert_array_equal(history[-1]['means'][0], snapshot)
        self.assertFalse(np.shares_memory(history[-1]['covs'][0], recorded.var_nodes[0].cov))

    def test_default_disables_recording_and_clears_previous_history(self):
        graph = self.graph()
        _, _, saved = graph.solve(max_iters=2, tol=0., history=True)
        _, _, empty = graph.solve(max_iters=1)
        self.assertEqual(len(saved), 2)
        self.assertEqual(empty, [])
        self.assertEqual(graph.history, [])


if __name__ == '__main__':
    unittest.main()
