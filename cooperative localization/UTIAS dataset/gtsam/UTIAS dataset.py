"""Solve the same UTIAS localization problem using GTSAM Levenberg-Marquardt."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parents[2]))

import gtsam
import numpy as np

from build_gtsam_graph import build_graph
from load_data import load_data, prepare_data
from plottter import plot_estimates, print_rmse


DATASET_DIR = ROOT.parents[2] / 'dataset' / 'MRCLAM1' / 'MRCLAM_Dataset1'
HISTORY_PATH = ROOT / 'history.csv'

# Match the settings in ../UTIAS dataset.py when comparing the solvers.
hz = 10
start_offset = 10.0
duration = 100
max_iters = 9000
options = 1  # 1 = full, 2 = landmark-only, 3 = robot-only

prior_cov = 1e-6 * np.eye(3)
dt = 1 / hz
odom_v_std = 0.1
odom_w_std = 5 * np.pi / 180
odom_y_cov = 0.00001
odometry_cov = np.diag([
    odom_v_std**2 * dt**2, odom_y_cov, odom_w_std**2 * dt**2,
])
measurement_cov = np.diag([0.5**2, (3 * np.pi / 180)**2])


def write_history(writer, iteration, values, robot_keys):
    # Keep the parent plotter's iteration/variable_id/mean format (poses only).
    variable_id = 0
    for keys in robot_keys.values():
        for key in keys:
            writer.writerow([
                iteration, variable_id,
                json.dumps(values.atPose2(key).matrix().tolist(), separators=(',', ':')),
            ])
            variable_id += 1


def main(hz, duration, max_iters, *, prior_cov, odometry_cov, measurement_cov,
         start_offset=0.0, options=1, show_plot=True):
    raw_data = load_data(DATASET_DIR)
    data = prepare_data(raw_data, hz, duration, start_offset=start_offset)
    graph, initial, robot_keys = build_graph(
        data, prior_cov=prior_cov, odometry_cov=odometry_cov,
        measurement_cov=measurement_cov, options=options,
    )
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iters)
    params.setRelativeErrorTol(1e-6)
    params.setAbsoluteErrorTol(1e-6)

    print(f'GTSAM LM: {initial.size()} poses, {graph.size()} factors', flush=True)
    print(f'Initial error: {graph.error(initial):.6f}', flush=True)
    with HISTORY_PATH.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['iteration', 'variable_id', 'mean'])
        write_history(writer, 0, initial, robot_keys)

        def record_iteration(iteration, error_before, error_after):
            write_history(writer, iteration, optimizer.values(), robot_keys)
            stream.flush()
            print(f'Iteration {iteration}: error {error_after:.6f}', flush=True)

        params.iterationHook = record_iteration
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

    poses = np.asarray([
        [result.atPose2(key).matrix() for key in keys]
        for keys in robot_keys.values()
    ])
    print('Iterations:', optimizer.iterations())
    print(f'Final error: {graph.error(result):.6f}')
    print('History saved:', HISTORY_PATH)
    print_rmse(data, poses)
    if show_plot:
        plot_estimates(
            data, poses, options=options, iteration=optimizer.iterations(),
            raw_data=raw_data, show=True, solver_name='GTSAM LM',
        )
    return graph, result, poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plot', action='store_true', help='Run without opening a plot window.')
    args = parser.parse_args()
    graph, result, poses = main(
        hz, duration, max_iters,
        start_offset=start_offset, options=options,
        prior_cov=prior_cov, odometry_cov=odometry_cov,
        measurement_cov=measurement_cov, show_plot=not args.no_plot,
    )
