from pathlib import Path
import sys

# Allow direct execution by path without installing the local solver package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from load_data import load_data, prepare_data
from build_graph import build_graph
from build_graph_landmark import build_graph as build_landmark_graph
from build_graph_robot_meas import build_graph as build_robot_measurement_graph
from plottter import plot_estimates, print_rmse

import numpy as np


DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / 'dataset' / 'MRCLAM1' / 'MRCLAM_Dataset1'
)
HISTORY_PATH = Path(__file__).resolve().with_name('history.csv')

hz=10
start_offset=10.0  # Seconds after the common start of all robots' state data.
duration=100
max_iters=9000

# covariance matrices
prior_cov = 1e-6 * np.eye(3)

dt=1 / hz
odom_v_std = 0.1
odom_w_std = 5 * np.pi / 180
odom_y_cov = 0.00001

odometry_cov = np.diag([(odom_v_std**2)*(dt**2), odom_y_cov, (odom_w_std**2)*(dt**2)])  # 0.1 m and 5 deg

measurement_cov = np.diag([0.5**2, (3*np.pi/180)**2])  # 0.5 m and 3 deg

# 1 = full measurement model
# 2 = landmark-only
# 3 = robot-only
options = 2


def main(hz, duration, max_iters, *, prior_cov, odometry_cov, measurement_cov,
         start_offset=0.0, options=1, show_plot=True):
    builders = {
        1: build_graph,
        2: build_landmark_graph,
        3: build_robot_measurement_graph,
    }
    if options not in builders:
        raise ValueError('options must be 1 (full), 2 (landmark-only), or 3 (robot-only).')
    raw_data = load_data(DATASET_DIR)
    data = prepare_data(raw_data, hz, duration, start_offset=start_offset)
    graph, variables = builders[options](
        data, prior_cov=prior_cov, odometry_cov=odometry_cov,
        measurement_cov=measurement_cov,
    )
    converged, n_iters, history = graph.solve(
        max_iters=max_iters, tol=1, monitoring=True, history=True,
        history_path=HISTORY_PATH,
    )
    poses = np.asarray([
        [node.mean.copy() for node in nodes]
        for nodes in variables['robots'].values()
    ])
    print('Converged:', converged, 'Iterations:', n_iters)
    print('History saved:', HISTORY_PATH)
    print_rmse(data, poses)
    if show_plot:
        plot_estimates(data, poses, options=options, iteration=n_iters,
                       raw_data=raw_data, show=True)
    return graph, variables, converged, n_iters, history


if __name__ == '__main__':
    graph, variables, converged, n_iters, history = main(
        hz, duration, max_iters,
        start_offset=start_offset,
        options=options,
        prior_cov=prior_cov,
        odometry_cov=odometry_cov,
        measurement_cov=measurement_cov,
    )
