from pathlib import Path

from load_data import load_data, prepare_data
from build_graph import build_graph

import numpy as np


DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / 'dataset' / 'MRCLAM1' / 'MRCLAM_Dataset1'
)
HISTORY_PATH = Path(__file__).resolve().with_name('history.csv')

hz=2
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



def main(hz, duration, max_iters, *, prior_cov, odometry_cov, measurement_cov):
    raw_data = load_data(DATASET_DIR)
    data = prepare_data(raw_data, hz, duration)
    graph, variables = build_graph(
        data, prior_cov=prior_cov, odometry_cov=odometry_cov,
        measurement_cov=measurement_cov,
    )
    converged, n_iters, history = graph.solve(
        max_iters=max_iters, tol=1e-6, monitoring=True, history=True,
        history_path=HISTORY_PATH,
    )
    return graph, variables, converged, n_iters, history


if __name__ == '__main__':
    graph, variables, converged, n_iters, history = main(
        hz, duration, max_iters,
        prior_cov=prior_cov,
        odometry_cov=odometry_cov,
        measurement_cov=measurement_cov,
    )
    print('Converged:', converged, 'Iterations:', n_iters)

    print('History saved:', HISTORY_PATH)
