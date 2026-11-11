"""The UTIAS pose graph, optimized with GTSAM 4.3 or newer."""

from functools import partial
from importlib.metadata import version

import gtsam
import numpy as np

if tuple(int(part) for part in version('gtsam').split('.')[:2]) < (4, 3):
    raise ImportError('This graph requires GTSAM >= 4.3. Run it with /usr/local/bin/python3.')

from build_graph import (
    initial_pose,
    integrate_odometry,
    nearest_time_index,
    range_bearing,
    range_bearing_jacobian,
)


def as_pose2(matrix):
    return gtsam.Pose2(
        matrix[0, 2], matrix[1, 2],
        np.arctan2(matrix[1, 0], matrix[0, 0]),
    )


def noise_model(covariance, dimension):
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.shape != (dimension, dimension)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T)):
        raise ValueError(f'Covariance must be a finite, symmetric {dimension}x{dimension} matrix.')
    np.linalg.cholesky(covariance)
    return gtsam.noiseModel.Gaussian.Covariance(covariance)


def build_graph(data, *, prior_cov, odometry_cov, measurement_cov, options=1):
    """Return (graph, initial_values, robot_keys), with fixed landmark positions.

    options: 1 = all observations, 2 = landmark-only, 3 = robot-only.
    Pose residuals use Log(Z^-1 X_i^-1 X_j), as in the GBP problem.
    Covariances are used directly; observation order remains [range, bearing].
    """
    if options not in (1, 2, 3):
        raise ValueError('options must be 1 (full), 2 (landmark-only), or 3 (robot-only).')
    prior_noise = noise_model(prior_cov, 3)
    odometry_noise = noise_model(odometry_cov, 3)
    measurement_noise = noise_model(measurement_cov, 2)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    robot_keys = {}
    times = data['node_times']

    # One prior per robot; all later poses start at integrated odometry estimates.
    for robot_id, robot in data['robots'].items():
        keys = [gtsam.symbol(chr(ord('a') + robot_id - 1), i) for i in range(len(times))]
        robot_keys[robot_id] = keys
        pose = initial_pose(robot['groundtruth'], times[0])
        initial.insert(keys[0], as_pose2(pose))
        graph.add(gtsam.PriorFactorPose2(keys[0], as_pose2(pose), prior_noise))
        for i, (start, end) in enumerate(zip(times[:-1], times[1:])):
            motion = integrate_odometry(robot['odometry'], start, end)
            pose = pose @ motion
            initial.insert(keys[i + 1], as_pose2(pose))
            graph.add(gtsam.BetweenFactorPose2(
                keys[i], keys[i + 1], as_pose2(motion), odometry_noise,
            ))

    barcodes = {int(code): int(subject) for subject, code in data['barcodes']}
    landmarks = {int(row[0]): row[1:3].copy() for row in data['landmarks']}
    for robot_id, robot in data['robots'].items():
        for time, barcode, distance, bearing in robot['measurements']:
            if int(barcode) not in barcodes:
                raise ValueError(f'Unknown measurement barcode {int(barcode)} at {time}.')
            subject = barcodes[int(barcode)]
            is_robot = subject in robot_keys
            if not is_robot and subject not in landmarks:
                raise ValueError(f'No robot or landmark for barcode {int(barcode)}.')
            if (options == 2 and is_robot) or (options == 3 and not is_robot):
                continue
            index = nearest_time_index(times, time)
            keys = [robot_keys[robot_id][index]]
            landmark = None
            if is_robot:
                keys.append(robot_keys[subject][index])
            else:
                landmark = landmarks[subject]
            graph.add(gtsam.CustomFactor(
                measurement_noise, keys,
                partial(measurement_error, np.array([distance, bearing]), landmark),
            ))
    return graph, initial, robot_keys


def measurement_error(measurement, landmark, factor, values, jacobians):
    """Reuse GBP's wrapped bearing residual and right-tangent Jacobians."""
    states = [values.atPose2(key).matrix() for key in factor.keys()]
    if jacobians is not None:
        jacobian = range_bearing_jacobian(states, measurement, landmark)
        for i in range(len(states)):
            jacobians[i] = np.asfortranarray(jacobian[:, 3 * i:3 * (i + 1)])
    return range_bearing(states, measurement, landmark) - measurement
