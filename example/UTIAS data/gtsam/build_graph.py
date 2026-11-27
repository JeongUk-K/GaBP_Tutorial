from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utias_data import (
    ROBOT_IDS, UTIASData, checked_covariance, initial_pose, integrate_odometry,
    load_dataset, planar_pose, wrap,
)


def pose_key(robot_id, step):
    return gtsam.symbol("x", step * len(ROBOT_IDS) + robot_id - 1)


def landmark_key(landmark_id):
    return gtsam.symbol("l", landmark_id)


def inverse_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def compose(pose, motion):
    xy = pose[:2] + inverse_rotation(pose[2]).T @ motion[:2]
    return np.array([*xy, wrap(pose[2] + motion[2])])


def prior_factor(key, measurement, noise):
    measured_rotation = inverse_rotation(measurement[2])
    jacobian = np.eye(3)
    jacobian[:2, :2] = measured_rotation

    def residual(_, values, jacobians):
        difference = values.atVector(key) - measurement
        if jacobians is not None:
            jacobians[0] = jacobian
        return np.array([*(measured_rotation @ difference[:2]), wrap(difference[2])])

    return gtsam.CustomFactor(noise, [key], residual)


def odometry_factor(first_key, second_key, measurement, noise):
    measured_rotation = inverse_rotation(measurement[2])

    def residual(_, values, jacobians):
        first, second = values.atVector(first_key), values.atVector(second_key)
        first_rotation = inverse_rotation(first[2])
        translation = first_rotation @ (second[:2] - first[:2])
        error_xy = measured_rotation @ (translation - measurement[:2])
        error_yaw = wrap(second[2] - first[2] - measurement[2])

        if jacobians is not None:
            first_jacobian, second_jacobian = np.eye(3), np.eye(3)
            second_jacobian[:2, :2] = measured_rotation @ first_rotation
            first_jacobian[:2, :2] = -second_jacobian[:2, :2]
            first_jacobian[:2, 2] = measured_rotation @ np.array([translation[1], -translation[0]])
            first_jacobian[2, 2] = -1.0
            jacobians[0], jacobians[1] = first_jacobian, second_jacobian
        return np.array([*error_xy, error_yaw])

    return gtsam.CustomFactor(noise, [first_key, second_key], residual)


def range_bearing_factor(observer_key, target_key, measurement, covariance):
    measured_range, measured_bearing = measurement
    noise = gtsam.noiseModel.Gaussian.Covariance(covariance)

    def residual(_, values, jacobians):
        observer, target = values.atVector(observer_key), values.atVector(target_key)
        dx, dy = target[:2] - observer[:2]
        distance = np.hypot(dx, dy)
        if distance < 1e-9:
            if jacobians is not None:
                jacobians[0], jacobians[1] = np.zeros((2, 3)), np.zeros((2, 3))
            return np.zeros(2)

        if jacobians is not None:
            distance_squared = distance**2
            target_jacobian = np.array([
                [dx / distance, dy / distance, 0.0],
                [-dy / distance_squared, dx / distance_squared, 0.0],
            ])
            observer_jacobian = -target_jacobian
            observer_jacobian[1, 2] = -1.0
            jacobians[0], jacobians[1] = observer_jacobian, target_jacobian
        bearing_error = wrap(np.arctan2(dy, dx) - observer[2] - measured_bearing)
        return np.array([distance - measured_range, bearing_error])

    return gtsam.CustomFactor(noise, [observer_key, target_key], residual)


@dataclass
class GraphResult:
    graph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    timestamps: np.ndarray
    robot_states: dict[int, list[np.ndarray]]
    landmarks: dict[int, np.ndarray]
    observation_counts: dict[int, tuple[int, int]]


@dataclass
class StepFactors:
    index: int
    measurements: gtsam.NonlinearFactorGraph
    odometry: gtsam.NonlinearFactorGraph


@dataclass
class GraphBuilder:
    result: GraphResult
    data: UTIASData
    landmark_priors: gtsam.NonlinearFactorGraph
    robot_prior_noise: gtsam.noiseModel.Gaussian
    odometry_noise: gtsam.noiseModel.Gaussian
    active_steps: deque[StepFactors] = field(default_factory=deque)

    def add_step(self, step, *, first_step=0):
        result = self.result
        while self.active_steps and self.active_steps[0].index < first_step:
            self.active_steps.popleft()

        measurements = gtsam.NonlinearFactorGraph()
        odometry = gtsam.NonlinearFactorGraph()
        timestamp = result.timestamps[step]
        for robot_id, robot in self.data.robots.items():
            key = pose_key(robot_id, step)
            states = result.robot_states[robot_id]
            if step == 0:
                pose = planar_pose(initial_pose(robot, timestamp))
                measurements.add(prior_factor(key, pose, self.robot_prior_noise))
            else:
                motion = planar_pose(integrate_odometry(robot.odometry, result.timestamps[step - 1], timestamp))
                pose = compose(states[-1], motion)
                if step > first_step:
                    odometry.add(odometry_factor(pose_key(robot_id, step - 1), key, motion, self.odometry_noise))
            states.append(pose)

        for observation in self.data.observations[step]:
            subject = observation.subject
            target = pose_key(subject, step) if subject in result.robot_states else landmark_key(subject)
            measurements.add(range_bearing_factor(
                pose_key(observation.robot_id, step), target, observation.measurement, observation.covariance,
            ))
        self.active_steps.append(StepFactors(step, measurements, odometry))

        graph = gtsam.NonlinearFactorGraph(self.landmark_priors)
        values = gtsam.Values()
        for landmark_id, pose in result.landmarks.items():
            values.insert(landmark_key(landmark_id), pose)
        for active in self.active_steps:
            graph.push_back(active.measurements)
            if active.index > first_step:
                graph.push_back(active.odometry)
            for robot_id, states in result.robot_states.items():
                values.insert(pose_key(robot_id, active.index), states[active.index])
        result.graph, result.values = graph, values

    def store_estimate(self, values):
        result = self.result
        result.values = gtsam.Values(values)
        for landmark_id in result.landmarks:
            result.landmarks[landmark_id] = values.atVector(landmark_key(landmark_id))
        for active in self.active_steps:
            for robot_id, states in result.robot_states.items():
                states[active.index] = values.atVector(pose_key(robot_id, active.index))


def prepare_graph(dataset_dir, *, start_time, duration, odom_dt, landmark_dt,
                  robot_prior_covariance, landmark_prior_covariance, odometry_covariance,
                  robot_measurement_covariance, landmark_measurement_covariance):
    """Prepare the same MRCLAM measurements as the GBP example."""
    robot_prior_noise = gtsam.noiseModel.Gaussian.Covariance(
        checked_covariance(robot_prior_covariance, 3, "robot_prior_covariance")
    )
    landmark_prior_noise = gtsam.noiseModel.Gaussian.Covariance(
        checked_covariance(landmark_prior_covariance, 3, "landmark_prior_covariance")
    )
    odometry_noise = gtsam.noiseModel.Gaussian.Covariance(
        checked_covariance(odometry_covariance, 3, "odometry_covariance")
    )
    data = load_dataset(
        dataset_dir, start_time=start_time, duration=duration, odom_dt=odom_dt, landmark_dt=landmark_dt,
        robot_measurement_covariance=robot_measurement_covariance,
        landmark_measurement_covariance=landmark_measurement_covariance,
    )
    landmarks = {int(subject): np.array([x, y, 0.0]) for subject, x, y, _, _ in data.landmark_rows}
    landmark_priors = gtsam.NonlinearFactorGraph()
    for landmark_id, pose in landmarks.items():
        landmark_priors.add(prior_factor(landmark_key(landmark_id), pose, landmark_prior_noise))
    result = GraphResult(
        gtsam.NonlinearFactorGraph(), gtsam.Values(), data.timestamps,
        {robot_id: [] for robot_id in data.robots}, landmarks, data.observation_counts,
    )
    return GraphBuilder(result, data, landmark_priors, robot_prior_noise, odometry_noise)
