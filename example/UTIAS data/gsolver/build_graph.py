from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gabp
import gsolver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utias_data import (
    ROBOT_IDS, Observation, RobotData, checked_covariance, initial_pose,
    integrate_odometry, load_dataset, transform,
)


@dataclass
class GraphResult:
    graph: gabp.MultiRobotGraph
    timestamps: np.ndarray
    robot_states: dict[int, list[gsolver.SE3PoseVel]]
    landmarks: dict[int, gsolver.SE3PoseVel]
    observation_counts: dict[int, tuple[int, int]]


@dataclass
class GraphBuilder:
    result: GraphResult
    robots: dict[int, RobotData]
    observations: list[list[Observation]]
    state_covariance: np.ndarray
    robot_prior_covariance: np.ndarray
    odometry_covariance: np.ndarray

    def add_step(self, step, *, connect_odometry=True):
        result = self.result
        for robot_id, data in self.robots.items():
            add_robot_pose(
                result.graph, data, result.timestamps, step, result.robot_states[robot_id],
                robot_id=robot_id, state_covariance=self.state_covariance,
                prior_covariance=self.robot_prior_covariance, odometry_covariance=self.odometry_covariance,
                connect_odometry=connect_odometry,
            )

        for observation in self.observations[step]:
            robot_id, subject = observation.robot_id, observation.subject
            observer = result.robot_states[robot_id][step]
            if subject in result.robot_states:
                target = result.robot_states[subject][step]
            else:
                target = result.landmarks[subject]
            factor = gabp.RangeBearingFactor(
                f"rb_r{robot_id}_{observation.row}", observation.measurement, observation.covariance,
            )
            result.graph.add_factor(ROBOT_IDS.index(robot_id), factor, observer, target)


def make_state(node_id, timestamp, pose, velocity, covariance):
    state = gsolver.SE3PoseVelValue()
    state.pose = pose
    state.vel = velocity
    return gsolver.SE3PoseVel(node_id, state, covariance, timestamp)


def add_robot_pose(graph, data, timestamps, step, states, *, robot_id, state_covariance, prior_covariance,
                   odometry_covariance, connect_odometry):
    graph_index = ROBOT_IDS.index(robot_id)
    timestamp = timestamps[step]
    if step:
        motion = integrate_odometry(data.odometry, timestamps[step - 1], timestamp)
        pose = states[-1].mu.pose @ motion
    else:
        pose = initial_pose(data, timestamp)

    sample = np.searchsorted(data.odometry[:, 0], timestamp, side="right") - 1
    speed, turn_rate = data.odometry[sample, 1:]
    velocity = np.array([speed, 0.0, 0.0, 0.0, 0.0, turn_rate])
    node = make_state(f"r{robot_id}_{step}", timestamp, pose, velocity, state_covariance)
    graph.add_pose(graph_index, timestamp, node)
    if step == 0:
        factor = gsolver.SE3PoseVelPrior(f"prior_r{robot_id}", pose, prior_covariance)
        graph.add_factor(graph_index, factor, node)
    elif connect_odometry:
        factor = gsolver.SE3PoseVelPoseVel(f"odom_r{robot_id}_{step}", motion, odometry_covariance)
        graph.add_factor(graph_index, factor, states[-1], node)
    states.append(node)


def add_landmark_graph(graph, rows, timestamp, state_covariance, prior_covariance):
    map_graph = len(ROBOT_IDS)
    landmarks = {}
    for subject, x, y, _, _ in rows:
        landmark_id = int(subject)
        pose = transform(x, y, 0.0)
        node = make_state(f"landmark_{landmark_id}", timestamp, pose, np.zeros(6), state_covariance)
        graph.add_pose(map_graph, timestamp, node)
        factor = gsolver.SE3PoseVelPrior(f"prior_landmark_{landmark_id}", pose, prior_covariance)
        graph.add_factor(map_graph, factor, node)
        landmarks[landmark_id] = node
    return landmarks


def prepare_graph(dataset_dir, *, start_time, duration, odom_dt, landmark_dt,
                  state_covariance, robot_prior_covariance, landmark_prior_covariance, odometry_covariance,
                  robot_measurement_covariance, landmark_measurement_covariance):
    """Load MRCLAM data and prepare observations for incremental graph construction."""
    state_covariance = checked_covariance(state_covariance, 12, "state_covariance")
    robot_prior_covariance = checked_covariance(robot_prior_covariance, 6, "robot_prior_covariance")
    landmark_prior_covariance = checked_covariance(landmark_prior_covariance, 6, "landmark_prior_covariance")
    odometry_covariance = checked_covariance(odometry_covariance, 6, "odometry_covariance")

    data = load_dataset(
        dataset_dir, start_time=start_time, duration=duration, odom_dt=odom_dt, landmark_dt=landmark_dt,
        robot_measurement_covariance=robot_measurement_covariance,
        landmark_measurement_covariance=landmark_measurement_covariance,
    )
    graph = gabp.MultiRobotGraph(len(ROBOT_IDS) + 1)
    # Keep static landmark nodes inside every window by assigning the last timestamp.
    landmarks = add_landmark_graph(
        graph, data.landmark_rows, data.timestamps[-1], state_covariance, landmark_prior_covariance,
    )
    result = GraphResult(
        graph, data.timestamps, {robot_id: [] for robot_id in data.robots}, landmarks, data.observation_counts,
    )
    return GraphBuilder(
        result, data.robots, data.observations, state_covariance, robot_prior_covariance, odometry_covariance,
    )


def build_graph(dataset_dir, **parameters):
    """Build all robot poses and factors for a batch run."""
    builder = prepare_graph(dataset_dir, **parameters)
    for step in range(len(builder.result.timestamps)):
        builder.add_step(step)
    return builder.result
