from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gabp
import gsolver

if __package__:
    from .multirobot_example import make_state, range_bearing, transform
else:
    from multirobot_example import make_state, range_bearing, transform


# Parameters
TOTAL_TIME = 20.0
TIME_STEP = 1.0
WINDOW_SIZE = 5  # Number of active pose timestamps per robot.
MESSAGE_PASSES_PER_WINDOW = 5

GIF_PATH = Path(__file__).with_name("multirobot_sliding_window.gif")
GIF_FPS = 10
ELLIPSE_CONFIDENCE = 0.95

ROBOT_START_POSES = np.array([
    (0.0, 0.0, 0.10),
    (0.9, 3.1, 0.05),
    (5.4, 1.5, 2.45),
])
FORWARD_SPEEDS = (0.23, 0.20, 0.14)
TURN_RATES = (0.015, -0.020, 0.027)
INITIAL_OFFSETS = np.array([
    (0.12, -0.08, 0.06),
    (-0.14, 0.16, -0.08),
    (0.10, 0.12, 0.05),
])
ODOMETRY_BIAS_PER_SECOND = np.array([0.018, -0.005, np.deg2rad(0.8)])

ROBOT_OBSERVATIONS = ((0, 1), (1, 2), (2, 0))
LANDMARK_POSES = ((-0.35, 4.4, 0.0), (6.0, 4.5, 0.0))
PRIOR_COVARIANCE = 0.01**2 * np.eye(6)
ODOMETRY_COVARIANCE = 0.05**2 * np.eye(6)
RANGE_BEARING_COVARIANCE = np.diag([0.1**2, np.deg2rad(3.0)**2])


# Graph
def reference_poses(timestamps):
    references = []
    for (x, y, yaw), speed, turn_rate in zip(ROBOT_START_POSES, FORWARD_SPEEDS, TURN_RATES):
        half_turn = 0.5 * turn_rate * timestamps
        distance = speed * timestamps * np.sinc(half_turn / np.pi)
        references.append([
            transform(px, py, heading)
            for px, py, heading in zip(
                x + distance * np.cos(yaw + half_turn),
                y + distance * np.sin(yaw + half_turn),
                yaw + 2.0 * half_turn,
            )
        ])
    return np.array(references)


def build_graph(end_time):
    landmark_graph = len(ROBOT_START_POSES)
    graph = gabp.MultiRobotGraph(landmark_graph + 1)
    landmarks = []

    # Give static landmarks an end-of-run timestamp so the time cutoff leaves them in the graph.
    for index, pose in enumerate(LANDMARK_POSES, start=1):
        mean = transform(*pose)
        landmark = make_state(f"landmark_{index}", end_time, mean)
        graph.add_pose(landmark_graph, end_time, landmark)
        graph.add_factor(
            landmark_graph,
            gsolver.SE3PoseVelPrior(f"landmark_prior_{index}", mean, PRIOR_COVARIANCE),
            landmark,
        )
        landmarks.append(landmark)

    return graph, landmarks


def add_observations(graph, step, states, references, landmarks):
    for observer, target in ROBOT_OBSERVATIONS:
        factor = gabp.RangeBearingFactor(
            f"r{observer}_sees_r{target}_{step}",
            range_bearing(references[observer], references[target]),
            RANGE_BEARING_COVARIANCE,
        )
        graph.add_factor(observer, factor, states[observer], states[target])

    for robot, state in enumerate(states):
        for index, (landmark, landmark_pose) in enumerate(zip(landmarks, LANDMARK_POSES)):
            factor = gabp.RangeBearingFactor(
                f"r{robot}_sees_landmark{index}_{step}",
                range_bearing(references[robot], transform(*landmark_pose)),
                RANGE_BEARING_COVARIANCE,
            )
            graph.add_factor(robot, factor, state, landmark)


# Solve
@dataclass(frozen=True)
class Frame:
    timestamps: tuple
    message_pass: int
    poses: np.ndarray
    xy_covariances: np.ndarray
    ground_truth: np.ndarray
    landmark_positions: np.ndarray


def capture_frame(window, message_pass, ground_truth, landmarks):
    poses = np.array([[state.mu.pose for state in states] for _, states in window])
    local_covariances = np.array([[state.sigma[:3, :3] for state in states] for _, states in window])
    rotation_xy = poses[..., :2, :3]
    xy_covariances = rotation_xy @ local_covariances @ rotation_xy.swapaxes(-1, -2)
    return Frame(
        tuple(timestamp for timestamp, _ in window),
        message_pass,
        poses,
        xy_covariances,
        ground_truth,
        np.array([landmark.mu.pose[:2, 3] for landmark in landmarks]),
    )


def simulate():
    timestamps = np.arange(0.0, np.nextafter(TOTAL_TIME, np.inf), TIME_STEP)
    references = reference_poses(timestamps)
    ground_truth = references[:, :, :2, 3].swapaxes(0, 1)
    graph, landmarks = build_graph(float(timestamps[-1]))
    window = deque()
    history = []

    for step, time in enumerate(timestamps):
        timestamp = float(time)
        motions = []
        if window:
            previous_states = window[-1][1]
            bias = transform(*(TIME_STEP * ODOMETRY_BIAS_PER_SECOND))
            motions = np.linalg.inv(references[:, step - 1]) @ references[:, step] @ bias
            initial_poses = [state.mu.pose @ motion for state, motion in zip(previous_states, motions)]
        else:
            initial_poses = [
                transform(*(pose + offset))
                for pose, offset in zip(ROBOT_START_POSES, INITIAL_OFFSETS)
            ]

        if len(window) == WINDOW_SIZE:
            window.popleft()
            cutoff = window[0][0] if window else timestamp
            graph.prune_before(cutoff)

        states = []
        for robot, initial_pose in enumerate(initial_poses):
            node = make_state(f"r{robot}_{step}", timestamp, initial_pose)
            graph.add_pose(robot, timestamp, node)
            states.append(node)
            if window:
                odometry = gsolver.SE3PoseVelPoseVel(
                    f"odom_r{robot}_{step}", motions[robot], ODOMETRY_COVARIANCE
                )
                graph.add_factor(robot, odometry, window[-1][1][robot], node)

        if step == 0:
            prior = gsolver.SE3PoseVelPrior("initial_pose_prior", references[0, 0], PRIOR_COVARIANCE)
            graph.add_factor(0, prior, states[0])

        add_observations(graph, step, states, references[:, step], landmarks)
        window.append((timestamp, states))
        history.append(capture_frame(window, 0, ground_truth, landmarks))
        for message_pass in range(1, MESSAGE_PASSES_PER_WINDOW + 1):
            graph.iterate()
            history.append(capture_frame(window, message_pass, ground_truth, landmarks))

    return history


# Plot
def save_iteration_gif(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    chi_squared = -2.0 * np.log1p(-ELLIPSE_CONFIDENCE)
    ground_truth = history[0].ground_truth
    positions = np.concatenate([frame.poses[..., :2, 3].reshape(-1, 2) for frame in history])
    variances = np.concatenate([
        np.diagonal(frame.xy_covariances, axis1=-2, axis2=-1).reshape(-1, 2) for frame in history
    ])
    radii = np.sqrt(chi_squared * np.maximum(variances, 0.0))
    reference_points = np.concatenate(
        [ground_truth.reshape(-1, 2)]
        + [frame.landmark_positions for frame in history]
    )
    lower = np.minimum((positions - radii).min(axis=0), reference_points.min(axis=0)) - 0.3
    upper = np.maximum((positions + radii).max(axis=0), reference_points.max(axis=0)) + 0.3

    colors = ("#0072B2", "#D55E00", "#009E73")
    robot_edge_color = "#72529B"
    landmark_edge_color = "#566676"
    anchor_color = "#293644"
    legend = [
        Line2D([], [], color=color, marker="o", label=f"Robot {robot + 1}")
        for robot, color in enumerate(colors)
    ]
    legend.extend([
        Line2D([], [], color=anchor_color, marker="*", linestyle="none", label="Landmark prior"),
        Line2D([], [], color=anchor_color, label="Odometry"),
        Line2D([], [], color=robot_edge_color, linestyle="--", label="Robot observation"),
        Line2D([], [], color=landmark_edge_color, linestyle=":", label="Landmark observation"),
        Line2D([], [], color=anchor_color, linestyle="--", marker="D", markerfacecolor="none",
               markersize=5, label="Full ground truth"),
    ])

    fig, ax = plt.subplots(figsize=(8.1, 6.5), constrained_layout=True)
    writer = PillowWriter(fps=GIF_FPS)
    with writer.saving(fig, str(GIF_PATH), dpi=100):
        for frame in history:
            points = frame.poses[..., :2, 3]
            symmetric_covariances = 0.5 * (
                frame.xy_covariances + frame.xy_covariances.swapaxes(-1, -2)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariances)
            diameters = 2.0 * np.sqrt(chi_squared * np.maximum(eigenvalues, 0.0))
            major_axes = eigenvectors[..., :, 1]
            angles = np.rad2deg(np.arctan2(major_axes[..., 1], major_axes[..., 0]))

            ax.clear()
            ax.set(
                xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal",
                xlabel="x [m]", ylabel="y [m]",
                title=f"Active window {frame.timestamps[0]:g} to {frame.timestamps[-1]:g} s"
                      f"  |  GBP pass {frame.message_pass}/{MESSAGE_PASSES_PER_WINDOW}\n"
                      f"{len(frame.timestamps)} pose times per robot"
                      f"  |  {ELLIPSE_CONFIDENCE:.0%} XY covariance ellipses",
            )
            ax.set_axisbelow(True)
            ax.grid(color="#DCE3EA", linewidth=0.7)
            ax.add_collection(LineCollection(
                [(stamp[observer], stamp[target]) for stamp in points for observer, target in ROBOT_OBSERVATIONS],
                colors=robot_edge_color, linestyles="--", linewidths=0.9, alpha=0.5, zorder=2,
            ))
            ax.add_collection(LineCollection(
                [(point, landmark) for point in points.reshape(-1, 2) for landmark in frame.landmark_positions],
                colors=landmark_edge_color, linestyles=":", linewidths=0.9, alpha=0.27, zorder=2,
            ))
            ax.scatter(*frame.landmark_positions.T, marker="*", s=122, color=anchor_color, zorder=6)
            if frame.timestamps[0] == 0.0:
                ax.scatter(*points[0, 0], marker="s", s=108, facecolors="none",
                           edgecolors=anchor_color, linewidths=1.1, zorder=5)

            for robot, color in enumerate(colors):
                track = points[:, robot]
                ax.plot(*track.T, "o-", color=color, linewidth=1.5, markersize=4, zorder=3)
                ax.scatter(*track[-1], s=46, color=color, edgecolors="white", linewidths=0.8, zorder=5)
                ax.plot(*ground_truth[:, robot].T, "D--", color=color, linewidth=1.7,
                        markersize=5.5, markerfacecolor="none", markeredgewidth=1.1, zorder=5.5)
                heading = frame.poses[-1, robot, :2, 0]
                ax.quiver(*track[-1], *heading, color=color, angles="xy", scale_units="xy",
                          scale=4.0, width=0.004, zorder=4)
                for index, point in enumerate(track):
                    height, width = diameters[index, robot]
                    ax.add_patch(Ellipse(
                        point, width, height, angle=angles[index, robot],
                        facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.0,
                    ))

            ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=4, frameon=False)
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    save_iteration_gif(simulate())
    print(f"Saved {GIF_PATH}")
