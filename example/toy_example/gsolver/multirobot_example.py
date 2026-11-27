from pathlib import Path

import numpy as np

import gabp
import gsolver


# Parameters
TIMESTAMPS = (0.0, 1.0, 2.0)
ITERATIONS = 30
GIF_PATH = Path(__file__).with_name("multirobot_iterations.gif")
GIF_FPS = 4
ELLIPSE_CONFIDENCE = 0.95

# Each pose is (x, y, yaw), with yaw in radians.
ROBOT_1_POSES = (
    (0.0, 0.0, 0.00),
    (1.0, 0.1, 0.15),
    (2.0, 0.2, 0.20),
)
ROBOT_2_POSES = (
    (0.0, 3.0, 0.10),
    (1.0, 3.1, 0.12),
    (2.0, 3.4, 0.22),
)
ROBOT_3_POSES = (
    (3.0, 1.5, 2.20),
    (2.5, 2.0, 2.20),
    (2.0, 2.6, 2.20),
)
ROBOT_POSES = np.array([ROBOT_1_POSES, ROBOT_2_POSES, ROBOT_3_POSES])

INITIAL_OFFSETS = np.array([
    (0.15, -0.08, 0.06),
    (-0.14, 0.16, -0.08),
    (0.10, 0.12, 0.05),
])
ROBOT_OBSERVATIONS = ((0, 1), (1, 2), (2, 0))
LANDMARK_POSE = (3.5, 4.5, 0.0)
ANCHOR_ROBOT = 0

STATE_COVARIANCE = 0.1 * np.eye(12)
PRIOR_COVARIANCE = 0.01**2 * np.eye(6)
ODOMETRY_COVARIANCE = 0.05**2 * np.eye(6)
RANGE_BEARING_COVARIANCE = np.diag([0.1**2, np.deg2rad(3.0)**2])


# Graph
def transform(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0, x],
        [s, c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def make_state(node_id, timestamp, pose):
    state = gsolver.SE3PoseVelValue()
    state.pose = pose
    state.vel = np.zeros(6)
    return gsolver.SE3PoseVel(node_id, state, STATE_COVARIANCE, timestamp)


def range_bearing(observer, target):
    local = observer[:3, :3].T @ (target[:3, 3] - observer[:3, 3])
    return np.array([np.linalg.norm(local), np.arctan2(local[1], local[0])])


def build_graph():
    graph = gabp.MultiRobotGraph(len(ROBOT_POSES))
    references = [[transform(*pose) for pose in trajectory] for trajectory in ROBOT_POSES]
    states = [[] for _ in ROBOT_POSES]

    landmark_pose = transform(*LANDMARK_POSE)
    landmark = make_state("landmark", TIMESTAMPS[0], landmark_pose)
    graph.add_pose(ANCHOR_ROBOT, TIMESTAMPS[0], landmark)
    graph.add_factor(
        ANCHOR_ROBOT,
        gsolver.SE3PoseVelPrior("landmark_prior", landmark_pose, PRIOR_COVARIANCE),
        landmark,
    )

    for robot, trajectory in enumerate(ROBOT_POSES):
        for step, timestamp in enumerate(TIMESTAMPS):
            initial_pose = transform(*(trajectory[step] + INITIAL_OFFSETS[robot]))
            node = make_state(f"r{robot + 1}_t{step}", timestamp, initial_pose)
            graph.add_pose(robot, timestamp, node)
            states[robot].append(node)

    # The first robot pose fixes the heading of the map around the landmark.
    graph.add_factor(
        ANCHOR_ROBOT,
        gsolver.SE3PoseVelPrior("initial_pose_prior", references[ANCHOR_ROBOT][0], PRIOR_COVARIANCE),
        states[ANCHOR_ROBOT][0],
    )

    for robot, trajectory in enumerate(references):
        for step, (first, second) in enumerate(zip(trajectory, trajectory[1:])):
            motion = np.linalg.inv(first) @ second
            factor = gsolver.SE3PoseVelPoseVel(f"odom_r{robot + 1}_t{step}", motion, ODOMETRY_COVARIANCE)
            graph.add_factor(robot, factor, states[robot][step], states[robot][step + 1])

    for step in range(len(TIMESTAMPS)):
        for observer, target in ROBOT_OBSERVATIONS:
            measurement = range_bearing(references[observer][step], references[target][step])
            factor = gabp.RangeBearingFactor(
                f"r{observer + 1}_sees_r{target + 1}_t{step}", measurement, RANGE_BEARING_COVARIANCE
            )
            graph.add_factor(observer, factor, states[observer][step], states[target][step])

        for robot, trajectory in enumerate(references):
            measurement = range_bearing(trajectory[step], landmark_pose)
            factor = gabp.RangeBearingFactor(
                f"r{robot + 1}_sees_landmark_t{step}", measurement, RANGE_BEARING_COVARIANCE
            )
            graph.add_factor(robot, factor, states[robot][step], landmark)

    return graph, states, landmark


# Solve
def capture_states(robot_states, landmark):
    poses = np.array([[state.mu.pose for state in states] for states in robot_states])
    local_covariances = np.array([[state.sigma[:3, :3] for state in states] for states in robot_states])

    # SE3PoseVel stores translation uncertainty in the local tangent frame.
    rotation_xy = poses[..., :2, :3]
    xy_covariances = rotation_xy @ local_covariances @ rotation_xy.swapaxes(-1, -2)
    return poses, xy_covariances, landmark.mu.pose[:2, 3].copy()


def solve(graph, robot_states, landmark):
    history = [capture_states(robot_states, landmark)]
    for _ in range(ITERATIONS):
        graph.iterate()
        history.append(capture_states(robot_states, landmark))
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

    poses, covariances, landmark_positions = (np.stack(values) for values in zip(*history))
    positions = poses[..., :2, 3]
    headings = poses[..., :2, 0]

    symmetric_covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariances)
    chi_squared = -2.0 * np.log1p(-ELLIPSE_CONFIDENCE)
    diameters = 2.0 * np.sqrt(chi_squared * np.maximum(eigenvalues, 0.0))
    major_axes = eigenvectors[..., :, 1]
    angles = np.rad2deg(np.arctan2(major_axes[..., 1], major_axes[..., 0]))

    radius = 0.5 * diameters[..., 1, None]
    lower = np.minimum((positions - radius).min(axis=(0, 1, 2)), landmark_positions.min(axis=0)) - 0.3
    upper = np.maximum((positions + radius).max(axis=(0, 1, 2)), landmark_positions.max(axis=0)) + 0.3

    fig, ax = plt.subplots(figsize=(8.1, 7.0), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    writer = PillowWriter(fps=GIF_FPS)
    with writer.saving(fig, str(GIF_PATH), dpi=100):
        for iteration, robot_positions in enumerate(positions):
            ax.clear()
            ax.set(
                xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                xlabel="x [m]", ylabel="y [m]", aspect="equal",
                title=f"GBP iteration {iteration:02d} / {len(history) - 1:02d}"
                      f"  |  {ELLIPSE_CONFIDENCE:.0%} XY covariance ellipses",
            )
            ax.set_axisbelow(True)
            ax.grid(color="#DCE3EA", linewidth=0.7)

            landmark_xy = landmark_positions[iteration]
            robot_edges = LineCollection(
                [segment for observer, target in ROBOT_OBSERVATIONS
                 for segment in zip(robot_positions[observer], robot_positions[target])],
                colors="#72529B", linestyles="--", linewidths=1.1, alpha=0.7,
                label="Robot observation", zorder=2,
            )
            landmark_edges = LineCollection(
                [(point, landmark_xy) for point in robot_positions.reshape(-1, 2)],
                colors="#566676", linestyles=":", linewidths=1.2, alpha=0.55,
                label="Landmark observation", zorder=2,
            )
            ax.add_collection(robot_edges)
            ax.add_collection(landmark_edges)

            prior_x, prior_y = np.array([robot_positions[ANCHOR_ROBOT, 0], landmark_xy]).T
            prior = ax.scatter(prior_x, prior_y, marker="s", s=172, facecolors="none",
                               edgecolors="#293644", linewidths=1.25, label="Pose prior", zorder=5)
            landmark_marker = ax.scatter(*landmark_xy, marker="*", s=100, color="#293644",
                                         label="Landmark", zorder=6)

            robot_markers = []
            for robot, (points, color) in enumerate(zip(robot_positions, colors)):
                x, y = points.T
                trajectory, = ax.plot(x, y, "o-", color=color, linewidth=1.4, markersize=5,
                                      label=f"Robot {robot + 1}", zorder=3)
                robot_markers.append(trajectory)
                heading = headings[iteration, robot]
                ax.quiver(x, y, heading[:, 0], heading[:, 1], color=color,
                          angles="xy", scale_units="xy", scale=4.0, width=0.004, zorder=4)

                for step, (point, timestamp) in enumerate(zip(points, TIMESTAMPS)):
                    height, width = diameters[iteration, robot, step]
                    ax.add_patch(Ellipse(point, width, height, angle=angles[iteration, robot, step],
                                         facecolor=color, edgecolor=color, alpha=0.28, linewidth=1.2))
                    ax.annotate(f"{timestamp:g} s", point, xytext=(7, 8), textcoords="offset points",
                                color=color, fontsize=8, zorder=5)

            odometry = Line2D([], [], color="#293644", linewidth=1.4, label="Odometry")
            legend_columns = zip([*robot_markers, landmark_marker], [odometry, robot_edges, landmark_edges, prior])
            handles = [handle for column in legend_columns for handle in column]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    graph, robot_states, landmark = build_graph()
    history = solve(graph, robot_states, landmark)
    save_iteration_gif(history)

    print(f"{'Robot':>5} {'Time [s]':>9} {'x [m]':>9} {'y [m]':>9} {'yaw [deg]':>10}")
    for robot, states in enumerate(robot_states, start=1):
        for timestamp, state in zip(TIMESTAMPS, states):
            pose = state.mu.pose
            x, y = pose[:2, 3]
            yaw = np.rad2deg(np.arctan2(pose[1, 0], pose[0, 0]))
            print(f"{robot:>5} {timestamp:>9.1f} {x:>9.3f} {y:>9.3f} {yaw:>10.2f}")
