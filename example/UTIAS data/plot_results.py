import csv
from dataclasses import dataclass

import numpy as np

from utias_data import WindowCost, wrap


@dataclass
class Frame:
    first: int
    last: int
    iteration: int
    poses: np.ndarray
    landmarks: np.ndarray
    cost: float
    factor_count: int


class RunHistory:
    def __init__(self, timestamps, robots, landmark_priors, observations, **covariances):
        self.timestamps = timestamps
        self.elapsed = timestamps - timestamps[0]
        self.robot_ids = tuple(robots)
        self.landmark_ids = tuple(landmark_priors)
        self.landmark_priors = np.array(list(landmark_priors.values()))
        ground_truth = []
        for robot in robots.values():
            time, x, y, yaw = robot.ground_truth.T
            ground_truth.append(np.column_stack((
                np.interp(timestamps, time, x),
                np.interp(timestamps, time, y),
                wrap(np.interp(timestamps, time, np.unwrap(yaw))),
            )))
        self.ground_truth = np.stack(ground_truth, axis=1)
        self.final_poses = np.full_like(self.ground_truth, np.nan)
        self.objective = WindowCost(timestamps, robots, landmark_priors, observations, **covariances)
        self.frames = []

    def record(self, first, last, iteration, read_pose, read_landmark):
        poses = np.array([
            [read_pose(robot_id, step) for robot_id in self.robot_ids]
            for step in range(first, last + 1)
        ])
        poses[..., 2] = wrap(poses[..., 2])
        landmarks = np.array([read_landmark(landmark_id) for landmark_id in self.landmark_ids])
        landmarks[:, 2] = wrap(landmarks[:, 2])
        cost, factor_count = self.objective.evaluate(first, last, poses, landmarks)
        self.frames.append(Frame(first, last, iteration, poses, landmarks, cost, factor_count))
        self.final_poses[first:last + 1] = poses

    def save(self, output_dir, *, solver, solve_iter, fps):
        if not self.frames:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_csv(output_dir)
        plot_iterations(self, output_dir / "iterations.gif", solver=solver, solve_iter=solve_iter, fps=fps)

    def save_csv(self, output_dir):
        final = self.frames[-1]
        with (output_dir / "states.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow((
                "kind", "id", "pose_step", "timestamp_s", "elapsed_s",
                "x_m", "y_m", "yaw_rad", "gt_x_m", "gt_y_m", "gt_yaw_rad",
            ))
            for step in range(final.last + 1):
                for index, robot_id in enumerate(self.robot_ids):
                    writer.writerow((
                        "robot", robot_id, step, self.timestamps[step], self.elapsed[step],
                        *self.final_poses[step, index], *self.ground_truth[step, index],
                    ))
            for landmark_id, estimate, prior in zip(self.landmark_ids, final.landmarks, self.landmark_priors):
                writer.writerow(("landmark", landmark_id, "", "", "", *estimate, *prior))

        with (output_dir / "cost.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow((
                "window_step", "start_step", "window_start_s", "window_end_s", "iteration", "factor_count", "cost",
            ))
            for frame in self.frames:
                writer.writerow((
                    frame.last, frame.first, self.elapsed[frame.first], self.elapsed[frame.last],
                    frame.iteration, frame.factor_count, frame.cost,
                ))


def plot_iterations(history, path, *, solver, solve_iter, fps):
    from matplotlib.animation import PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(8.4, 4.9), dpi=84, facecolor="white")
    FigureCanvasAgg(figure)
    map_ax, cost_ax = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
    figure.subplots_adjust(left=0.09, bottom=0.19, right=0.97, top=0.83, wspace=0.38)
    title = figure.suptitle("", fontsize=12, y=0.96, fontweight="bold")
    colors = ("#0072B2", "#D55E00", "#009E73", "#A35284", "#9D7600")
    map_ax.grid(color="#E5E7EB", linewidth=0.6)
    map_ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    map_ax.set_axisbelow(True)

    reference = np.concatenate((history.ground_truth[..., :2].reshape(-1, 2), history.landmark_priors[:, :2]))
    margin = max(0.12 * np.ptp(reference, axis=0).max(), 0.2)
    lower, upper = reference.min(axis=0) - margin, reference.max(axis=0) + margin
    map_ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    map_ax.scatter(*history.landmark_priors[:, :2].T, marker="x", s=19, color="#6B7280", zorder=2)
    landmarks = map_ax.scatter([], [], marker="*", s=35, color="#374151", zorder=4)
    off_map = map_ax.text(0.02, 0.97, "", transform=map_ax.transAxes, va="top", color="#9A3412", fontsize=8)

    active, archived = [], []
    for index, (robot_id, color) in enumerate(zip(history.robot_ids, colors)):
        map_ax.plot(*history.ground_truth[:, index, :2].T, "--", lw=1.0, color=color, alpha=0.55)
        archived.append(map_ax.plot([], [], color=color, lw=1.0, alpha=0.28)[0])
        active.append(map_ax.plot([], [], ".-", lw=1.6, ms=4, color=color, label=f"R{robot_id}")[0])
    figure.legend(
        handles=active + [
            Line2D([], [], color="#6B7280", ls="--", label="GT"),
            Line2D([], [], color="#9CA3AF", label="Archived"),
            Line2D([], [], color="#374151", marker="*", ls="none", label="Landmark"),
        ],
        loc="lower center", ncol=8, frameon=False, fontsize=8, columnspacing=1.2,
    )

    cost_ax.set(xlabel="Iteration", ylabel="Window cost", xlim=(-0.2, solve_iter + 0.2))
    cost_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    cost_ax.grid(color="#E5E7EB", linewidth=0.6)
    cost_line, = cost_ax.plot([], [], "o-", color="#2563EB", linewidth=1.4, markersize=3)
    cost_label = cost_ax.set_title("", fontsize=10)
    window_maxima = {}
    for frame in history.frames:
        if np.isfinite(frame.cost):
            window_maxima[frame.last] = max(window_maxima.get(frame.last, 0.0), frame.cost)

    writer = PillowWriter(fps=fps)
    last_step = None
    with writer.saving(figure, str(path), dpi=figure.dpi):
        for frame in history.frames:
            if last_step != frame.last:
                last_step = frame.last
                iterations, costs = [], []
                maximum = window_maxima.get(frame.last, 0.0)
                if maximum >= 100.0:
                    cost_ax.set_yscale("symlog", linthresh=1.0)
                else:
                    cost_ax.set_yscale("linear")
                    cost_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                cost_ax.set_ylim(0.0, max(maximum * 1.1, 1e-6))
            iterations.append(frame.iteration)
            costs.append(frame.cost if np.isfinite(frame.cost) else np.nan)

            for index, (active_line, archived_line) in enumerate(zip(active, archived)):
                active_line.set_data(*frame.poses[:, index, :2].T)
                archived_line.set_data(*history.final_poses[:frame.first, index, :2].T)
            landmarks.set_offsets(frame.landmarks[:, :2])
            xy = frame.poses[..., :2]
            outside = np.count_nonzero(((xy < lower) | (xy > upper)).any(axis=-1))
            off_map.set_text(f"{outside} active poses outside map" if outside else "")
            cost_line.set_data(iterations, costs)
            cost = f"{frame.cost:.4g}" if np.isfinite(frame.cost) else "non-finite"
            cost_label.set_text(f"Cost {cost}")
            title.set_text(
                f"{solver}   |   Window {frame.last + 1}/{len(history.timestamps)}"
                f"   |   Iteration {frame.iteration}/{solve_iter}\n"
                f"Active pose times {history.elapsed[frame.first]:g}–{history.elapsed[frame.last]:g} s"
            )
            writer.grab_frame()
