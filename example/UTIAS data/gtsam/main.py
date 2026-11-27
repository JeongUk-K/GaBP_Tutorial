from pathlib import Path

import gtsam
import numpy as np

from build_graph import landmark_key, pose_key, prepare_graph
from plot_results import RunHistory


# Parameters
dataset_dir = Path(__file__).resolve().parents[3] / "dataset" / "MRCLAM1" / "MRCLAM_Dataset1"
start_time = 0.0  # Seconds from the start of the common odometry and ground-truth interval.
duration = 100.0  # sec
window = 30.0  # sec
solve_iter = 10  # LM iterations per window.
odom_dt = 0.5  # sec
landmark_dt = 1.0  # sec
output_dir = Path(__file__).resolve().parent / "results"
gif_fps = 15

# Planar pose covariance order is [tx, ty, yaw].
robot_prior_covariance = 0.01**2 * np.eye(3)
landmark_prior_covariance = 0.01**2 * np.eye(3)
odometry_covariance = np.diag(np.square([0.05, 0.01, np.deg2rad(2.5)]))

# Measurement order is [range (m), bearing (rad)].
robot_measurement_covariance = np.diag(np.square([0.5, np.deg2rad(3.0)]))
landmark_measurement_covariance = np.diag(np.square([0.5, np.deg2rad(3.0)]))


def main():
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    if not isinstance(solve_iter, (int, np.integer)) or solve_iter < 1:
        raise ValueError("solve_iter must be a positive integer")

    builder = prepare_graph(
        dataset_dir,
        start_time=start_time,
        duration=duration,
        odom_dt=odom_dt,
        landmark_dt=landmark_dt,
        robot_prior_covariance=robot_prior_covariance,
        landmark_prior_covariance=landmark_prior_covariance,
        odometry_covariance=odometry_covariance,
        robot_measurement_covariance=robot_measurement_covariance,
        landmark_measurement_covariance=landmark_measurement_covariance,
    )
    result = builder.result
    history = RunHistory(
        result.timestamps, builder.data.robots, result.landmarks, builder.data.observations,
        robot_prior_covariance=robot_prior_covariance,
        landmark_prior_covariance=landmark_prior_covariance,
        odometry_covariance=odometry_covariance,
    )

    def record(first_step, step, iteration, values):
        history.record(
            first_step, step, iteration,
            lambda robot_id, pose_step: values.atVector(pose_key(robot_id, pose_step)),
            lambda landmark_id: values.atVector(landmark_key(landmark_id)),
        )

    odometry_factors = 0
    try:
        for step, timestamp in enumerate(result.timestamps):
            first_step = np.searchsorted(result.timestamps, timestamp - window)
            builder.add_step(step, first_step=first_step)
            odometry_factors += step > first_step
            optimizer = gtsam.LevenbergMarquardtOptimizer(result.graph, result.values)
            record(first_step, step, 0, result.values)
            for iteration in range(1, solve_iter + 1):
                optimizer.iterate()
                record(first_step, step, iteration, optimizer.values())
            builder.store_estimate(optimizer.values())
    finally:
        history.save(output_dir, solver="GTSAM · LM", solve_iter=solve_iter, fps=gif_fps)

    active_count = len(builder.active_steps)
    print(f"MRCLAM  |  {result.timestamps[-1] - result.timestamps[0]:g} s"
          f"  |  {window:g} s window  |  {solve_iter} LM iterations per window"
          f"  |  {len(result.landmarks)} landmark priors")
    print(f"{'Robot':>5} {'Poses added':>12} {'Active':>7} {'Odometry':>10}"
          f" {'Robot obs.':>12} {'Landmark obs.':>15} {'XY RMSE (m)':>12}")
    for robot, poses in result.robot_states.items():
        robot_observations, landmark_observations = result.observation_counts[robot]
        gt_time, gt_x, gt_y, _ = builder.data.robots[robot].ground_truth.T
        gt_xy = np.column_stack((
            np.interp(result.timestamps, gt_time, gt_x),
            np.interp(result.timestamps, gt_time, gt_y),
        ))
        error_xy = np.asarray(poses)[:, :2] - gt_xy
        rmse = np.sqrt(np.mean(np.sum(error_xy**2, axis=1)))
        print(f"{robot:>5} {len(poses):>12} {active_count:>7} {odometry_factors:>10}"
              f" {robot_observations:>12} {landmark_observations:>15} {rmse:>12.3f}")
    return result


if __name__ == "__main__":
    main()
