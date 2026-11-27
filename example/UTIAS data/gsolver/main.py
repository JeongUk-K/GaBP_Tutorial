from pathlib import Path

import numpy as np

from build_graph import prepare_graph
from plot_results import RunHistory
from utias_data import planar_pose


# Parameters
dataset_dir = Path(__file__).resolve().parents[3] / "dataset" / "MRCLAM1" / "MRCLAM_Dataset1"
start_time = 200.0  # 공통 odometry 및 ground-truth 구간의 시작으로부터 지난 시간 [s]. 최대 1487.796
duration = 100.0  # sec
window = 30  # sec
solve_iter = 10  # 각 window에서의 반복 최적화 횟수
message_weight = 0.25  # 새 pose-to-factor 메시지의 가중치.
odom_dt = 0.5  # Pose와 odometry factor의 간격 [s].
landmark_dt = 1.0  # 각 기준 시각 ± 반 간격 안에서 같은 랜드마크의 가장 가까운 관측을 사용 [s].
output_dir = Path(__file__).resolve().parent / "results"
gif_fps = 15

# Pose covariances use tangent order [tx, ty, tz, rx, ry, rz], with velocity last for the 12D state.
state_covariance = 0.1 * np.eye(12)
robot_prior_covariance = 0.01**2 * np.eye(6)
landmark_prior_covariance = 0.01**2 * np.eye(6)
odometry_covariance = np.diag(np.square([0.05, 0.01, 0.01, 0.01, 0.01, np.deg2rad(2.5)]))

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
        state_covariance=state_covariance,
        robot_prior_covariance=robot_prior_covariance,
        landmark_prior_covariance=landmark_prior_covariance,
        odometry_covariance=odometry_covariance,
        robot_measurement_covariance=robot_measurement_covariance,
        landmark_measurement_covariance=landmark_measurement_covariance,
    )
    result = builder.result
    planar = np.ix_([0, 1, 5], [0, 1, 5])

    def read_pose(robot_id, step):
        return planar_pose(result.robot_states[robot_id][step].mu.pose)

    def read_landmark(landmark_id):
        return planar_pose(result.landmarks[landmark_id].mu.pose)

    history = RunHistory(
        result.timestamps, builder.robots,
        {landmark_id: read_landmark(landmark_id) for landmark_id in result.landmarks},
        builder.observations,
        robot_prior_covariance=robot_prior_covariance[planar],
        landmark_prior_covariance=landmark_prior_covariance[planar],
        odometry_covariance=odometry_covariance[planar],
    )
    odometry_factors = 0
    try:
        for step, timestamp in enumerate(result.timestamps):
            cutoff = timestamp - window
            first_step = np.searchsorted(result.timestamps, cutoff)
            result.graph.prune_before(cutoff)
            previous_in_window = step > first_step
            builder.add_step(step, connect_odometry=previous_in_window)
            odometry_factors += previous_in_window
            history.record(first_step, step, 0, read_pose, read_landmark)
            for iteration in range(1, solve_iter + 1):
                result.graph.iterate(message_weight)
                history.record(first_step, step, iteration, read_pose, read_landmark)
    finally:
        history.save(output_dir, solver="gsolver · GBP", solve_iter=solve_iter, fps=gif_fps)

    final_window_count = np.count_nonzero(result.timestamps >= result.timestamps[-1] - window)
    print(f"MRCLAM  |  {result.timestamps[-1] - result.timestamps[0]:g} s"
          f"  |  {window:g} s window  |  {solve_iter} GBP iterations per window"
          f"  |  {len(result.landmarks)} landmark priors")
    print(f"{'Robot':>5} {'Poses added':>12} {'Active':>7} {'Odometry':>10}"
          f" {'Robot obs.':>12} {'Landmark obs.':>15}")
    for robot, poses in result.robot_states.items():
        robot_observations, landmark_observations = result.observation_counts[robot]
        print(f"{robot:>5} {len(poses):>12} {final_window_count:>7} {odometry_factors:>10}"
              f" {robot_observations:>12} {landmark_observations:>15}")
    return result


if __name__ == "__main__":
    main()
