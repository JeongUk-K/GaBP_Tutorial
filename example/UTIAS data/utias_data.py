from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROBOT_IDS = (1, 2, 3, 4, 5)


@dataclass
class RobotData:
    ground_truth: np.ndarray
    odometry: np.ndarray
    measurements: np.ndarray


@dataclass
class Observation:
    robot_id: int
    subject: int
    row: int
    measurement: np.ndarray
    covariance: np.ndarray


def read_table(path, columns, *, time_series=False):
    rows = np.loadtxt(path, ndmin=2)
    if not rows.size or rows.shape[1] != columns or not np.isfinite(rows).all():
        raise ValueError(f"Invalid data table {path}")
    if time_series and np.any(np.diff(rows[:, 0]) < 0):
        raise ValueError(f"Unsorted timestamps in {path}")
    return rows


def load_robot_data(dataset_dir, robot_id):
    prefix = f"Robot{robot_id}"
    return RobotData(
        ground_truth=read_table(dataset_dir / f"{prefix}_Groundtruth.dat", 4, time_series=True),
        odometry=read_table(dataset_dir / f"{prefix}_Odometry.dat", 3, time_series=True),
        measurements=read_table(dataset_dir / f"{prefix}_Measurement.dat", 4, time_series=True),
    )


def checked_covariance(value, size, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be a finite symmetric {size}x{size} matrix")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def pose_times(robots, start_time, duration, odom_dt):
    if not np.isfinite(start_time) or start_time < 0:
        raise ValueError("start_time must be finite and nonnegative")
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(odom_dt) or odom_dt <= 0:
        raise ValueError("duration and odom_dt must be finite and positive")

    state_tables = [table for robot in robots.values() for table in (robot.ground_truth, robot.odometry)]
    start = max(table[0, 0] for table in state_tables) + start_time
    end = min(table[-1, 0] for table in state_tables)
    if start + duration > end:
        raise ValueError("Requested interval exceeds the common ground-truth and odometry coverage")
    offsets = np.arange(0.0, np.nextafter(duration, np.inf), odom_dt)
    return start + offsets


def transform(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0, x],
        [s, c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def integrate_odometry(odometry, start, end):
    times = odometry[:, 0]
    first = np.searchsorted(times, start, side="right")
    last = np.searchsorted(times, end, side="left")

    # Hold each recorded command until the next sample and integrate its constant-curvature arc.
    dt = np.diff(np.r_[start, times[first:last], end])
    speed, turn_rate = odometry[first - 1:last, 1:].T
    heading_change = turn_rate * dt
    heading = np.cumsum(heading_change) - 0.5 * heading_change
    distance = speed * dt * np.sinc(heading_change / (2.0 * np.pi))
    return transform(distance @ np.cos(heading), distance @ np.sin(heading), heading_change.sum())


def nearest_state(timestamps, time):
    right = np.searchsorted(timestamps, time)
    if right and (right == len(timestamps) or time - timestamps[right - 1] <= timestamps[right] - time):
        return right - 1
    return right


def prepare_observations(robots, landmarks, barcodes, timestamps, *, landmark_dt, robot_covariance,
                         landmark_covariance):
    barcode_subject = {int(barcode): int(subject) for subject, barcode in barcodes}
    by_step = [[] for _ in timestamps]
    counts = {}
    start, end = timestamps[0], timestamps[-1]
    landmark_times = start + np.arange(landmark_dt, np.nextafter(end, np.inf) - start, landmark_dt)
    landmark_times = landmark_times[landmark_times <= end]
    landmark_steps = [nearest_state(timestamps, time) for time in landmark_times]
    # Share each boundary between adjacent half-open windows to assign it only once.
    landmark_boundaries = start + (np.arange(len(landmark_times) + 1) + 0.5) * landmark_dt

    for robot_id in ROBOT_IDS:
        samples = robots[robot_id].measurements
        first = np.searchsorted(samples[:, 0], start, side="left")
        last = np.searchsorted(samples[:, 0], end, side="right")
        observations = []
        closest_landmarks = {}
        for row, (time, barcode, distance, _) in enumerate(samples[first:last], start=first):
            subject = barcode_subject.get(int(barcode))
            if subject == robot_id or distance <= 0:
                continue
            if subject in robots:
                observations.append((row, nearest_state(timestamps, time), subject))
            elif subject in landmarks:
                slot = np.searchsorted(landmark_boundaries, time, side="right") - 1
                if not 0 <= slot < len(landmark_times):
                    continue
                key = (slot, subject)
                gap = abs(time - landmark_times[slot])
                previous = closest_landmarks.get(key)
                if previous is None or gap < previous[0]:
                    closest_landmarks[key] = (gap, row)

        robot_count = len(observations)
        observations.extend((row, landmark_steps[slot], subject)
                            for (slot, subject), (_, row) in closest_landmarks.items())
        for row, step, subject in observations:
            covariance = robot_covariance if subject in robots else landmark_covariance
            by_step[step].append(Observation(robot_id, subject, row, samples[row, 2:], covariance))
        counts[robot_id] = (robot_count, len(closest_landmarks))
    return by_step, counts


@dataclass
class UTIASData:
    timestamps: np.ndarray
    robots: dict[int, RobotData]
    landmark_rows: np.ndarray
    observations: list[list[Observation]]
    observation_counts: dict[int, tuple[int, int]]


def load_dataset(dataset_dir, *, start_time, duration, odom_dt, landmark_dt,
                 robot_measurement_covariance, landmark_measurement_covariance):
    if not np.isfinite(landmark_dt) or landmark_dt <= 0:
        raise ValueError("landmark_dt must be finite and positive")
    robot_covariance = checked_covariance(robot_measurement_covariance, 2, "robot_measurement_covariance")
    landmark_covariance = checked_covariance(landmark_measurement_covariance, 2, "landmark_measurement_covariance")

    dataset_dir = Path(dataset_dir)
    robots = {robot_id: load_robot_data(dataset_dir, robot_id) for robot_id in ROBOT_IDS}
    timestamps = pose_times(robots, start_time, duration, odom_dt)
    landmark_rows = read_table(dataset_dir / "Landmark_Groundtruth.dat", 5)
    observations, counts = prepare_observations(
        robots, {int(subject) for subject in landmark_rows[:, 0]},
        read_table(dataset_dir / "Barcodes.dat", 2), timestamps,
        landmark_dt=landmark_dt, robot_covariance=robot_covariance, landmark_covariance=landmark_covariance,
    )
    return UTIASData(timestamps, robots, landmark_rows, observations, counts)


def initial_pose(data, timestamp):
    gt_time, gt_x, gt_y, gt_yaw = data.ground_truth.T
    return transform(
        np.interp(timestamp, gt_time, gt_x),
        np.interp(timestamp, gt_time, gt_y),
        np.interp(timestamp, gt_time, np.unwrap(gt_yaw)),
    )


def planar_pose(pose):
    return np.array([pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0])])


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotate_to_local(xy, yaw):
    x, y = np.moveaxis(xy, -1, 0)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c * x + s * y, -s * x + c * y), axis=-1)


def pose_error(prediction, measurement):
    delta = prediction - measurement
    xy = rotate_to_local(delta[..., :2], measurement[..., 2])
    return np.concatenate((xy, wrap(delta[..., 2:3])), axis=-1)


def quadratic_cost(residual, precision):
    return 0.5 * np.einsum("...i,...ij,...j->...", residual, precision, residual).sum()


class WindowCost:
    """Evaluate the active graph's planar objective at the reported pose estimates."""

    def __init__(self, timestamps, robots, landmark_priors, observations, *,
                 robot_prior_covariance, landmark_prior_covariance, odometry_covariance):
        robot_indices = {robot_id: index for index, robot_id in enumerate(robots)}
        subject_indices = robot_indices | {
            landmark_id: len(robots) + index for index, landmark_id in enumerate(landmark_priors)
        }
        self.robot_priors = np.array([planar_pose(initial_pose(robot, timestamps[0])) for robot in robots.values()])
        self.landmark_priors = np.array(list(landmark_priors.values()))
        self.robot_precision = np.linalg.inv(robot_prior_covariance)
        self.landmark_precision = np.linalg.inv(landmark_prior_covariance)
        self.odom_precision = np.linalg.inv(odometry_covariance)
        self.motions = np.zeros((len(timestamps), len(robots), 3))
        for step in range(1, len(timestamps)):
            for index, robot in enumerate(robots.values()):
                motion = integrate_odometry(robot.odometry, timestamps[step - 1], timestamps[step])
                self.motions[step, index] = planar_pose(motion)

        self.offsets = np.r_[0, np.cumsum([len(items) for items in observations])]
        selected = [(step, item) for step, items in enumerate(observations) for item in items]
        self.observation_steps = np.array([step for step, _ in selected], dtype=int)
        self.observers = np.array([robot_indices[item.robot_id] for _, item in selected], dtype=int)
        self.targets = np.array([subject_indices[item.subject] for _, item in selected], dtype=int)
        self.measurements = np.array([item.measurement for _, item in selected]).reshape(-1, 2)
        self.obs_precision = np.linalg.inv(
            np.array([item.covariance for _, item in selected]).reshape(-1, 2, 2)
        )

    def evaluate(self, first, last, poses, landmarks):
        start, stop = self.offsets[first], self.offsets[last + 1]
        factor_count = len(landmarks) + poses.shape[1] * (last - first + (first == 0)) + stop - start
        if not np.isfinite(poses).all() or not np.isfinite(landmarks).all():
            return np.nan, factor_count

        with np.errstate(over="ignore", invalid="ignore"):
            cost = quadratic_cost(pose_error(landmarks, self.landmark_priors), self.landmark_precision)
            if first == 0:
                cost += quadratic_cost(pose_error(poses[0], self.robot_priors), self.robot_precision)

            delta = np.diff(poses, axis=0)
            translation = rotate_to_local(delta[..., :2], poses[:-1, :, 2])
            prediction = np.concatenate((translation, delta[..., 2:3]), axis=-1)
            cost += quadratic_cost(
                pose_error(prediction, self.motions[first + 1:last + 1]), self.odom_precision,
            )

            active = slice(start, stop)
            steps = self.observation_steps[active] - first
            all_poses = np.concatenate((poses, np.broadcast_to(landmarks, (len(poses), *landmarks.shape))), axis=1)
            observer = poses[steps, self.observers[active]]
            target = all_poses[steps, self.targets[active]]
            dx, dy = (target[:, :2] - observer[:, :2]).T
            distance = np.hypot(dx, dy)
            measurement = self.measurements[active]
            residual = np.column_stack((
                distance - measurement[:, 0],
                wrap(np.arctan2(dy, dx) - observer[:, 2] - measurement[:, 1]),
            ))
            residual[distance < 1e-9] = 0.0
            cost += quadratic_cost(residual, self.obs_precision[active])
        return cost, factor_count
