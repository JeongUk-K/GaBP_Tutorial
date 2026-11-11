from pathlib import Path

import numpy as np


def load_data(dataset_dir):
    """Read the original MRCLAM tables without changing timestamps or values.

    Robot groundtruth: [time, x, y, theta].
    Robot odometry: [time, forward_velocity, angular_velocity].
    Robot measurements: [time, barcode, range, bearing].
    Landmarks: [subject_id, x, y, x_std, y_std].
    Barcodes: [subject_id, barcode].
    """
    dataset_dir = Path(dataset_dir)
    robots = {}

    for robot_id in range(1, 6):
        robots[robot_id] = {
            'groundtruth': np.loadtxt(
                dataset_dir / f'Robot{robot_id}_Groundtruth.dat', ndmin=2,
            ),
            'odometry': np.loadtxt(
                dataset_dir / f'Robot{robot_id}_Odometry.dat', ndmin=2,
            ),
            'measurements': np.loadtxt(
                dataset_dir / f'Robot{robot_id}_Measurement.dat', ndmin=2,
            ),
        }

    return {
        'robots': robots,
        'landmarks': np.loadtxt(
            dataset_dir / 'Landmark_Groundtruth.dat', ndmin=2,
        ),
        'barcodes': np.loadtxt(
            dataset_dir / 'Barcodes.dat', dtype=int, ndmin=2,
        ),
    }


def prepare_data(raw_data, hz, duration):
    """Create node times and select a window from the shared state-data interval.

    Input tables must be ordered by time. All returned times are absolute.
    State tables retain boundary support samples; measurements stay in the window.
    """
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError('hz must be finite and positive.')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive.')

    state_tables = [
        robot[name]
        for robot in raw_data['robots'].values()
        for name in ('odometry', 'groundtruth')
    ]
    start_time = max(table[0, 0] for table in state_tables)
    available_end = min(table[-1, 0] for table in state_tables)
    end_time = start_time + duration
    if start_time >= available_end:
        raise ValueError('Robot state data have no shared time interval.')
    if end_time > available_end:
        raise ValueError(
            f'duration exceeds the shared interval of {available_end - start_time:.6f} seconds.'
        )

    # Include the endpoint only when it lies on the regular node-time grid.
    offsets = np.arange(int(np.ceil(duration * hz)) + 1) / hz
    offsets = offsets[offsets <= np.nextafter(float(duration), np.inf)]
    node_times = start_time + offsets

    robots = {}
    for robot_id, robot in raw_data['robots'].items():
        prepared = {}
        for name in ('odometry', 'groundtruth'):
            table = robot[name]
            times = table[:, 0]
            # Keep support samples for later integration or interpolation.
            first = max(0, int(np.searchsorted(times, start_time, side='right')) - 1)
            last = min(len(table), int(np.searchsorted(times, end_time, side='left')) + 1)
            prepared[name] = table[first:last].copy()

        measurements = robot['measurements']
        in_window = (
            (measurements[:, 0] >= start_time)
            & (measurements[:, 0] <= end_time)
        )
        prepared['measurements'] = measurements[in_window].copy()
        robots[robot_id] = prepared

    return {
        'start_time': start_time,
        'end_time': end_time,
        'node_times': node_times,
        'robots': robots,
        'landmarks': raw_data['landmarks'].copy(),
        'barcodes': raw_data['barcodes'].copy(),
    }
