"""Plot the latest GBP estimates, ground truth and factor measurements.

CSV timestamps are reconstructed using hz, start_offset and duration from the
neighbouring UTIAS dataset.py. These settings must match the CSV's solver run.
"""
from pathlib import Path
import argparse
import ast
import csv
import json
import sys

import numpy as np
import matplotlib

if __name__ == '__main__':
    # Open a native window even when the editor defaults to static inline plots.
    matplotlib.use('TkAgg')
    matplotlib.rcParams['toolbar'] = 'toolbar2'

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from load_data import load_data, prepare_data


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT.parents[1] / 'dataset' / 'MRCLAM1' / 'MRCLAM_Dataset1'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#AA4499', '#B58A00']
OPTION_NAMES = {1: 'Full', 2: 'Landmark only', 3: 'Robot only'}
# Edit these to set the initial figure size and axis limits; None uses map bounds.
FIGSIZE = (10, 8)
XLIM = None  # e.g. (-2, 8)
YLIM = None  # e.g. (-7, 7)

# Reuse the solver's exact time association and odometry integration when run by path.
if str(ROOT.parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT.parents[1]))
from build_graph import integrate_odometry, nearest_time_index


def run_settings():
    values = {'start_offset': 0.0, 'options': 1}
    tree = ast.parse((ROOT / 'UTIAS dataset.py').read_text(encoding='utf-8-sig'))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in ('hz', 'duration', 'start_offset', 'options'):
                    values[target.id] = ast.literal_eval(statement.value)
    return values


def load_history(path, variable_count):
    """Load complete iteration snapshots in the solver's variable order."""
    frames = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'iteration', 'variable_id', 'mean'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain iteration, variable_id and mean columns.')
        for row in reader:
            iteration, variable = int(row['iteration']), int(row['variable_id'])
            state = np.asarray(json.loads(row['mean']), dtype=float)
            if state.shape != (3, 3) or not np.isfinite(state).all():
                raise ValueError(f'Invalid SE(2) mean at iteration {iteration}, variable {variable}.')
            frame = frames.setdefault(iteration, {})
            if variable in frame:
                raise ValueError(f'Duplicate variable {variable} at iteration {iteration}.')
            frame[variable] = state
    if not frames:
        raise ValueError('CSV has no completed iterations.')
    expected = set(range(variable_count))
    last = max(frames)
    iterations, states = [], []
    for iteration in sorted(frames):
        frame = frames[iteration]
        if set(frame) != expected:
            if iteration == last and set(frame) < expected:
                print(f'Skipping incomplete final iteration {iteration} ({len(frame)}/{variable_count} states).')
                continue
            raise ValueError(
                f'Iteration {iteration}: CSV has {len(frame)} states; settings expect '
                f'{variable_count}. Use the same --hz and --duration as the solver run.'
            )
        iterations.append(iteration)
        states.append([frame[i] for i in range(variable_count)])
    if not states:
        raise ValueError('CSV has no completed iterations matching the run settings.')
    return np.asarray(iterations), np.asarray(states)


def measurement_geometry(data, poses, options=1):
    """Return actual factor connections and measurements at each GBP source pose.

    Connections end at the target robot node or the fixed landmark position.
    range_segments separately retain endpoints computed from range and bearing.
    The odometry factor's zero residual measurement encodes a relative motion Z;
    its measured next pose is X_i @ Z, including the measured heading change.
    """
    if options not in OPTION_NAMES:
        raise ValueError('options must be 1 (full), 2 (landmark-only), or 3 (robot-only).')
    mapping = {int(barcode): int(subject) for subject, barcode in data['barcodes']}
    robot_indices = {robot_id: r for r, robot_id in enumerate(data['robots'])}
    landmarks = {int(row[0]): row[1:3] for row in data['landmarks']}
    known_subjects = set(robot_indices) | set(landmarks)
    result = []
    for r, robot in enumerate(data['robots'].values()):
        segments, connections, robot_targets = [], [], []
        for time, barcode, distance, bearing in robot['measurements']:
            if int(barcode) not in mapping:
                raise ValueError(f'Unknown measurement barcode {int(barcode)} at {time}.')
            subject = mapping[int(barcode)]
            if subject not in known_subjects:
                raise ValueError(f'No robot or landmark for barcode {int(barcode)}.')
            is_robot = subject in robot_indices
            if (options == 2 and is_robot) or (options == 3 and not is_robot):
                continue
            index = nearest_time_index(data['node_times'], time)
            observer = poses[r, index]
            origin = observer[:2, 2]
            endpoint = origin + observer[:2, :2] @ (
                distance * np.array([np.cos(bearing), np.sin(bearing)])
            )
            segments.append([origin, endpoint])
            target = (poses[robot_indices[subject], index, :2, 2]
                      if is_robot else landmarks[subject])
            connections.append([origin, target])
            robot_targets.append(is_robot)
        predicted = np.asarray([
            pose @ integrate_odometry(robot['odometry'], start, end)
            for pose, start, end in zip(
                poses[r, :-1], data['node_times'][:-1], data['node_times'][1:],
            )
        ]).reshape(-1, 3, 3)
        result.append({
            'range_segments': np.asarray(segments).reshape(-1, 2, 2),
            'range_connections': np.asarray(connections).reshape(-1, 2, 2),
            'robot_targets': np.asarray(robot_targets, dtype=bool),
            'odometry_poses': predicted,
            'odometry_segments': np.stack((poses[r, :-1, :2, 2], predicted[:, :2, 2]), axis=1),
        })
    return result


def print_rmse(data, poses):
    """Print position RMSE in metres at the exact GBP node timestamps.

    RMSE = sqrt(mean((x_est - x_gt)^2 + (y_est - y_gt)^2)); heading is excluded.
    The total includes every robot and time node with equal weight.
    """
    truth = np.asarray([
        np.column_stack([
            np.interp(data['node_times'], robot['groundtruth'][:, 0], robot['groundtruth'][:, 1]),
            np.interp(data['node_times'], robot['groundtruth'][:, 0], robot['groundtruth'][:, 2]),
        ])
        for robot in data['robots'].values()
    ])
    squared_error = np.sum((np.asarray(poses)[..., :2, 2] - truth) ** 2, axis=-1)
    per_robot = {
        robot_id: float(np.sqrt(np.mean(errors)))
        for robot_id, errors in zip(data['robots'], squared_error)
    }
    total = float(np.sqrt(np.mean(squared_error)))
    print('Position RMSE (x, y) [m]:')
    for robot_id, value in per_robot.items():
        print(f'  R{robot_id}: {value:.6f}')
    print(f'  All robots: {total:.6f}', flush=True)
    return per_robot, total


def plot_estimates(data, poses, *, options=1, iteration=None, raw_data=None, show=False,
                   solver_name='GBP'):
    """Plot supplied solver states using its prepared data and observation option."""
    if show:
        plt.switch_backend('TkAgg')
        matplotlib.rcParams['toolbar'] = 'toolbar2'
    robot_ids = list(data['robots'])  # Same order as build_graph.add_variables.
    poses = np.asarray(poses).reshape(len(robot_ids), len(data['node_times']), 3, 3)
    estimates = poses[..., :2, 2]
    measurements = measurement_geometry(data, poses, options=options)
    map_data = data if raw_data is None else raw_data

    # Keep the landmark map in one standard, interactive Matplotlib axes.
    landmarks = data['landmarks']
    all_positions = np.vstack([
        landmarks[:, 1:3], estimates.reshape(-1, 2),
        *[robot['groundtruth'][:, 1:3] for robot in map_data['robots'].values()],
        *[item['range_segments'][:, 1] for item in measurements],
        *[item['odometry_poses'][:, :2, 2] for item in measurements],
    ])
    low, high = all_positions.min(axis=0) - .7, all_positions.max(axis=0) + .7
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='#F7F9FC')
    ax.set_facecolor('white')
    ax.set(xlim=XLIM if XLIM is not None else (low[0], high[0]),
           ylim=YLIM if YLIM is not None else (low[1], high[1]),
           xlabel='x [m]', ylabel='y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#DFE5ED', linewidth=.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#CFD7E2')
    ax.scatter(landmarks[:, 1], landmarks[:, 2], marker='^', s=75,
               color='#536375', edgecolor='white', zorder=4)
    for subject, x, y, *_ in landmarks:
        ax.annotate(f'L{int(subject)}', (x, y), xytext=(6, 6),
                    textcoords='offset points', fontsize=9, color='#536375')
    duration = data['end_time'] - data['start_time']
    ax.set_title(f'{solver_name}, GT and factor connections | {OPTION_NAMES[options]}\n'
                 f'Iteration {iteration} | duration {duration:g} s')
    estimate_lines, truth_lines = [], []
    for r, (robot_id, color) in enumerate(zip(robot_ids, COLORS)):
        measurement = measurements[r]
        segments = measurement['range_segments']
        connections = measurement['range_connections']
        robot_targets = measurement['robot_targets']
        for mask, style in ((~robot_targets, 'solid'), (robot_targets, 'dashed')):
            ax.add_collection(LineCollection(connections[mask], colors=color,
                                             linewidths=.9, linestyles=style,
                                             alpha=.25, zorder=2))
        ax.scatter(segments[:, 1, 0], segments[:, 1, 1], marker='+', s=18,
                   linewidths=.8, color=color, alpha=.5, zorder=3)
        odometry = measurement['odometry_poses']
        ax.add_collection(LineCollection(measurement['odometry_segments'], colors=color,
                                         linewidths=1.3, linestyles='dotted',
                                         alpha=.75, zorder=3))
        ax.scatter(odometry[:, 0, 2], odometry[:, 1, 2], marker='x', s=24,
                   linewidths=1., color=color, alpha=.8, zorder=4)
        gt = data['robots'][robot_id]['groundtruth']
        # Use the full GT path within the selected window, including its boundaries.
        inside = (gt[:, 0] > data['start_time']) & (gt[:, 0] < data['end_time'])
        times = np.r_[data['start_time'], gt[inside, 0], data['end_time']]
        truth_x = np.interp(times, gt[:, 0], gt[:, 1])
        truth_y = np.interp(times, gt[:, 0], gt[:, 2])
        truth_line, = ax.plot(truth_x, truth_y, '--', color=color,
                              linewidth=1.5, alpha=.65, zorder=2, label=f'R{robot_id} GT')
        estimate_line, = ax.plot(estimates[r, :, 0], estimates[r, :, 1], '-o',
                                 color=color, linewidth=1.8, markersize=3.5,
                                 zorder=5, label=f'R{robot_id} {solver_name}')
        estimate_lines.append(estimate_line)
        truth_lines.append(truth_line)

    factor_handles = []
    if options in (1, 2):
        factor_handles.append(Line2D([], [], color='#536375', linewidth=1.,
                                     label='Landmark factor connection'))
    if options in (1, 3):
        factor_handles.append(Line2D([], [], color='#536375', linewidth=1., linestyle='--',
                                     label='Robot factor connection'))
    legend = ax.legend(handles=estimate_lines + truth_lines + factor_handles + [
        Line2D([], [], color='#536375', linestyle='None', marker='+',
               label='Range-bearing measured position'),
        Line2D([], [], color='#536375', linewidth=1.3, linestyle=':', marker='x',
               label='Odometry: predicted next pose'),
    ], loc='upper left', bbox_to_anchor=(1.02, 1.), fontsize=8, framealpha=.9)
    legend.set_draggable(True)
    fig.tight_layout()
    if show:
        fig.canvas.manager.set_window_title(f'UTIAS {solver_name} | Pan and zoom')
        plt.show(block=True)
    return fig, ax


def plot(args):
    """Standalone CSV entry point; main() can instead pass its solved states directly."""
    raw_data = load_data(args.data)
    data = prepare_data(raw_data, args.hz, args.duration, start_offset=args.start_offset)
    iterations, states = load_history(args.csv, len(data['robots']) * len(data['node_times']))
    poses = states[-1].reshape(len(data['robots']), len(data['node_times']), 3, 3)
    print_rmse(data, poses)
    return plot_estimates(data, poses, options=getattr(args, 'options', run_settings()['options']),
                          iteration=iterations[-1], raw_data=raw_data)


def main():
    settings = run_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'history.csv')
    parser.add_argument('--data', type=Path, default=DATASET_DIR)
    parser.add_argument('--hz', type=float, default=settings['hz'])
    parser.add_argument('--duration', type=float, default=settings['duration'])
    parser.add_argument('--start-offset', type=float, default=settings['start_offset'])
    parser.add_argument('--options', type=int, choices=(1, 2, 3), default=settings['options'])
    args = parser.parse_args()
    fig, _ = plot(args)
    fig.canvas.manager.set_window_title('UTIAS GBP | Pan and zoom')
    plt.show(block=True)


if __name__ == '__main__':
    main()
