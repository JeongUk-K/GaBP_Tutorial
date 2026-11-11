"""Render MRCLAM1 ground-truth motion and recorded range/bearing observations.

Requires numpy, matplotlib and imageio-ffmpeg. Run this file from any directory.
Measurement endpoints are computed from the observer's interpolated ground-truth
pose at the measurement timestamp; they are not snapped to true target positions.
"""

from pathlib import Path
import argparse
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent
COLORS = ["#0072B2", "#D55E00", "#009E73", "#AA4499", "#B58A00"]


def interpolate_pose(gt, timestamps):
    return np.column_stack([
        np.interp(timestamps, gt[:, 0], gt[:, 1]),
        np.interp(timestamps, gt[:, 0], gt[:, 2]),
        np.interp(timestamps, gt[:, 0], np.unwrap(gt[:, 3])),
    ])


def load_dataset(folder):
    barcodes = np.loadtxt(folder / "Barcodes.dat", ndmin=2)
    mapping = {int(barcode): int(subject) for subject, barcode in barcodes}
    landmarks = np.loadtxt(folder / "Landmark_Groundtruth.dat", ndmin=2)
    robots = []
    for robot_id in range(1, 6):
        gt = np.loadtxt(folder / f"Robot{robot_id}_Groundtruth.dat", ndmin=2)
        measurements = np.loadtxt(folder / f"Robot{robot_id}_Measurement.dat", ndmin=2)
        for name, array in [("ground truth", gt), ("measurements", measurements)]:
            if not np.isfinite(array).all() or np.any(np.diff(array[:, 0]) < 0):
                raise ValueError(f"Robot {robot_id}: invalid {name}")
        valid = (measurements[:, 0] >= gt[0, 0]) & (measurements[:, 0] <= gt[-1, 0])
        if not valid.all():
            raise ValueError(f"Robot {robot_id}: measurement outside pose coverage")
        subjects = np.array([mapping.get(int(b), -1) for b in measurements[:, 1]])
        poses = interpolate_pose(gt, measurements[:, 0])
        direction = poses[:, 2] + measurements[:, 3]
        endpoints = poses[:, :2] + measurements[:, 2, None] * np.column_stack([
            np.cos(direction), np.sin(direction)
        ])
        robots.append(dict(gt=gt, measurements=measurements, subjects=subjects,
                           segments=np.stack([poses[:, :2], endpoints], axis=1)))
    return landmarks, robots


def render(args):
    landmarks, robots = load_dataset(args.data)
    start = max(robot["gt"][0, 0] for robot in robots)
    end = min(robot["gt"][-1, 0] for robot in robots)
    duration = end - start
    frame_count = math.ceil(duration * args.fps / args.speed) + 1
    timestamps = np.minimum(start + np.arange(frame_count) * args.speed / args.fps, end)
    all_positions = np.vstack([landmarks[:, 1:3]] + [r["gt"][:, 1:3] for r in robots])
    low, high = all_positions.min(axis=0) - .7, all_positions.max(axis=0) + .7

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#233248", "axes.labelcolor": "#233248"})
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="#F7F9FC")
    ax = fig.add_axes([.055, .13, .60, .76], facecolor="white")
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#DFE5ED", linewidth=.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#CFD7E2")
    ax.scatter(landmarks[:, 1], landmarks[:, 2], marker="^", s=75,
               color="#536375", edgecolor="white", zorder=4)
    for subject, x, y, *_ in landmarks:
        ax.annotate(f"L{int(subject)}", (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color="#536375")

    fig.text(.055, .948, "MRCLAM 1  |  Multi-robot observations", fontsize=23, weight="bold")
    clock = fig.text(.055, .906, "", fontsize=13)
    fig.text(.70, .87, "LATEST OBSERVATION", fontsize=12, weight="bold")
    fig.text(.70, .842, "Target ID  |  range  |  bearing", fontsize=11, color="#65758B")
    fig.text(.70, .235, "HOW TO READ THE MAP", fontsize=12, weight="bold")
    fig.text(.70, .208, "Robot position and heading: ground truth\n"
             "Fading rays: recorded range + bearing\n"
             "Ray origins: pose at acquisition time\n"
             "Ray endpoints: measured position", fontsize=10.5, linespacing=1.65, va="top")
    fig.text(.70, .045, "Rays persist for 1 s of dataset time.\n"
             "Trails show the last 30 s of motion.\n"
             "Unmapped barcode 43: unknown target.", fontsize=10, linespacing=1.5,
             color="#65758B")
    progress_bg = fig.add_axes([.055, .055, .60, .012])
    progress_bg.set(xlim=(0, duration), ylim=(0, 1))
    progress_bg.axis("off")
    progress_bg.axhspan(0, 1, color="#DCE3EC")
    progress = progress_bg.barh(.5, 0, height=1, color="#233248")[0]
    fig.text(.055, .027, "UTIAS MRCLAM Dataset 1  |  5 robots  |  15 landmarks", fontsize=10,
             color="#65758B")
    count_text = fig.text(.70, .275, "", fontsize=11, color="#65758B")

    handles = [Line2D([], [], color="#536375", lw=1.5, label="Landmark observation"),
               Line2D([], [], color="#536375", lw=1.5, ls="--", label="Robot observation"),
               Line2D([], [], color="#536375", lw=1.5, ls=":", label="Unknown target")]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=.9)

    artists = []
    for i, color in enumerate(COLORS):
        trail, = ax.plot([], [], color=color, lw=1.6, alpha=.5, zorder=2)
        rays = []
        for style in ["solid", "dashed", "dotted"]:
            collection = LineCollection([], colors=color, linewidths=1.3,
                                        linestyles=style, zorder=3)
            ax.add_collection(collection)
            rays.append(collection)
        endpoints = ax.scatter([], [], marker="+", s=20, linewidths=.9, color=color, zorder=4)
        body = Circle((0, 0), radius=.115, facecolor=color, edgecolor="white", lw=1.5, zorder=6)
        heading = Polygon(np.zeros((3, 2)), facecolor=color, edgecolor="white", lw=.8, zorder=7)
        ax.add_patch(body)
        ax.add_patch(heading)
        label = ax.text(0, 0, f"R{i+1}", color=color, weight="bold", fontsize=11, zorder=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=1))
        top = .785 - i * .103
        fig.text(.70, top, f"ROBOT {i+1}", color=color, weight="bold", fontsize=13)
        detail = fig.text(.70, top-.027, "", fontsize=11, family="DejaVu Sans Mono")
        tally = fig.text(.70, top-.049, "", fontsize=10, color="#65758B")
        artists.append(dict(trail=trail, rays=rays, endpoints=endpoints, body=body,
                            heading=heading, label=label, detail=detail, tally=tally))

    frame_poses = [interpolate_pose(robot["gt"], timestamps) for robot in robots]
    # Boundaries give every measurement a frame, with fading in dataset time.
    seen = [np.zeros(len(robot["measurements"]), dtype=bool) for robot in robots]
    total_measurements = sum(len(robot["measurements"]) for robot in robots)
    moving_artists = [clock, count_text, progress]
    for art in artists:
        moving_artists.extend([art["trail"], *art["rays"], art["endpoints"],
                               art["body"], art["heading"], art["label"],
                               art["detail"], art["tally"]])
    moving_artists.append(legend)
    for artist in moving_artists:
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    draw_order = sorted(moving_artists, key=lambda artist: artist.get_zorder())

    def update(frame):
        now = timestamps[frame]
        clock.set_text(f"Dataset time  {now-start:7.1f} / {duration:.1f} s     |     "
                       f"Playback  {args.speed:g}x     |     {args.fps} fps")
        progress.set_width(now-start)
        total = 0
        for i, (robot, art) in enumerate(zip(robots, artists)):
            gt, measurements, subjects = robot["gt"], robot["measurements"], robot["subjects"]
            x, y, theta = frame_poses[i][frame]
            art["body"].center = x, y
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            triangle = np.array([[.31, 0], [.075, .085], [.075, -.085]])
            art["heading"].set_xy(triangle @ rotation.T + [x, y])
            art["label"].set_position((x+.16, y+.18))
            a, b = np.searchsorted(gt[:, 0], [now-30., now], side="right")
            trail = np.vstack([gt[a:b, 1:3], [x, y]])[::4]
            art["trail"].set_data(trail[:, 0], trail[:, 1])
            lo, hi = np.searchsorted(measurements[:, 0], [now-1., now], side="right")
            seen[i][lo:hi] = True
            total += hi
            active = subjects[lo:hi]
            alpha = .18 + .67 * np.clip(1. - (now-measurements[lo:hi, 0]), 0., 1.)
            for collection, mask in zip(art["rays"],
                                        [active > 5, (active >= 1) & (active <= 5), active < 0]):
                collection.set_segments(robot["segments"][lo:hi][mask])
                collection.set_alpha(alpha[mask] if np.any(mask) else np.ones(1))
            art["endpoints"].set_offsets(robot["segments"][lo:hi, 1])
            art["endpoints"].set_alpha(alpha if len(alpha) else np.ones(1))
            if hi:
                measurement = measurements[hi-1]
                target = subjects[hi-1]
                target_name = (f"R{target}" if target <= 5 else f"L{target}") if target > 0 else f"?{int(measurement[1])}"
                art["detail"].set_text(f"{target_name:>3s}   {measurement[2]:5.2f} m   {np.degrees(measurement[3]):+6.1f} deg")
                art["tally"].set_text(f"{hi:,} observations  |  last {now-measurement[0]:.1f} s ago")
            else:
                art["detail"].set_text("Waiting for first observation")
                art["tally"].set_text("0 observations")
        count_text.set_text(f"Recorded observations: {total:,} / {total_measurements:,}")
        fig.canvas.restore_region(background)
        for artist in draw_order:
            fig.draw_artist(artist)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    update(int(np.argmin(np.abs(timestamps - (start + 300)))))
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(args.preview)
    if args.preview_only:
        return
    for mask in seen:
        mask[:] = False
    writer = imageio_ffmpeg.write_frames(str(args.output), (1600, 1000), fps=args.fps,
                codec="libx264", quality=7, macro_block_size=2,
                output_params=["-movflags", "+faststart", "-preset", "veryfast"])
    writer.send(None)
    started = time.monotonic()
    try:
        for frame in range(frame_count):
            update(frame)
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.send(rgb)
            if frame % 300 == 0 or frame == frame_count-1:
                print(f"Frame {frame+1}/{frame_count}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    finally:
        writer.close()
        plt.close(fig)
    for i, mask in enumerate(seen, 1):
        if not mask.all():
            raise RuntimeError(f"Robot {i}: {np.sum(~mask)} measurements were not visualized")
    decoded_frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(args.output))
    if decoded_frames != frame_count:
        raise RuntimeError(f"Decoded {decoded_frames} frames; expected {frame_count}")
    print(f"Verified {decoded_frames} frames, {seconds:.3f}s, all {total_measurements} measurements visualized.")
    print(args.output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "MRCLAM_Dataset1")
    parser.add_argument("--output", type=Path, default=ROOT / "mrclam1_measurements.mp4")
    parser.add_argument("--speed", type=float, default=10.)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.speed <= 0 or arguments.fps <= 0:
        parser.error("speed and fps must be positive")
    render(arguments)
