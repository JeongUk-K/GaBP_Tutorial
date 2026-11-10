import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse


def plot_history(history, title='GBP', interval=300):
    """Animate 2D variable means and one-standard-deviation covariance ellipses."""
    if not history:
        raise ValueError('history is empty')

    means = np.array([
        [mean.ravel() for mean in step['means']] for step in history
    ])
    covs = np.array([step['covs'] for step in history])
    stds = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    lower = (means - stds).min(axis=(0, 1))
    upper = (means + stds).max(axis=(0, 1))
    margin = 0.1 * np.maximum(upper - lower, 1.0)

    fig, ax = plt.subplots()
    ax.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    ax.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    points, ellipses = [], []
    for i in range(means.shape[1]):
        point, = ax.plot([], [], 'o', label=f'x{i + 1}')
        ellipse = Ellipse((0, 0), 0, 0, fill=False,
                          edgecolor=point.get_color())
        ax.add_patch(ellipse)
        points.append(point)
        ellipses.append(ellipse)
    ax.legend()

    def update(frame):
        ax.set_title(f"{title} | iter {history[frame]['iter']} | 1-sigma ellipses")
        for i, (point, ellipse) in enumerate(zip(points, ellipses)):
            mean = means[frame, i]
            cov = covs[frame, i]
            values, vectors = np.linalg.eigh((cov + cov.T) / 2)
            point.set_data([mean[0]], [mean[1]])
            ellipse.set_center(mean)
            ellipse.set_width(2 * np.sqrt(max(values[1], 0)))
            ellipse.set_height(2 * np.sqrt(max(values[0], 0)))
            ellipse.set_angle(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))

    update(0)
    animation = FuncAnimation(fig, update, frames=len(history),
                              interval=interval, repeat=False)
    fig.tight_layout()
    return fig, animation


if __name__ == '__main__':
    from .toy_example import history_ad, history_numerical, history_analytic

    animations = [
        plot_history(history_ad, 'AD'),
        plot_history(history_numerical, 'Numerical'),
        plot_history(history_analytic, 'Analytic'),
    ]
    plt.show()
