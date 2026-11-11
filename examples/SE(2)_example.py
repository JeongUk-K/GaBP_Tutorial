import numpy as np
import jax
import jax.numpy as jnp

from my_gbp import FactorGraph
from my_gbp.manifold import SE2


def pose(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def wrap_angle(angle, xp=np):
    return xp.arctan2(xp.sin(angle), xp.cos(angle))


def range_bearing(states, measurement, xp=np):
    observer, target = states
    relative = observer[:2, :2].T @ (target[:2, 2] - observer[:2, 2])
    distance = xp.linalg.norm(relative)
    bearing = xp.arctan2(relative[1], relative[0])

    # Lift the predicted angle near the observation so h - z is wrapped.
    angle_error = wrap_angle(bearing - measurement[1], xp=xp)
    return xp.array([distance, measurement[1] + angle_error])

# analytic jacobian
def range_bearing_jacobian(states, measurement, xp=np):
    observer, target = states
    R_i, R_j = observer[:2, :2], target[:2, :2]
    p = R_i.T @ (target[:2, 2] - observer[:2, 2])
    r = np.linalg.norm(p)

    # Chain rule: [range, bearing] -> relative position -> right increments.
    measurement_jac = np.array([[p[0] / r, p[1] / r],
                                [-p[1] / r ** 2, p[0] / r ** 2]])
    observer_jac = np.column_stack((-np.eye(2), np.array([p[1], -p[0]])))
    target_jac = np.column_stack((R_i.T @ R_j, np.zeros(2)))
    return measurement_jac @ np.hstack((observer_jac, target_jac))

# numerical jacobian
def range_bearing_numerical_jacobian(states, measurement, xp=np):
    # Central differences use the circular bearing difference, not a 2*pi jump.
    space = SE2()
    jacobian = np.zeros((2, 6))
    for i, delta in enumerate(np.eye(6) * 1e-6):
        plus = tuple(space.retract(state, d) for state, d in zip(states, delta.reshape(2, 3)))
        minus = tuple(space.retract(state, -d) for state, d in zip(states, delta.reshape(2, 3)))
        difference = range_bearing(plus, measurement) - range_bearing(minus, measurement)
        difference[1] = wrap_angle(difference[1])
        jacobian[:, i] = difference / 2e-6
    return jacobian




def build_graph(jacobian, jac_fn=None):
    factor_graph = FactorGraph()
    xp = jnp if jacobian in ('auto', 'jax') else np
    if jacobian == 'analytic' and jac_fn is None:
        jac_fn = range_bearing_jacobian
    if jacobian == 'numerical' and jac_fn is None:
        jac_fn = range_bearing_numerical_jacobian

    # Prior covariance is expressed in each pose's right tangent coordinates.
    x1 = factor_graph.add_variable(
        dofs=3,
        prior_mean=pose(1.0, 2.0, 0.0),
        prior_cov=np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]]),
        manifold=SE2(xp=xp),
    )
    x2 = factor_graph.add_variable(
        dofs=3,
        prior_mean=pose(0.5, 1.5, np.pi / 4),
        prior_cov=np.array([[0.5, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        manifold=SE2(xp=xp),
    )
    x3 = factor_graph.add_variable(
        dofs=3,
        prior_mean=pose(1.5, 2.5, np.pi / 2),
        prior_cov=np.array([[1.5, 0.7, 0.0], [0.7, 3.0, 0.0], [0.0, 0.0, 1.0]]),
        manifold=SE2(xp=xp),
    )

    # Directed measurements: observer -> target, [range, bearing in radians].
    meas_x1_x2 = np.array([2.0, np.pi / 4])
    meas_x2_x3 = np.array([1.0, np.pi / 2])
    meas_cov = 0.1 * np.eye(2)

    factor_graph.add_factor(
        [x1, x2], meas_fn=range_bearing,
        measurement=meas_x1_x2, noise_cov=meas_cov,
        jacobian=jacobian, jac_fn=jac_fn,
        args=(meas_x1_x2, xp),
    )
    factor_graph.add_factor(
        [x2, x3], meas_fn=range_bearing,
        measurement=meas_x2_x3, noise_cov=meas_cov,
        jacobian=jacobian, jac_fn=jac_fn,
        args=(meas_x2_x3, xp),
    )
    return factor_graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if __package__:
        from .utils.plotter import plot_history
    else:
        from utils.plotter import plot_history

    with jax.enable_x64():
        graph_ad = build_graph(jacobian='auto')
        converged_ad, n_iters_ad, history_ad = graph_ad.solve(history=True)

    graph_numerical = build_graph(jacobian='numerical')
    converged_numerical, n_iters_numerical, history_numerical = graph_numerical.solve(history=True)

    graph_analytic = build_graph(jacobian='analytic', jac_fn=range_bearing_jacobian)
    converged_analytic, n_iters_analytic, history_analytic = graph_analytic.solve(history=True)

    print('AD:', converged_ad, 'iters:', n_iters_ad)
    print('Numerical:', converged_numerical, 'iters:', n_iters_numerical)
    print('Analytic:', converged_analytic, 'iters:', n_iters_analytic)

    fig, animation = plot_history(history_analytic, 'SE(2) range-bearing')
    plt.show()
