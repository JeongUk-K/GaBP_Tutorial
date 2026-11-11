import numpy as np

from my_gbp import FactorGraph
from my_gbp.manifold import SE2


def nearest_time_index(node_times, measurement_time):
    # Equal distances select the first node; node_times must be increasing.
    return int(np.argmin(np.abs(np.asarray(node_times) - measurement_time)))


def build_graph(data, *, prior_cov, odometry_cov, measurement_cov,
                measurement_type='all'):
    """Build the graph using a supplied covariance for each relative pose factor.

    prior_cov and odometry_cov are 3x3; measurement_cov is 2x2.
    odometry_cov is the final Q for Log(Z^-1 @ X_i^-1 @ X_j), in the
    predicted pose's right tangent coordinates [rho_x(m), rho_y(m), theta(rad)].
    It is used directly without dt scaling or a world-frame rotation.
    Only robot poses are variables; landmark positions are fixed measurement inputs.
    measurement_type selects range-bearing factors: 'all', 'landmark' or 'robot'.
    Priors and odometry factors are included in every case.
    """
    if measurement_type not in ('all', 'landmark', 'robot'):
        raise ValueError("measurement_type must be 'all', 'landmark' or 'robot'.")
    odometry_cov = np.asarray(odometry_cov, dtype=float)
    if odometry_cov.shape != (3, 3):
        raise ValueError('Set odometry_cov to the final 3x3 pose-error covariance Q in main.')
    if not np.all(np.isfinite(odometry_cov)) or not np.allclose(odometry_cov, odometry_cov.T):
        raise ValueError('odometry_cov must be finite and symmetric.')
    np.linalg.cholesky(odometry_cov)
    barcode_to_subject = {int(code): int(subject) for subject, code in data['barcodes']}
    subjects = set(data['robots']) | {int(row[0]) for row in data['landmarks']}
    for robot in data['robots'].values():
        for time, barcode, _, _ in robot['measurements']:
            if int(barcode) not in barcode_to_subject:
                raise ValueError(f'Unknown measurement barcode {int(barcode)} at {time}.')
            if barcode_to_subject[int(barcode)] not in subjects:
                raise ValueError(f'No robot or landmark for barcode {int(barcode)}.')

    node_times = data['node_times']
    motions = {
        robot_id: [
            integrate_odometry(robot['odometry'], start, end)
            for start, end in zip(node_times[:-1], node_times[1:])
        ]
        for robot_id, robot in data['robots'].items()
    }
    graph = FactorGraph()
    variables = add_variables(graph, data, motions, prior_cov)
    add_odometry_factors(graph, variables, motions, odometry_cov)
    add_measurement_factors(graph, data, variables, measurement_cov,
                            measurement_type=measurement_type)
    return graph, variables


def add_variables(graph, data, motions, prior_cov):
    space = SE2()
    variables = {'robots': {}}
    for robot_id, robot in data['robots'].items():
        state = initial_pose(robot['groundtruth'], data['node_times'][0])
        nodes = []
        for index in range(len(data['node_times'])):
            if index:
                state = state @ motions[robot_id][index - 1]
            node = graph.add_variable(
                dofs=3, prior_mean=state,
                prior_cov=prior_cov if index == 0 else None,
                manifold=space,
            )
            if index:
                # Later poses have an initial estimate, but no absolute prior.
                node.prior.lam.fill(0.)
            nodes.append(node)
        variables['robots'][robot_id] = nodes

    return variables


def add_odometry_factors(graph, variables, motions, odometry_cov):
    for robot_id, nodes in variables['robots'].items():
        for first, second, motion in zip(nodes[:-1], nodes[1:], motions[robot_id]):
            factor = graph.add_factor(
                [first, second], meas_fn=odometry_residual,
                measurement=np.zeros(3), noise_cov=odometry_cov,
                jacobian='analytic', jac_fn=odometry_jacobian, args=(motion,),
            )
            # Propagate the first-pose prior before solve initializes all beliefs.
            factor.compute_messages(eta_damping=0.)


def add_measurement_factors(graph, data, variables, measurement_cov, *,
                            measurement_type='all'):
    barcode_to_subject = {int(code): int(subject) for subject, code in data['barcodes']}
    landmarks = {int(row[0]): row[1:3].copy() for row in data['landmarks']}
    for robot_id, robot in data['robots'].items():
        for time, barcode, distance, bearing in robot['measurements']:
            subject = barcode_to_subject[int(barcode)]
            is_robot = subject in variables['robots']
            if measurement_type == 'landmark' and is_robot:
                continue
            if measurement_type == 'robot' and not is_robot:
                continue
            index = nearest_time_index(data['node_times'], time)
            observer = variables['robots'][robot_id][index]
            adjacent = [observer]
            landmark = None
            if is_robot:
                adjacent.append(variables['robots'][subject][index])
            else:
                landmark = landmarks[subject]
            measurement = np.array([distance, bearing])
            graph.add_factor(
                adjacent, meas_fn=range_bearing,
                measurement=measurement, noise_cov=measurement_cov,
                jacobian='analytic', jac_fn=range_bearing_jacobian,
                args=(measurement, landmark),
            )


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def initial_pose(groundtruth, time):
    # Interpolate only the first pose; use the shortest angular difference.
    index = max(0, int(np.searchsorted(groundtruth[:, 0], time, side='right')) - 1)
    first, second = groundtruth[index:index + 2]
    alpha = (time - first[0]) / (second[0] - first[0])
    x, y = first[1:3] + alpha * (second[1:3] - first[1:3])
    theta = first[3] + alpha * wrap_angle(second[3] - first[3])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def integrate_odometry(odometry, start, end):
    # Each velocity sample is held until the next sample.
    times = odometry[:, 0]
    boundaries = np.r_[start, times[(times > start) & (times < end)], end]
    indices = np.searchsorted(times, boundaries[:-1], side='right') - 1
    space = SE2()
    motion = space.identity()
    for index, dt in zip(indices, np.diff(boundaries)):
        velocity, angular_velocity = odometry[index, 1:3]
        motion = motion @ space.exp(np.array([velocity, 0., angular_velocity]) * dt)
    return motion


def odometry_residual(states, motion):
    # Right-tangent pose error; use Q in the same coordinates.
    space = SE2()
    first, second = states
    relative = space.inverse(first) @ second
    return space.log(space.inverse(motion) @ relative)


def odometry_jacobian(states, motion):
    space = SE2()
    error = odometry_residual(states, motion)
    first_jac = -space.right_jacobian_inverse(-error) @ space.adjoint(space.inverse(motion))
    second_jac = space.right_jacobian_inverse(error)
    return np.hstack((first_jac, second_jac))


def range_bearing(states, measurement, landmark=None):
    observer = states[0]
    position = states[1][:2, 2] if landmark is None else landmark
    relative = observer[:2, :2].T @ (position - observer[:2, 2])
    distance = np.linalg.norm(relative)
    bearing = np.arctan2(relative[1], relative[0])
    angle_error = wrap_angle(bearing - measurement[1])
    return np.array([distance, measurement[1] + angle_error])


def range_bearing_jacobian(states, measurement, landmark=None):
    observer = states[0]
    rotation = observer[:2, :2]
    position = states[1][:2, 2] if landmark is None else landmark
    p = rotation.T @ (position - observer[:2, 2])
    r = np.linalg.norm(p)
    measurement_jac = np.array([
        [p[0] / r, p[1] / r],
        [-p[1] / r ** 2, p[0] / r ** 2],
    ])
    observer_jac = np.column_stack((-np.eye(2), np.array([p[1], -p[0]])))
    if landmark is not None:
        return measurement_jac @ observer_jac
    target_jac = np.column_stack((rotation.T @ states[1][:2, :2], np.zeros(2)))
    return measurement_jac @ np.hstack((observer_jac, target_jac))
