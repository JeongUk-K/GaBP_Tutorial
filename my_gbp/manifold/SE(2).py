import numpy as np


class SE2:
    """3x3 poses with right increments [rho_x, rho_y, theta] in radians."""

    tangent_dim = 3

    def __init__(self, *, xp=np):
        self._xp = xp

    def identity(self):
        return self._xp.eye(3)

    def compose(self, first, second):
        return self._xp.asarray(first, dtype=float) @ self._xp.asarray(second, dtype=float)

    def inverse(self, state):
        # [R, t]^-1 = [R.T, -R.T @ t].
        state = self._xp.asarray(state, dtype=float)
        R = state[:2, :2]
        t = state[:2, 2]
        inverse_t = -R.T @ t
        return self._xp.vstack((self._xp.column_stack((R.T, inverse_t)), self._xp.array([0., 0., 1.])))

    def exp(self, delta):
        # Tangent vector -> pose: rotation R(theta), translation V(theta) @ rho.
        delta = self._xp.asarray(delta, dtype=float).reshape(3)
        rho, theta = delta[:2], delta[2]
        cosine, sine = self._xp.cos(theta), self._xp.sin(theta)
        R = self._xp.array([[cosine, -sine], [sine, cosine]])
        t = self._V(theta) @ rho
        return self._xp.vstack((self._xp.column_stack((R, t)), self._xp.array([0., 0., 1.])))

    def log(self, state):
        # Pose -> tangent vector; atan2 selects the principal angle.
        state = self._xp.asarray(state, dtype=float)
        R, t = state[:2, :2], state[:2, 2]
        theta = self._xp.arctan2(R[1, 0], R[0, 0])
        rho = self._xp.linalg.solve(self._V(theta), t)
        return self._xp.array([rho[0], rho[1], theta])

    def retract(self, state, delta):
        # X_new = X @ Exp(delta).
        return self.compose(state, self.exp(delta))

    def local(self, reference, state):
        # delta = Log(X_reference^-1 @ X).
        return self.log(self.compose(self.inverse(reference), state))

    def hat(self, delta):
        # Tangent vector -> Lie algebra matrix.
        rho_x, rho_y, theta = self._xp.asarray(delta, dtype=float).reshape(3)
        return self._xp.array([[0., -theta, rho_x], [theta, 0., rho_y], [0., 0., 0.]])

    def vee(self, tangent):
        # Lie algebra matrix -> tangent vector.
        tangent = self._xp.asarray(tangent, dtype=float)
        return self._xp.array([tangent[0, 2], tangent[1, 2], tangent[1, 0]])

    def adjoint(self, state):
        state = self._xp.asarray(state, dtype=float)
        R, t = state[:2, :2], state[:2, 2]
        cross_t = self._xp.array([t[1], -t[0]])
        return self._xp.vstack((self._xp.column_stack((R, cross_t)), self._xp.array([0., 0., 1.])))

    def right_jacobian(self, delta):
        # Jr(delta) = [[V(-theta), q], [0, 1]].
        delta = self._xp.asarray(delta, dtype=float).reshape(3)
        rho, theta = delta[:2], delta[2]
        V = self._V(-theta)
        q = self._right_translation(rho, theta)
        return self._xp.vstack((self._xp.column_stack((V, q)), self._xp.array([0., 0., 1.])))

    def right_jacobian_inverse(self, delta):
        return self._xp.linalg.solve(self.right_jacobian(delta), self._xp.eye(3))

    def retract_jacobians(self, state, delta):
        # Derivatives with respect to state and delta, in the output's right tangent.
        delta = self._xp.asarray(delta, dtype=float).reshape(3)
        return self.adjoint(self.exp(-delta)), self.right_jacobian(delta)

    def local_jacobians(self, reference, state):
        # Right perturbations of reference and state give (-Jl^-1, Jr^-1).
        delta = self.local(reference, state)
        return -self.right_jacobian_inverse(-delta), self.right_jacobian_inverse(delta)

    # Stable scalar formulas used by the pose operations and Jacobians.
    def _V(self, theta):
        xp = self._xp
        sin_ratio = xp.sinc(theta / xp.pi)  # sin(theta) / theta.
        cos_ratio = 0.5 * theta * xp.sinc(theta / (2 * xp.pi)) ** 2  # (1 - cos(theta)) / theta.
        return xp.array([[sin_ratio, -cos_ratio], [cos_ratio, sin_ratio]])

    def _right_translation(self, rho, theta):
        xp = self._xp
        theta2 = theta * theta
        small = xp.abs(theta) < 0.1
        safe_theta = xp.where(small, 1., theta)
        series = theta * (1 / 6 - theta2 / 120 + theta2 ** 2 / 5040 - theta2 ** 3 / 362880)
        c = xp.where(small, series, (theta - xp.sin(theta)) / safe_theta ** 2)
        d = 0.5 * xp.sinc(theta / (2 * xp.pi)) ** 2
        # q = ((theta - sin(theta)) / theta^2) rho + ((1 - cos(theta)) / theta^2) Omega rho.
        perpendicular_rho = xp.array([-rho[1], rho[0]])
        return c * rho + d * perpendicular_rho
