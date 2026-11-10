"""
    Jacobian dispatch for measurement functions.

    A factor's measurement function h(x, *args) maps the (stacked) adjacent variable
    values to a predicted measurement. GBP linearises h at the current linearisation
    point, which requires its Jacobian. This module lets the user pick how that
    Jacobian is obtained:

        'analytic'  -> a hand-written jac_fn(x, *args) supplied by the user.
        'numerical' -> finite differences (utils.derivatives.jac_fd). No extra deps.
        'auto'/'jax'-> automatic differentiation via JAX (imported lazily).

    All returned callables share the signature jac(x, *args) -> ndarray of shape
    (dim(measurement), dim(x)), matching what Factor.compute_factor expects.
"""

import numpy as np

from .differentiation import jac_fd


def _numerical_jac(meas_fn, delta=1e-6):
    def jac(inp, *args):
        return jac_fd(np.asarray(inp, dtype=float), meas_fn, *args, delta=delta)
    return jac


# meas_fn must b edefined by jax.numpy
def _jax_jac(meas_fn):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "jacobian='auto' requires a working JAX install (see cause above). "
            "Install/repair it, e.g. `pip install -U \"jax[cpu]\"`, or use "
            "jacobian='numerical'/'analytic' instead."
        ) from e

    def jac(inp, *args):
        x = jnp.asarray(inp, dtype=float)
        J = jax.jacobian(lambda z: jnp.asarray(meas_fn(z, *args)))(x)
        return np.asarray(J)

    return jac


def make_jac_fn(mode, meas_fn, jac_fn=None):
    """
        Resolve a Jacobian callable jac(x, *args) for the given mode.

        mode: one of 'analytic', 'numerical', 'auto'/'jax'.
        meas_fn: the measurement function h(x, *args).
        jac_fn: user-supplied analytic Jacobian, required when mode == 'analytic'.
    """
    if mode in ('analytic', None):
        if jac_fn is None:
            raise ValueError(
                "jacobian='analytic' requires a jac_fn. Pass jac_fn=... or choose "
                "jacobian='numerical'/'auto'."
            )
        return jac_fn
    if mode == 'numerical':
        return _numerical_jac(meas_fn)
    if mode in ('auto', 'jax'):
        return _jax_jac(meas_fn)
    raise ValueError(
        f"Unknown jacobian mode '{mode}'. Expected 'analytic', 'numerical' or 'auto'."
    )
