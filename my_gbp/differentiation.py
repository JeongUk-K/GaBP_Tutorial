import numpy as np

"""
    Generic numerical differentiation utilities for the GBP core.

    These are domain independent: given a measurement function h(x, *args) they
    approximate its Jacobian by finite differences, or check a hand-written
    Jacobian against the finite-difference one.
"""


# Finite difference Jacobian computation
def jac_fd(inp, meas_fn, *args, delta=1e-8):
    """
        Compute the Jacobian of meas_fn at inp using the finite difference method.
    """
    z = meas_fn(inp, *args)
    if isinstance(z, float):
        jac = np.zeros([1, len(inp)])
    else:
        jac = np.zeros([len(z), len(inp)])
    for i in range(len(inp)):
        d_inp = np.copy(inp)
        d_inp[i] += delta
        jac[:, i] = (meas_fn(d_inp, *args) - meas_fn(inp, *args)) / delta
    return jac


# Jacobian check utility
def check_jac(jac_fn, inp, meas_fn, *args, threshold=1e-3):
    jac = jac_fn(inp, *args)
    jacfd = jac_fd(inp, meas_fn, *args)

    if np.max(jac - jacfd) < threshold:
        print(f"Passed! Jacobian correct to within {threshold}")
    else:
        print(f"Failed: Jacobian difference to finite difference Jacobian not within threshold ({threshold})"
              f"\nMaximum discrepancy between Jacobian and finite diff Jacobian: {np.max(jac - jacfd)}")
