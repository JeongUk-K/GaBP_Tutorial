"""Build a graph with robot-to-robot range-bearing observations, priors and odometry."""

from build_graph import build_graph as _build_graph


def build_graph(data, *, prior_cov, odometry_cov, measurement_cov):
    """Return (graph, variables), excluding landmark range-bearing factors."""
    return _build_graph(
        data, prior_cov=prior_cov, odometry_cov=odometry_cov,
        measurement_cov=measurement_cov, measurement_type='robot',
    )
