"""
    General-purpose Gaussian Belief Propagation solver.

    The core is domain independent: build a graph with add_variable / add_factor,
    supply your own measurement functions, and run solve(). It was extracted and
    refactored from a SLAM / Bundle Adjustment implementation into a reusable solver.
"""

from . import gaussian
from . import differentiation
from . import autodiff
from . import gbp

from .gaussian import NdimGaussian
from .gbp import FactorGraph, VariableNode, Factor

__all__ = [
    'FactorGraph', 'VariableNode', 'Factor', 'NdimGaussian',
    'gbp', 'autodiff', 'differentiation', 'gaussian',
]
