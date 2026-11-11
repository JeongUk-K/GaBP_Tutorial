"""Manifold geometry implementations."""

from importlib import import_module

from .Euclidean import Euclidean

# Keep the requested filename while exposing a regular class import.
SE2 = import_module('.SE(2)', __name__).SE2

__all__ = ['Euclidean', 'SE2']
