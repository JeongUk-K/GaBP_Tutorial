#!/usr/bin/env python
"""Packaging for the general-purpose Gaussian Belief Propagation solver."""

from setuptools import find_packages, setup

setup(
    name='gbp',
    version='0.1.0',
    description='General-purpose Gaussian Belief Propagation factor-graph solver',
    packages=find_packages(exclude=('examples', 'examples.*')),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
    ],
    extras_require={
        'jax': ['jax'],
    },
)
