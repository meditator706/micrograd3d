"""
MicroGrad3D - A lightweight autograd engine with 3D visualization
Supports various optimization algorithms and second-order derivatives
"""

from .engine import Value
from .optimizer import GradientDescent, Optimizer
from .viz import Visualizer3D

__all__ = [
    'Value',
    'GradientDescent',
    'Optimizer',
    'bowl',
    'bowl_numpy',
    'rosenbrock',
    'rosenbrock_numpy',
    'Visualizer3D',
]
