"""Gauss6: utilities for high-order time integration using JAX."""

from .gauss6 import GaussLegendre6Solver
from .central_differences import (
    dx_order_2,
    dx_order_4,
    dx_order_6,
    dxx_order_2,
    dxx_order_4,
    dxx_order_6,
    dxxx_order_2,
    dxxx_order_4,
    dxxx_order_6,
)

__all__ = [
    "GaussLegendre6Solver",
    "dx_order_2",
    "dx_order_4",
    "dx_order_6",
    "dxx_order_2",
    "dxx_order_4",
    "dxx_order_6",
    "dxxx_order_2",
    "dxxx_order_4",
    "dxxx_order_6",
]

__version__ = "0.1.0"
