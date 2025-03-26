"""Ordinary differential equation integration.

This module provides a high-level interface for solving ordinary differential equations
using various numerical methods. It supports both fixed-step and adaptive-step integration,
with a focus on performance through JAX transformations.
"""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.odeutil import _odeint, AdaptiveParams


def odeint(
    drift: Callable,
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "rk4",
    dtype=jnp.float32,
    return_state: bool = False,
    filter_state: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
):
    """Solve an ordinary differential equation.

    This is a high-level interface for solving ODEs using various numerical methods.
    It supports both fixed-step and adaptive-step integration, with a focus on
    performance through JAX transformations.

    Args:
        drift: The drift function f(t, y, *args) that defines the ODE dy/dt = f(t, y, *args)
        y0: Initial state
        ts: Time points at which to evaluate the solution
        *args: Additional arguments for the drift function
        method: Integration method to use. Available methods include:
            - "euler": Forward Euler method
            - "rk4": 4th order Runge-Kutta method
            - "dopri5": Dormand-Prince 5th order method (adaptive)
            - "tsit5": Tsitouras 5th order method (adaptive)
            - "dopri8": Dormand-Prince 8th order method (adaptive)
            - "tsit8": Tsitouras 8th order method (adaptive)
            - "bogacki_shampine": Bogacki-Shampine 3rd order method (adaptive)
        dtype: Data type for computation
        return_state: Whether to return solver state
        filter_state: Optional state filter function
        check_points: Optional check points for grid integration
        adaptive_params: Parameters for adaptive integration

    Returns:
        Solution trajectory or tuple of (state, trajectory)

    Example:
        >>> def drift(t, y):
        ...     return -y
        >>> y0 = jnp.array([1.0])
        >>> ts = jnp.linspace(0, 1, 100)
        >>> ys = odeint(drift, y0, ts)
    """
    return _odeint(
        drift,
        y0,
        ts,
        *args,
        method=method,
        dtype=dtype,
        return_state=return_state,
        filter_state=filter_state,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )
