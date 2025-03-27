"""Automated inversion support for ODE solvers."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.core.transformation import inverse, inverse_and_logabsdet
from probjax.utils.odeutil import odeint_adaptive, _odeint_on_grid
from probjax.utils.odeutil.adaptive import AdaptiveParams
from probjax.utils.odeutil.solvers.base import ODESolver


def _inv_odeint(
    drift: Callable,
    ys: ArrayLike,
    ts: ArrayLike,
    *args,
    method: str = "rk4",
    adaptive_params: Optional[AdaptiveParams] = None,
    **kwargs,
) -> Array:
    """Inverse of ODE solver that maps final state to initial state.

    Args:
        drift: Function defining the ODE system
        ys: Final state values
        ts: Time points
        *args: Additional arguments for drift
        method: Integration method to use
        adaptive_params: Parameters for adaptive integration
        **kwargs: Additional keyword arguments for solver

    Returns:
        Initial state values
    """
    if adaptive_params is None:
        adaptive_params = AdaptiveParams()

    y0 = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    xs = odeint_adaptive(method, drift, adaptive_params, y0, ts[::-1], *args, **kwargs)
    yT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    return yT


def _inv_logdet_odeint(
    drift: Callable,
    ys: ArrayLike,
    ts: ArrayLike,
    *args,
    method: str = "rk4",
    adaptive_params: Optional[AdaptiveParams] = None,
    **kwargs,
) -> Tuple[Array, Array]:
    """Inverse of ODE solver with log determinant computation.

    Args:
        drift: Function defining the ODE system
        ys: Final state values
        ts: Time points
        *args: Additional arguments for drift
        method: Integration method to use
        adaptive_params: Parameters for adaptive integration
        **kwargs: Additional keyword arguments for solver

    Returns:
        Tuple of (initial state values, log determinant)
    """
    if adaptive_params is None:
        adaptive_params = AdaptiveParams()

    _jac = jax.jacfwd(drift, argnums=1)
    jac = lambda t, x: jnp.atleast_2d(_jac(t, x))

    def aug_drift(t, state, *args):
        x, logdet = state
        dx = jnp.atleast_1d(drift(t, x, *args))
        dlogdet = jnp.atleast_1d(jnp.trace(jac(t, x)))
        return dx, dlogdet

    y0 = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    logdet0 = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(jnp.atleast_1d(x)[-1]), ys
    )
    xs, logdets = odeint_adaptive(
        method, aug_drift, adaptive_params, (y0, logdet0), ts[::-1], *args, **kwargs
    )

    yT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    logdetsT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], logdets)

    return yT, logdetsT


# Register inverse transformations
odeint_inv = inverse(_inv_odeint, static_argnums=(0, 2))
odeint_inv_and_logdet = inverse_and_logabsdet(_inv_logdet_odeint, static_argnums=(0, 2))
