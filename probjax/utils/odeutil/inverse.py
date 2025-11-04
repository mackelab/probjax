"""Automated inversion support for ODE solvers."""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.core.transformation import inverse, inverse_and_logabsdet
from probjax.utils.odeutil.adaptive import AdaptiveParams
from probjax.utils.odeutil.core import _odeint


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
    xs = _odeint(
        drift,
        y0,
        ts[::-1],
        *args,
        method=method,
        adaptive_params=adaptive_params,
        **kwargs,
    )
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
        trace_val = jnp.asarray(jnp.trace(jac(t, x)), dtype=dx.dtype)
        dlogdet = jnp.broadcast_to(trace_val, logdet.shape)
        return dx, dlogdet

    y0 = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    logdet0 = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(jnp.atleast_1d(x)[-1]), ys
    )
    xs, logdets = _odeint(
        aug_drift,
        (y0, logdet0),
        ts[::-1],
        *args,
        method=method,
        adaptive_params=adaptive_params,
        **kwargs,
    )

    yT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    logdetsT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], logdets)

    return yT, logdetsT


# Register inverse transformations
odeint_inv = inverse(_inv_odeint, static_argnums=(0,))
odeint_inv_and_logdet = inverse_and_logabsdet(_inv_logdet_odeint, static_argnums=(0,))
