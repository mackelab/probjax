import jax
import jax.numpy as jnp
from jax import Array

from probjax.utils.odeutil.core import _odeint


def _inv_odeint(drift, ys: Array, ts: Array, *args, **kwargs):
    """Inverse ODE integration.

    Args:
        drift: The drift function
        ys: The final state
        ts: Time points
        *args: Additional arguments for the drift function
        **kwargs: Additional keyword arguments for _odeint

    Returns:
        The initial state
    """
    y0 = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    xs = _odeint(drift, y0, ts[::-1], *args, **kwargs)
    yT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    return yT


def _inv_logdet_odeint(drift, ys, ts, *args, **kwargs):
    """Inverse ODE integration with log determinant computation.

    Args:
        drift: The drift function
        ys: The final state
        ts: Time points
        *args: Additional arguments for the drift function
        **kwargs: Additional keyword arguments for _odeint

    Returns:
        Tuple of (initial state, log determinant)
    """
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
    xs, logdets = _odeint(aug_drift, (y0, logdet0), ts[::-1], *args, **kwargs)

    yT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    logdetsT = jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x)[-1], logdets)

    return yT, logdetsT
