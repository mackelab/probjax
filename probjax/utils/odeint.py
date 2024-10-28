from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil import get_method
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.integrate_on_grid import _odeint_on_grid

STATIC_NAMES = (
    "drift",
    "method",
    "dtype",
    "filter_state",
    "check_points",
    "rtol",
    "atol",
    "mxstep",
    "dtmin",
    "dtmax",
    "maxerror",
    "safety",
    "ifactor",
    "dfactor",
    "error_norm",
)


@partial(
    jax.jit,
    static_argnames=STATIC_NAMES,
)
def _odeint(
    drift,
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "rk4",
    dtype=jnp.float32,
    filter_state: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    dtmin: float = 0.0,
    dtmax: float = jnp.inf,
    maxerror: float = 1.2,
    safety: float = 0.95,
    ifactor: float = 10.0,
    dfactor: float = 0.1,
    error_norm: float = 2,
):
    """Solve an ordinary differential equation."""
    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_map(lambda x: x.astype(dtype), y0)

    y0 = jax.tree_map(jnp.atleast_1d, y0)
    ts = jnp.atleast_1d(ts)

    flat_y0, unravel = ravel_args(y0)
    drift = ravel_arg_fun(drift, unravel, 1)

    method, info = get_method(method)

    is_adaptive = info["adaptive"]

    if not is_adaptive:

        def filter_unravel(state, _):
            if filter_state is None:
                return unravel(state.y0)
            else:
                return filter_state(unravel(state.y0))

        _, ys = _odeint_on_grid(
            method,
            drift,
            flat_y0,
            ts,
            *args,
            filter_output=filter_unravel,
            check_points=check_points,
        )
        ys = jax.tree_map(lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys)
    else:
        if filter_state is not None:

            def filter_unravel(y):
                if filter_state is None:
                    return unravel(y)
                else:
                    return filter_state(unravel(y))

        else:
            filter_unravel = None

        order = info["order"]
        interpolation_order = info.get("interpolation_order", 3)
        params = {
            "rtol": rtol,
            "atol": atol,
            "mxstep": mxstep,
            "dtmin": dtmin,
            "dtmax": dtmax,
            "maxerror": maxerror,
            "safety": safety,
            "ifactor": ifactor,
            "dfactor": dfactor,
            "error_norm": error_norm,
            "order": order,
            "interpolation_order": interpolation_order,
            "filter_output": filter_unravel,
        }
        ys = odeint_adaptive(
            method,
            drift,
            params,
            flat_y0,
            ts,
            *args,
        )
        if filter_state is None:
            ys = jax.vmap(unravel)(ys)
        ys = jax.tree_map(lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys)

    return ys


# Register inverse


# Inverse odeint
def _inv_odeint(drift, ys: Array, ts: Array, *args, **kwargs):
    y0 = jax.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    xs = _odeint(drift, y0, ts[::-1], *args, **kwargs)
    yT = jax.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    return yT


# Ode and logabsdet
def _inv_logdet_odeint(drift, ys, ts, *args, **kwargs):
    _jac = jax.jacfwd(drift, argnums=1)
    jac = lambda t, x: jnp.atleast_2d(_jac(t, x))

    def aug_drift(t, state, *args):
        x, logdet = state
        dx = jnp.atleast_1d(drift(t, x, *args))
        dlogdet = jnp.atleast_1d(jnp.trace(jac(t, x)))
        return dx, dlogdet

    y0 = jax.tree_map(lambda x: jnp.atleast_1d(x)[-1], ys)
    logdet0 = jax.tree_map(lambda x: jnp.zeros_like(jnp.atleast_1d(x)[-1]), ys)
    xs, logdets = _odeint(aug_drift, (y0, logdet0), ts[::-1], *args, **kwargs)

    yT = jax.tree_map(lambda x: jnp.atleast_1d(x)[-1], xs)
    logdetsT = jax.tree_map(lambda x: jnp.atleast_1d(x)[-1], logdets)

    return yT, logdetsT


# ODEs are invertible, so we can define the inverse of the ODE solver
odeint = _odeint
# odeint = custom_inverse(_odeint, static_argnums=(0,), inv_argnum=1)
# odeint.definv(_inv_odeint)
# odeint.definv_and_logdet(_inv_logdet_odeint)
