from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.solvers import get_method
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.integrate_on_grid import _odeint_on_grid
from probjax.utils.odeutil.params import AdaptiveParams


STATIC_NAMES = (
    "drift",
    "method",
    "dtype",
    "filter_state",
    "check_points",
    "adaptive_params",
    "return_state",
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
    return_state: bool = False,
    filter_state: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
):
    """Solve an ordinary differential equation.

    Args:
        drift: The drift function
        y0: Initial state
        ts: Time points
        *args: Additional arguments for the drift function
        method: Integration method
        dtype: Data type for computation
        return_state: Whether to return solver state
        filter_state: Optional state filter function
        check_points: Optional check points for grid integration
        adaptive_params: Parameters for adaptive integration

    Returns:
        Solution trajectory or tuple of (state, trajectory)
    """
    if adaptive_params is None:
        adaptive_params = AdaptiveParams()

    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_util.tree_map(lambda x: x.astype(dtype), y0)

    y0 = jax.tree_util.tree_map(jnp.atleast_1d, y0)
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

        state, ys = _odeint_on_grid(
            method,
            drift,
            flat_y0,
            ts,
            *args,
            filter_output=filter_unravel,
            check_points=check_points,
        )
        if filter_state is None:
            ys = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys
            )
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
            "rtol": adaptive_params.rtol,
            "atol": adaptive_params.atol,
            "mxstep": adaptive_params.mxstep,
            "dtmin": adaptive_params.dtmin,
            "dtmax": adaptive_params.dtmax,
            "maxerror": adaptive_params.maxerror,
            "safety": adaptive_params.safety,
            "ifactor": adaptive_params.ifactor,
            "dfactor": adaptive_params.dfactor,
            "error_norm": adaptive_params.error_norm,
            "order": order,
            "interpolation_order": interpolation_order,
            "filter_output": filter_unravel,
            "return_state": return_state,
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
            ys = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys
            )

    if return_state:
        return state, ys
    else:
        return ys
