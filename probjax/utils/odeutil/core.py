from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree
import numpy as np

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.solvers import get_method, ODESolver
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.integrate_on_grid import _odeint_on_grid
from probjax.utils.odeutil.adaptive import AdaptiveParams


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
    method: str | ODESolver = "rk4",
    dtype=jnp.float32,
    return_state: bool = False,
    filter_state: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
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
        filter_state: Optional state filter function that operates on unraveled state
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

    if filter_state is not None:
        with jax.ensure_compile_time_eval():
            flat_y0_indices = np.arange(len(flat_y0), dtype=np.int32)
            y0_indices = unravel(flat_y0_indices)
            filtered_indices = filter_state(y0_indices)
            if filtered_indices is None:
                flat_filtered_indices = None
            else:
                flat_filtered_indices, _ = ravel_args(filtered_indices)

            def raveled_filter(yi, info):
                del info
                if flat_filtered_indices is not None:
                    return yi[flat_filtered_indices]
                else:
                    return None

    else:
        raveled_filter = None

    method, info = get_method(method)

    is_adaptive = info["adaptive"]

    # Precompute filter indices if filter_state is provided

    if not is_adaptive:
        state, ys = _odeint_on_grid(
            method,
            drift,
            flat_y0,
            ts,
            *args,
            filter_output=raveled_filter,
            check_points=check_points,
        )
    else:
        order = info["order"]
        interpolation_order = info.get("interpolation_order", 3)
        # Create new AdaptiveParams with the method's order
        adaptive_params = adaptive_params._replace(order=order)
        # Pass AdaptiveParams through kwargs
        kwargs = {
            "adaptive_params": adaptive_params,
            "interpolation_order": interpolation_order,
            "filter_output": raveled_filter,
            "return_state": return_state,
        }
        ys = odeint_adaptive(method, drift, kwargs, flat_y0, ts, *args)

    # Unravel and concat y0.
    if filter_state is None:
        ys = jax.vmap(unravel)(ys)
        ys = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys
        )
    else:
        y0_filtered = filter_state(y0)
        # Unravel the filtered state
        if y0_filtered is not None:
            _, unravel_filtered = ravel_args(y0_filtered)
            ys = jax.tree_util.tree_map(jnp.atleast_1d, ys)
            ys = jax.vmap(unravel_filtered)(ys)
            ys = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0_filtered, ys
            )

    if return_state:
        return state, ys
    else:
        return ys
