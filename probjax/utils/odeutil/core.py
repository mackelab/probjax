from functools import partial
from typing import Callable, Optional, Sequence, cast

import jax
import jax.numpy as jnp

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.adaptive import AdaptiveParams
from probjax.utils.odeutil.filters import TraceFilter
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.integrate_on_grid import _odeint_on_grid
from probjax.utils.odeutil.solvers import get_method
from probjax.utils.typing import Array, PyTree

STATIC_NAMES = (
    "drift",
    "method",
    "dtype",
    "filter_state",
    "check_points",
    "adaptive_params",
    "collect_trace",
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
    filter_state: Optional[TraceFilter] = None,
    collect_trace: bool = True,
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
        filter_state: Trace filter describing which quantity to store or return.
            If the filter returns ``None`` (e.g. :class:`TraceNothing`), the
            output is ``None`` regardless of `collect_trace`.
        collect_trace: Whether to record the filtered quantity for each time
            point (`True`) or return only the filtered terminal state (`False`).
        check_points: Optional check points for grid integration
        adaptive_params: Parameters for adaptive integration

    Returns:
        PyTree with either the stacked trajectory (when `collect_trace` is True)
        or the filtered terminal state (when `collect_trace` is False). If the
        filter returns ``None`` the result is ``None``.
    """
    if adaptive_params is None:
        adaptive_params = AdaptiveParams()

    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), y0)

    ts = jnp.atleast_1d(ts)

    flat_y0, unravel = ravel_args(y0)
    ravel_arg = getattr(drift, "ravel_arg", None)
    if callable(ravel_arg):
        drift = cast(Callable, ravel_arg(unravel, index=1))
    else:
        drift = ravel_arg_fun(drift, unravel, 1)

    def _apply_filter(state_tree: PyTree[Array]) -> Optional[PyTree[Array]]:
        if filter_state is None:
            return state_tree
        return filter_state(state_tree)

    init_filtered = _apply_filter(y0)
    trace_enabled = collect_trace and init_filtered is not None

    def raveled_filter(yi, info):
        del info
        return _apply_filter(unravel(yi))

    trace_filter_fn = raveled_filter if trace_enabled else None

    solver, info = get_method(method)

    is_adaptive = info["adaptive"]

    if not is_adaptive:
        state, ys = _odeint_on_grid(
            solver,
            drift,
            flat_y0,
            ts,
            *args,
            trace_filter=trace_filter_fn,
            check_points=check_points,
            collect_trace=trace_enabled,
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
            "filter_output": trace_filter_fn,
            "collect_trace": trace_enabled,
        }
        state, ys = odeint_adaptive(solver, drift, kwargs, flat_y0, ts, *args)

    state_y0 = getattr(state, "y0") if state is not None else flat_y0
    final_state = unravel(state_y0)
    final_filtered = _apply_filter(final_state)

    if trace_enabled and ys is not None:

        def _stack(init_leaf, trace_leaf):
            init_leaf = jnp.asarray(init_leaf)
            trace_leaf = jnp.asarray(trace_leaf)
            return jnp.concatenate([init_leaf[None], trace_leaf], axis=0)

        trace = jax.tree_util.tree_map(
            _stack,
            init_filtered,
            ys,
        )
        return trace

    return final_filtered
