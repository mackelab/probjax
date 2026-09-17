from functools import partial
from typing import Callable, Optional, Sequence, cast

import jax
import jax.numpy as jnp

from probjax.utils._solver_common import (
    ensure_dtype,
    make_filter_wrapper,
    stack_trace,
)
from probjax.utils.functions import Drift, generic_drift
from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.adaptive import StepSizeAdaptor
from probjax.utils.odeutil.filters import TraceFilter
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.integrate_on_grid import _odeint_on_grid
from probjax.utils.odeutil.solvers import get_method
from probjax.utils.typing import Array, PyTree

STATIC_NAMES = (
    "method",
    "dtype",
    "filter_state",
    "check_points",
    "step_size_adaptor",
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
    step_size_adaptor: Optional[StepSizeAdaptor] = None,
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
        step_size_adaptor: Step-size controller for adaptive solver methods.
            Ignored by fixed-step methods. Defaults to ``StepSizeAdaptor()``.

    Returns:
        PyTree with either the stacked trajectory (when `collect_trace` is True)
        or the filtered terminal state (when `collect_trace` is False). If the
        filter returns ``None`` the result is ``None``.
    """
    if step_size_adaptor is None:
        step_size_adaptor = StepSizeAdaptor()

    ts, y0 = ensure_dtype(ts, y0, dtype)

    flat_y0, unravel = ravel_args(y0)
    ravel_arg = getattr(drift, "ravel_arg", None)
    if callable(ravel_arg):
        drift = cast(Callable, ravel_arg(unravel, index=1))
    else:
        drift = ravel_arg_fun(drift, unravel, 1)

    # Ensure drift flows as a registered pytree so ``odeint_adaptive``'s
    # custom_vjp can tree-flatten it. Plain closures (e.g. from
    # ``ravel_arg_fun`` on non-Drift pytrees, or from the base
    # ``Drift.ravel_arg`` implementation) get wrapped in ``generic_drift`` so
    # the callable rides as aux data with zero array leaves.
    if not isinstance(drift, Drift):
        drift = generic_drift(fn=drift)

    _apply_filter = make_filter_wrapper(filter_state)
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
        # Lock the adaptor's local-error order to the method's.
        step_size_adaptor = step_size_adaptor.with_order(order)
        kwargs = {
            "step_size_adaptor": step_size_adaptor,
            "interpolation_order": interpolation_order,
            "filter_output": trace_filter_fn,
            "collect_trace": trace_enabled,
        }
        state, ys = odeint_adaptive(solver, drift, kwargs, flat_y0, ts, *args)

    state_y0 = state.y0 if state is not None else flat_y0
    final_state = unravel(state_y0)
    final_filtered = _apply_filter(final_state)

    if trace_enabled and ys is not None:
        return stack_trace(init_filtered, ys)

    return final_filtered
