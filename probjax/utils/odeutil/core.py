from functools import partial
from typing import Any, Callable, Optional, Sequence, cast

import jax
import jax.numpy as jnp

from probjax.utils._solver_common import (
    ensure_dtype,
    make_filter_wrapper,
    stack_trace,
)
from probjax.utils.functions import Drift, RaveledDrift, generic_drift
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
) -> tuple[PyTree[Array], Any]:
    """Solve an ordinary differential equation.

    Args:
        drift: The drift function
        y0: Initial state
        ts: Time points
        *args: Additional arguments for the drift function
        method: Integration method
        dtype: Data type for computation
        filter_state: Trace filter describing which quantity to store or return.
        collect_trace: Whether to record the filtered quantity for each time
            point (`True`) or return only the filtered terminal state (`False`).
        check_points: Optional check points for grid integration
        step_size_adaptor: Step-size controller for adaptive solver methods.
            Ignored by fixed-step methods. Defaults to ``StepSizeAdaptor()``.

    Returns:
        ``(value, solver_state)``. ``value`` is the stacked trajectory when
        `collect_trace` is true and the filtered terminal state otherwise;
        ``solver_state`` is the solver's final state.
    """
    ts, y0 = ensure_dtype(ts, y0, dtype)

    flat_y0, unravel = ravel_args(y0)
    ravel_arg = getattr(drift, "ravel_arg", None)
    if callable(ravel_arg):
        drift = cast(Callable, ravel_arg(unravel, index=1))
    elif isinstance(drift, Drift):
        drift = RaveledDrift(drift, unravel, index=1)
    else:
        # Plain callable or non-Drift pytree (e.g. a @register_drift dataclass
        # not subclassing Drift). If it carries array leaves, keep them as
        # explicit pytree children via RaveledDrift so they stay
        # differentiable: custom_vjp cannot differentiate closed-over values.
        leaves, _ = jax.tree_util.tree_flatten(drift)
        if leaves:
            drift = RaveledDrift(drift, unravel, index=1)
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
    trace_enabled = collect_trace

    def raveled_filter(yi, info):
        del info
        return _apply_filter(unravel(yi))

    trace_filter_fn = raveled_filter if trace_enabled else None

    solver, info = get_method(method)

    is_adaptive = info.get("adaptive", False)
    if is_adaptive:
        if step_size_adaptor is None:
            step_size_adaptor = StepSizeAdaptor()
    elif step_size_adaptor is not None:
        raise ValueError(
            f"step_size_adaptor is only supported by adaptive solvers, but "
            f"solver={method!r} is fixed-step."
        )

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
        adaptor = cast(StepSizeAdaptor, step_size_adaptor).with_order(order)
        kwargs = {
            "step_size_adaptor": adaptor,
            "interpolation_order": interpolation_order,
            "filter_output": trace_filter_fn,
            "collect_trace": trace_enabled,
        }
        state, ys = odeint_adaptive(solver, drift, kwargs, flat_y0, ts, *args)

    state_y0 = getattr(state, "y0", None) if state is not None else None
    if state_y0 is None:
        state_y0 = flat_y0
    final_state = unravel(state_y0)
    final_filtered = _apply_filter(final_state)

    if trace_enabled and ys is not None:
        return stack_trace(init_filtered, ys), state

    return final_filtered, state
