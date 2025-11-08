from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
from probjax.utils.odeutil.filters import TraceFilter
from probjax.utils.sdeutil.base import get_method
from probjax.utils.sdeutil.integrate_on_grid import _sdeint_on_grid

STATIC_NAMES = [
    "drift",
    "diffusion",
    "method",
    "dtype",
    "sde_type",
    "noise_type",
    "return_brownian",
    "return_state",
    "filter_state",
    "collect_trace",
    "check_points",
]


@partial(jax.jit, static_argnames=STATIC_NAMES)
def _sdeint(
    rng: Key,
    drift: Callable,
    diffusion: Callable,
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    return_state: bool = False,
    noise_type: Optional[str] = None,
    filter_state: Optional[TraceFilter] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
) -> Union[
    PyTree[Array], Tuple[Any, PyTree[Array]], Tuple[Any, PyTree[Array], PyTree[Array]]
]:
    """Core implementation of SDE integration.

    This is the internal implementation that handles the actual SDE integration.
    It supports various integration methods and can return both the solution
    and optional Brownian motion paths.

    Args:
        rng: Random number generator key
        drift: Drift function f(t, y, *args)
        diffusion: Diffusion function g(t, y, *args)
        y0: Initial state
        ts: Time points
        *args: Additional arguments passed to drift and diffusion
        method: Integration method
        dtype: Data type for computation
        sde_type: Type of SDE ("ito" or "stratonovich")
        return_brownian: Whether to return Brownian motion paths (requires `collect_trace=True`)
        return_state: Whether to return solver state
        noise_type: Type of noise ("diagonal" or "general")
        filter_state: Optional filter applied to the state (and Brownian paths when requested)
        collect_trace: Whether to record the filtered quantity at every time point (`True`)
            or return only the filtered terminal state (`False`). Must be `True` if
            `return_brownian` is requested.
        check_points: Optional sequence of indices for grid integration

    Returns:
        Depending on `return_brownian` and `collect_trace`:
        - If `return_brownian=False`: filtered trajectory when `collect_trace=True`
          or filtered terminal state when `collect_trace=False`.
        - If `return_brownian=True`: tuple of (state trace, Brownian trace), both stacked
          over all time points.
        If `return_state=True`, the solver state is returned as the leading element of
        the tuple.
    """
    if not collect_trace and return_brownian:
        raise ValueError("collect_trace must be True when returning Brownian paths.")

    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_util.tree_map(lambda x: x.astype(dtype), y0)

    y0 = jax.tree_util.tree_map(jnp.atleast_1d, y0)
    ts = jnp.atleast_1d(ts)

    flat_y0, unravel = ravel_args(y0)
    drift = ravel_arg_fun(drift, unravel, 1)
    diffusion = ravel_arg_fun(diffusion, unravel, 1)

    def apply_filter(tree: PyTree[Array]) -> Optional[PyTree[Array]]:
        if filter_state is None:
            return tree
        return filter_state(tree)

    init_filtered = apply_filter(y0)
    trace_enabled = collect_trace and (init_filtered is not None)

    if return_brownian and not trace_enabled:
        raise ValueError(
            "filter_state must return a value when collect_trace=True and "
            "return_brownian=True."
        )

    method, _ = get_method(method)

    if noise_type is None:
        g0 = jnp.asarray(diffusion(ts[0], flat_y0))
        noise_type = "diagonal" if g0.ndim <= 1 else "general"

    method = partial(method, sde_type=sde_type, noise_type=noise_type)

    if trace_enabled:

        def trace_filter(state, info):
            filtered_state = apply_filter(unravel(state.y0))
            if return_brownian:
                filtered_bm = apply_filter(unravel(info.dWt))
                return filtered_state, filtered_bm
            return filtered_state

        trace_fn = trace_filter
    else:
        trace_fn = None

    state, traced = _sdeint_on_grid(
        method,
        drift,
        diffusion,
        rng,
        flat_y0,
        ts,
        *args,
        filter_output=trace_fn,
        check_points=check_points,
        collect_trace=trace_enabled,
    )

    final_state = apply_filter(unravel(state.y0))

    trace_output: Optional[PyTree[Array]]
    brownian_output: Optional[PyTree[Array]] = None

    def _stack(init_tree, traced_tree):
        return jax.tree_util.tree_map(
            lambda init, tr: jnp.concatenate(
                [jnp.asarray(init)[None], jnp.asarray(tr)], axis=0
            ),
            init_tree,
            traced_tree,
        )

    if trace_enabled and traced is not None:
        if return_brownian:
            state_traced, brownian_traced = traced
            trace_output = _stack(init_filtered, state_traced)
            brownian_init = jax.tree_util.tree_map(
                lambda leaf: jnp.zeros_like(leaf[0]), brownian_traced
            )
            brownian_output = _stack(brownian_init, brownian_traced)
        else:
            trace_output = _stack(init_filtered, traced)
    else:
        trace_output = final_state if not collect_trace else None

    payload: Union[Optional[PyTree[Array]], Tuple[Optional[PyTree[Array]], Optional[PyTree[Array]]]]
    if return_brownian:
        payload = (trace_output, brownian_output)
    else:
        payload = trace_output

    if return_state:
        frozen_state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
        return frozen_state, payload
    return payload
