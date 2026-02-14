from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree

from probjax.utils.jaxutils import ravel_args
from probjax.utils.odeutil.filters import TraceFilter
from probjax.utils.sdeutil.base import get_method
from probjax.utils.sdeutil.integrate_on_grid import _sdeint_on_grid

STATIC_NAMES = [
    "drift",
    "diffusion",
    "method",
    "dtype",
    "sde_type",
    "return_brownian",
    "return_state",
    "filter_state",
    "collect_trace",
    "check_points",
]


def _bind_sde_function_args(
    drift: Callable,
    diffusion: Callable,
    sde_args: Sequence[Any],
    sde_kwargs: Optional[Mapping[str, Any]],
) -> tuple[Callable, Callable]:
    """Bind shared positional/keyword arguments for drift and diffusion."""
    args = tuple(sde_args)
    kwargs = {} if sde_kwargs is None else dict(sde_kwargs)

    def drift_bound(t, y):
        return drift(t, y, *args, **kwargs)

    def diffusion_bound(t, y):
        return diffusion(t, y, *args, **kwargs)

    return drift_bound, diffusion_bound


def _infer_noise_layout(diffusion_shape: Any, state_dim: int) -> tuple[str, int]:
    """Infer noise layout from diffusion output shape.

    - 0D/1D output => diagonal noise with noise dimension = state dimension.
    - 2D output => full/general noise with noise dimension = trailing axis.
    """
    if hasattr(diffusion_shape, "shape"):
        shape = tuple(diffusion_shape.shape)
        if len(shape) <= 1:
            return "diagonal", state_dim
        if len(shape) == 2:
            if int(shape[0]) != state_dim:
                raise ValueError(
                    "Full diffusion must have shape (state_dim, noise_dim)."
                )
            return "general", int(shape[1])
        raise ValueError(
            "Diffusion output must be scalar/vector (diagonal) or matrix (full)."
        )

    # PyTree diffusion outputs are treated as diagonal noise.
    return "diagonal", state_dim


@partial(jax.jit, static_argnames=STATIC_NAMES)
def _sdeint(
    rng: Key,
    drift: Callable,
    diffusion: Callable,
    y0: PyTree[Array],
    ts: Array,
    sde_args: Sequence[Any] = (),
    sde_kwargs: Optional[Mapping[str, Any]] = None,
    method: str = "euler_maruyama",
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    return_state: bool = False,
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
        sde_args: Additional positional arguments passed to drift and diffusion.
        sde_kwargs: Additional keyword arguments passed to drift and diffusion.
        method: Integration method
        dtype: Data type for computation
        sde_type: Type of SDE ("ito" or "stratonovich")
        return_brownian: Whether to return Brownian motion paths (requires `collect_trace=True`)
        return_state: Whether to return solver state
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

    drift, diffusion = _bind_sde_function_args(drift, diffusion, sde_args, sde_kwargs)

    flat_y0, unravel = ravel_args(y0)
    flat_state_dim = int(flat_y0.shape[0])

    def drift_raveled(t, yi):
        yi_tree = unravel(yi)
        drift_tree = drift(t, yi_tree)
        drift_flat, _ = ravel_args(drift_tree)
        return drift_flat

    def diffusion_unraveled(t, yi):
        yi_tree = unravel(yi)
        return diffusion(t, yi_tree)

    diffusion_shape = jax.eval_shape(diffusion_unraveled, ts[0], flat_y0)
    noise_type, noise_dim = _infer_noise_layout(diffusion_shape, flat_state_dim)

    if noise_type == "diagonal":

        def diffusion_solver(t, yi):
            diffusion_tree = diffusion_unraveled(t, yi)
            diffusion_flat, _ = ravel_args(diffusion_tree)
            return diffusion_flat

    else:

        def diffusion_solver(t, yi):
            diffusion_value = jnp.asarray(diffusion_unraveled(t, yi))
            if diffusion_value.ndim != 2:
                raise ValueError(
                    "Full diffusion must return a matrix with shape (state_dim, noise_dim)."
                )
            if int(diffusion_value.shape[0]) != flat_state_dim:
                raise ValueError(
                    "Full diffusion must have leading dimension equal to state dimension."
                )
            return diffusion_value

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

    solver_api, _ = get_method(method)

    solver_method = partial(
        solver_api,
        sde_type=sde_type,
        noise_type=noise_type,
        noise_dim=noise_dim,
    )

    can_unravel_brownian = noise_type == "diagonal" and noise_dim == flat_state_dim

    if trace_enabled:

        def trace_filter(state, info):
            filtered_state = apply_filter(unravel(state.y0))
            if return_brownian:
                if can_unravel_brownian:
                    filtered_bm = apply_filter(unravel(info.dWt))
                else:
                    filtered_bm = info.dWt
                return filtered_state, filtered_bm
            return filtered_state

        trace_fn = trace_filter
    else:
        trace_fn = None

    state, traced = _sdeint_on_grid(
        solver_method,
        drift_raveled,
        diffusion_solver,
        rng,
        flat_y0,
        ts,
        filter_output=trace_fn,
        check_points=check_points,
        collect_trace=trace_enabled,
    )

    state_y0 = getattr(state, "y0")
    final_state = apply_filter(unravel(state_y0))

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

    payload: Union[
        Optional[PyTree[Array]], Tuple[Optional[PyTree[Array]], Optional[PyTree[Array]]]
    ]
    if return_brownian:
        payload = (trace_output, brownian_output)
    else:
        payload = trace_output

    if return_state:
        frozen_state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
        return frozen_state, payload
    return payload
