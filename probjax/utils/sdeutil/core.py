from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree
import numpy as np

from probjax.utils.jaxutils import ravel_arg_fun, ravel_args
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
    "filter_output",
    "check_points",
]


def build_default_filter(filter_out: Optional[Callable], unravel: Callable) -> Callable:
    """Build a filter function for standard output without Brownian motion.

    Args:
        filter_out: Optional filter function for the output
        unravel: Function to unravel flattened arrays back to PyTree structure

    Returns:
        Callable: Filter function that processes the state and info
    """

    def filter_output(state: Any, info: Any) -> PyTree[Array]:
        if filter_out is None:
            return unravel(state.y0)
        return filter_out(unravel(state.y0))

    return filter_output


def build_default_and_brownian_filter(
    filter_out: Optional[Callable], unravel: Callable
) -> Callable:
    """Build a filter function for output including Brownian motion.

    Args:
        filter_out: Optional filter function for the output
        unravel: Function to unravel flattened arrays back to PyTree structure

    Returns:
        Callable: Filter function that processes the state and info
    """

    def filter_output(state: Any, info: Any) -> Tuple[PyTree[Array], PyTree[Array]]:
        if filter_out is None:
            return unravel(state.y0), unravel(info.dWt)
        return filter_out(unravel(state.y0)), filter_out(unravel(info.dWt))

    return filter_output


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
    filter_output: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
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
        return_brownian: Whether to return Brownian motion paths
        return_state: Whether to return solver state
        noise_type: Type of noise ("diagonal" or "general")
        filter_output: Optional function to filter the output
        check_points: Optional sequence of indices for grid integration

    Returns:
        Depending on return_brownian and return_state:
        - If return_brownian=False, return_state=False: solution trajectory
        - If return_brownian=False, return_state=True: (state, solution)
        - If return_brownian=True, return_state=False: (solution, brownian_paths)
        - If return_brownian=True, return_state=True: (state, solution, brownian_paths)
    """
    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_map(lambda x: x.astype(dtype), y0)

    y0 = jax.tree_map(jnp.atleast_1d, y0)
    ts = jnp.atleast_1d(ts)

    flat_y0, unravel = ravel_args(y0)
    drift = ravel_arg_fun(drift, unravel, 1)
    diffusion = ravel_arg_fun(diffusion, unravel, 1)

    if filter_output is not None:
        with jax.ensure_compile_time_eval():
            flat_y0_indices = np.arange(len(flat_y0))
            y0_indices = unravel(flat_y0_indices)
            filtered_indices = filter_output(y0_indices)
            flat_filtered_indices, _ = ravel_args(filtered_indices)

            def raveled_filter(yi, info):
                del info
                return yi[flat_filtered_indices]

    else:
        raveled_filter = None

    method, _ = get_method(method)

    if noise_type is None:
        g0 = jnp.asarray(diffusion(ts[0], flat_y0))
        noise_type = "diagonal" if g0.ndim <= 1 else "general"

    method = partial(method, sde_type=sde_type, noise_type=noise_type)

    if not return_brownian:
        filter_unravel = build_default_filter(filter_output, unravel)
        state, ys = _sdeint_on_grid(
            method,
            drift,
            diffusion,
            rng,
            y0,
            ts,
            *args,
            filter_output=filter_unravel,
            check_points=check_points,
        )
        if filter_output is None:
            ys = jax.vmap(unravel)(ys)
            ys = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys
            )
        else:
            y0_filtered = filter_output(y0)
            _, unravel_filtered = ravel_args(y0_filtered)
            ys = jax.tree_map(jnp.atleast_1d, ys)
            ys = jax.vmap(unravel_filtered)(ys)
            ys = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0_filtered, ys
            )
        if return_state:
            return state, ys
        else:
            return ys
    else:
        filter_unravel = build_default_and_brownian_filter(filter_output, unravel)
        state, (ys, dWt) = _sdeint_on_grid(
            method,
            drift,
            diffusion,
            rng,
            y0,
            ts,
            *args,
            filter_output=filter_unravel,
            check_points=check_points,
        )
        if filter_output is None:
            ys = jax.vmap(unravel)(ys)
            ys = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0, ys
            )
            dWt = jax.vmap(unravel)(dWt)
            dWt0 = jax.tree_map(jnp.zeros_like, y0)
            dWt = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), dWt0, dWt
            )
        else:
            y0_filtered = filter_output(y0)
            _, unravel_filtered = ravel_args(y0_filtered)
            ys = jax.tree_map(jnp.atleast_1d, ys)
            ys = jax.vmap(unravel_filtered)(ys)
            ys = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), y0_filtered, ys
            )
            dWt = jax.tree_map(jnp.atleast_1d, dWt)
            dWt = jax.vmap(unravel_filtered)(dWt)
            dWt0 = jax.tree_map(jnp.zeros_like, y0_filtered)
            dWt = jax.tree_map(
                lambda x, y: jnp.concatenate([x[None], y], axis=0), dWt0, dWt
            )
        if return_state:
            return state, ys, dWt
        else:
            return ys, dWt
