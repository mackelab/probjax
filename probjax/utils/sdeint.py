from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

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
    "filter_output",
    "check_points",
]


def build_default_filter(filter_out, unravel):
    def filter_output(state, info):
        if filter_out is None:
            return unravel(state.y0)
        return filter_out(unravel(state.y0))

    return filter_output


def build_default_and_brownian_filter(filter_out, unravel):
    def filter_output(state, info):
        if filter_out is None:
            return unravel(state.y0), unravel(info.dWt)
        return filter_out(unravel(state.y0), filter_out(unravel(info.dWt)))

    return filter_output


@partial(jax.jit, static_argnames=STATIC_NAMES)
def sdeint(
    rng: Key,
    drift: Callable,
    diffusion: Callable,
    y0: Array,
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    dtype=jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    noise_type: Optional[str] = None,
    filter_output: Optional[Callable] = None,
    check_points: Optional[Sequence[int]] = None,
) -> Array:
    """Solve a stochastic differential equation.

    Args:
        key: (PRNGKey): Random generator key.
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments, that are passed both to the drift and diffusion
            functions i.e. parameters!
        method (str, optional): Methods to use. Defaults to "euler_maruyama".
        noise_type (bool, optional): Whether the noise is diagonal. Defaults to False.
        sde_type: asdfasdf

    Raises:
        TypeError: The arguments passed not jax types.
        TypeError: The arguments passed not jax types.

    Returns:
        ys: Solution path of the SDE.
    """

    if dtype is not None:
        ts = ts.astype(dtype)
        y0 = jax.tree_map(lambda x: x.astype(dtype), y0)

    y0 = jax.tree_map(jnp.atleast_1d, y0)
    ts = jnp.atleast_1d(ts)

    flat_y0, unravel = ravel_args(y0)
    drift = ravel_arg_fun(drift, unravel, 1)
    diffusion = ravel_arg_fun(diffusion, unravel, 1)

    method, _ = get_method(method)

    if noise_type is None:
        g0 = jnp.asarray(diffusion(ts[0], flat_y0))
        noise_type = "diagonal" if g0.ndim <= 1 else "general"

    method = partial(method, sde_type=sde_type, noise_type=noise_type)

    if not return_brownian:
        filter_unravel = build_default_filter(filter_output, unravel)
        _, ys = _sdeint_on_grid(
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
        y0_filtered = filter_output(y0) if filter_output else y0
        ys = jax.tree_map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0), y0_filtered, ys
        )
        return ys
    else:
        filter_unravel = build_default_and_brownian_filter(filter_output, unravel)
        _, (ys, dWt) = _sdeint_on_grid(
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
        y0_filtered = filter_output(y0) if filter_output else y0
        ys = jax.tree_map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0), y0_filtered, ys
        )
        dWt0 = jax.tree_map(jnp.zeros_like, y0)
        dWt = jax.tree_map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0), dWt0, dWt
        )
        return ys, dWt
