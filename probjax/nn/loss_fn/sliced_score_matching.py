from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, PyTree

from flax import nnx

__all__ = [
    "build_sliced_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
]


def base_sliced_score_matching_loss(
    model_fn_and_jvp: Callable,
    slice_dist: Callable,
    vmap_in_args: Sequence[int],
    argnums: int,
    num_slices: int,
    rng,
    *args,
    **kwargs,
):
    x_shape = args[argnums].shape
    vs = slice_dist(rng, shape=(num_slices, *x_shape)).astype(jnp.float32)
    model_fn_and_jvp = partial(model_fn_and_jvp, **kwargs)
    # Over batches
    _model_fn_and_jvp = jax.vmap(model_fn_and_jvp, in_axes=(0,) + vmap_in_args)
    # Over slices
    _value_and_jvp = jax.vmap(_model_fn_and_jvp, in_axes=(0,) + (None,) * len(args))
    sliced_score, jac_trace, reg = _value_and_jvp(vs, *args)

    loss = 0.5 * sliced_score**2 + jac_trace
    if reg is not None:
        loss += reg

    # Average over slices
    loss = jnp.mean(loss, axis=0)
    return loss


def build_sliced_score_matching_loss(
    model: nnx.Module | Callable,
    num_slices: int,
    tikhonov: Optional[float] = None,
    reduction_fn: Callable = jnp.mean,
    update_params: Callable = nnx.update,
    argnums: int = 0,
    slice_dist: Callable = jax.random.rademacher,
    axis: int = -1,
):
    model_fn = model

    def value_and_jvp(v, *args, **kwargs):
        _f = lambda x: model_fn(args[:argnums] + (x,) + args[argnums + 1 :], **kwargs)
        value, jvp = jax.jvp(_f, (args[argnums],), (v,))
        sliced_value = jnp.sum(value * v, axis)
        sliced_jvp = jnp.sum(jvp * v, axis)

        reg = tikhonov * jnp.sum((jvp * v) ** 2, axis) if tikhonov is not None else None
        return sliced_value, sliced_jvp, reg

    def loss_fn(params: PyTree, *args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        vmap_in_args = (0,) * len(args)
        losses = base_sliced_score_matching_loss(
            value_and_jvp,
            slice_dist,
            vmap_in_args,
            argnums,
            num_slices,
            rng,
            *args,
            **kwargs,
        )

        return reduction_fn(losses)

    return loss_fn


def build_time_dependent_sliced_score_matching_loss(
    model: nnx.Module | Callable,
    mean_fn: Callable,
    std_fn: Callable,
    num_slices: int,
    update_params: Callable = nnx.update,
    argnums: int = 1,
    slice_dist: Callable = jax.random.rademacher,
    axis: int = -1,
):
    sliced_score_matching_loss = build_sliced_score_matching_loss(
        model,
        num_slices,
        update_params=update_params,
        argnums=argnums,
        slice_dist=slice_dist,
        axis=axis,
    )

    def loss_fn(
        params: PyTree, times: Array, xs_target: Array, *args, rng=None, **kwargs
    ):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        rng_samples, rng_slices = jax.random.split(rng)
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        eps = jax.random.normal(rng_samples, shape=xs_target.shape)

        xs_target = mean_t + std_t * eps

        return sliced_score_matching_loss(
            params, rng_slices, times, xs_target, *args, **kwargs
        )

    return loss_fn
