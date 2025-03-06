from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, ArrayLike, PyTree

__all__ = [
    "build_sliced_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
]


def base_target_score_matching_loss(
    model_fn: Callable,
    score_fn: Callable,
    eps: Array,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    axis: int,
    argnums: int,
    *args,
    **kwargs,
):
    x = args[argnums]
    x_noisy = x + eps * std
    new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]
    score_pred = model_fn(*new_args, **kwargs)
    score_target = score_fn(*args, **kwargs)

    loss = jnp.sum((score_pred - score_target) ** 2, axis=axis)
    loss = loss * weight if weight is not None else loss
    return loss


def build_target_score_matching_loss(
    model: nnx.Module | Callable,
    score_fn: Callable,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    update_params: Callable = nnx.update,
    reduction_fn: Callable = jnp.mean,
):
    def loss_fn(params, *args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        loss = base_target_score_matching_loss(
            model,
            score_fn,
            eps,
            std,
            weight,
            axis,
            argnums,
            *args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_target_score_matching_loss(
    model: nnx.Module | Callable,
    score_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Optional[Callable] = None,
    argnums: int = 0,
    axis: int = -1,
    update_params: Callable = nnx.update,
    reduction_fn: Callable = jnp.mean,
) -> Callable:
    def loss_fn(params, times, *args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        x = args[argnums]
        mean = mean_fn(times, x)
        std_t = std_fn(times, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = (times,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = weight_fn(times) if weight_fn is not None else None

        loss = base_target_score_matching_loss(
            model,
            score_fn,
            eps,
            std_t,
            weight,
            axis,
            argnums + 1,
            *new_args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn
