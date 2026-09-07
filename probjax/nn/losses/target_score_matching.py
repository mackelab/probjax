from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from probjax.nn.losses._common import (
    _require_rng,
    _resolve_weight,
    _sample_eps,
)
from probjax.utils.protocols import ModelFn, TimeDependentModelFn

__all__ = [
    "build_target_score_matching_loss",
    "build_time_dependent_target_score_matching_loss",
]


def base_target_score_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
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
    model_fn: ModelFn,
    score_fn: Callable,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    reduction_fn: Callable = jnp.mean,
):
    def loss_fn(*args, rng=None, **kwargs):
        _require_rng(rng)
        shape = args[argnums].shape
        eps = _sample_eps(rng, shape)

        loss = base_target_score_matching_loss(
            model_fn,
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
    model_fn: TimeDependentModelFn,
    score_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Optional[Callable] = None,
    argnums: int = 0,
    axis: int = -1,
    reduction_fn: Callable = jnp.mean,
) -> Callable:
    def loss_fn(times, *args, rng=None, **kwargs):
        _require_rng(rng)
        x = args[argnums]
        mean = mean_fn(times, x)
        std_t = std_fn(times, x)
        eps = _sample_eps(rng, x.shape)
        new_args = (times,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = _resolve_weight(weight_fn, times)

        loss = base_target_score_matching_loss(
            model_fn,
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
