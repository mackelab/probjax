from typing import Callable, Optional, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from jaxtyping import Array

from probjax.nn.loss_fn.protocols import (
    LossFn,
    ReductionFn,
    ScoreModelFn,
    TimeDependentFn,
    WeightFn,
)

__all__ = [
    "build_denoising_score_matching_loss",
    "build_time_dependent_denoising_score_matching_loss",
]


def base_denoising_score_matching_loss(
    model_fn: ScoreModelFn,
    eps: Array,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    *args,
    **kwargs,
):
    x = args[argnums]
    x_noisy = x + eps * std
    new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]
    score_pred = model_fn(*new_args, **kwargs)
    score_target = eps / std

    loss = jnp.sum((score_pred + score_target) ** 2, axis=axis)

    if control_variate:
        cv = control_variate_taylor(model_fn, eps, std, axis, argnums, *args, **kwargs)
        beta = control_variate_scaling(loss, cv)
        loss = loss - beta * cv
    loss = loss * weight if weight is not None else std**2 * loss
    return loss


def control_variate_taylor(
    model_fn: ScoreModelFn, eps, std, axis, argnums, *args, **kwargs
):
    # NOTE: maybe new jax.experimental.jet can be used for efficient higher order
    # control variate computation
    s = model_fn(*args, **kwargs)

    term1 = 2 / std * jnp.sum(eps * s, axis=axis, keepdims=True)
    term2 = jnp.sum(eps**2, axis=axis, keepdims=True) / std**2
    term3 = args[argnums].shape[axis] / std**2

    cv = jnp.mean(term3 - term1 - term2, axis=axis)

    return cv


def control_variate_scaling(
    loss,
    cv,
):
    assert loss.shape == cv.shape, "Loss and control variate must have the same shape."
    cv_var = jnp.std(cv)
    loss_var = jnp.std(loss)
    cv_loss_covar = jnp.mean((loss - jnp.mean(loss)) * (cv - jnp.mean(cv)))

    beta = cv_loss_covar / (cv_var * loss_var)
    return beta


def build_denoising_score_matching_loss(
    model: nnx.Module | ScoreModelFn,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    def loss_fn(*args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        loss = base_denoising_score_matching_loss(
            model, eps, std, weight, axis, argnums, control_variate, *args, **kwargs
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_score_matching_loss(
    model: ScoreModelFn,
    mean_fn: TimeDependentFn,
    std_fn: TimeDependentFn,
    weight_fn: Optional[WeightFn] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    def loss_fn(times, *args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        x = args[argnums]
        mean = mean_fn(times, x)
        std_t = std_fn(times, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = (times,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = weight_fn(times) if weight_fn is not None else None

        loss = base_denoising_score_matching_loss(
            model,
            eps,
            std_t,
            weight,
            axis,
            argnums + 1,
            control_variate,
            *new_args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn
