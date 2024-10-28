from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, PyTree
from jax.typing import ArrayLike

from flax import nnx

__all__ = [
    "build_denoising_score_matching_loss",
    "build_time_dependent_denoising_score_matching_loss",
]


def base_denoising_score_matching_loss(
    model_fn: Callable,
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
    model_fn: Callable, eps, std, axis, argnums, *args, **kwargs
):
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
    model: nnx.Module | Callable,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    update_params: Callable = nnx.update,
    reduction_fn: Callable = jnp.mean,
):
    def loss_fn(params, *args, rng=None, **kwargs):
        assert rng is not None, "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        loss = base_denoising_score_matching_loss(
            model, eps, std, weight, axis, argnums, control_variate, *args, **kwargs
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_score_matching_loss(
    model: nnx.Module | Callable,
    mean_fn: Callable,
    std_fn: Callable,
    argnums: int = 1,
    axis: int = -1,
    control_variate: bool = False,
    update_params: Callable = nnx.update,
    reduction_fn: Callable = jnp.mean,
):
    def loss_fn(params, times, *args, rng=None, **kwargs):
        assert rng is not None, "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        x = args[argnums]
        mean = mean_fn(times, x)
        std_t = std_fn(times, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = args[:argnums] + (mean,) + args[argnums + 1 :]

        loss = base_denoising_score_matching_loss(
            model, eps, std_t, axis, argnums, control_variate, *new_args, **kwargs
        )

        return reduction_fn(loss)

    return loss_fn
