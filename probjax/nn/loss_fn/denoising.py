from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, PyTree
from jax.typing import ArrayLike

from flax import nnx

__all__ = ["build_denoising_loss", "build_time_dependent_denoising_loss"]


def base_denoising_loss(
    model: nnx.Module | Callable,
    eps: ArrayLike,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    *args,
    **kwargs,
):
    x = args[argnums]
    x_noisy = x + std * eps

    new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]
    x_pred = model(*new_args, **kwargs)

    loss = jnp.sum((x_pred - x) ** 2, axis=axis)

    if control_variate:
        raise NotImplementedError("Control variate is not implemented yet.")

    loss = loss * jnp.squeeze(weight) if weight is not None else loss

    return loss


def build_denoising_loss(
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
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        loss = base_denoising_loss(
            model, eps, std, weight, axis, argnums, control_variate, *args, **kwargs
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_loss(
    model: nnx.Module | Callable,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Callable,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    update_params: Callable = nnx.update,
    reduction_fn: Callable = jnp.mean,
):
    def loss_fn(params, t, *args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        x = args[argnums]
        mean = mean_fn(t, x)
        std_t = std_fn(t, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = (t,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = weight_fn(t)

        loss = base_denoising_loss(
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
