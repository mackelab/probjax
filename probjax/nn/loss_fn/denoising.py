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
    loss_mask: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    *args,
    **kwargs,
):
    x = args[argnums]
    x_noisy = x + std * eps
    if loss_mask is not None:
        x_noisy = jnp.where(loss_mask, x, x_noisy)

    new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]
    x_pred = model(*new_args, **kwargs)
    
    loss = (x_pred - x) ** 2
    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    loss = jnp.sum(loss, axis=axis)

    if control_variate:
        raise NotImplementedError("Control variate is not implemented yet.")

    loss = loss * weight.reshape(loss.shape) if weight is not None else loss

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
    def loss_fn(params, *args, rng=None, loss_mask=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        update_params(model, params)
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        loss = base_denoising_loss(
            model,
            eps,
            std,
            weight,
            loss_mask,
            axis,
            argnums,
            control_variate,
            *args,
            **kwargs,
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
    def loss_fn(params, t, *args, rng=None, loss_mask=None, **kwargs):
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
            loss_mask,
            axis,
            argnums + 1,
            control_variate,
            *new_args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn
