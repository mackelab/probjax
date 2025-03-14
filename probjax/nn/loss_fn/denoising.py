from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from jaxtyping import Array

from probjax.nn.loss_fn.protocols import (
    LossFn,
    ReductionFn,
    TimeDependentFn,
    WeightFn,
)

__all__ = ["build_denoising_loss", "build_time_dependent_denoising_loss"]


class ModelFn(Protocol):
    """Protocol for model functions."""

    def __call__(self, *args, **kwargs) -> Array:
        """
        A function that predicts the denoised output from noisy input.

        Returns:
            Denoised prediction
        """
        ...


class CopulaFn(Protocol):
    """Protocol for copula transformation functions."""

    def __call__(self, x: Array, eps: Array) -> tuple[Array, Array]:
        """
        A function that transforms data and noise using a copula.

        Args:
            x: Input data
            eps: Random noise

        Returns:
            Transformed data and noise
        """
        ...


def base_denoising_loss(
    model: ModelFn,
    eps: ArrayLike,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    loss_mask: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    copula: Optional[CopulaFn],
    *args,
    **kwargs,
):
    x = args[argnums]

    if copula is not None:
        x, eps = copula(x, eps)

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
    model: ModelFn,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[CopulaFn] = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    def loss_fn(*args, rng=None, loss_mask=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_loss(
            model,
            eps,
            std,
            weight,
            loss_mask,
            _axis,
            argnums,
            control_variate,
            copula,
            *args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_loss(
    model: ModelFn,
    mean_fn: TimeDependentFn,
    std_fn: TimeDependentFn,
    weight_fn: WeightFn,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[CopulaFn] = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    def loss_fn(t, *args, rng=None, loss_mask=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        x = args[argnums]
        mean = mean_fn(t, x)
        std_t = std_fn(t, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = (t,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = weight_fn(t)

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_loss(
            model,
            eps,
            std_t,
            weight,
            loss_mask,
            _axis,
            argnums + 1,
            control_variate,
            copula,
            *new_args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn
