from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from probjax.utils.protocols import (
    LossFn,
    ModelFn,
    ReductionFn,
    TimeDependentFn,
    TimeDependentModelFn,
    WeightFn,
)

__all__ = ["build_denoising_loss", "build_time_dependent_denoising_loss"]


def base_denoising_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    eps: Array,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    loss_mask: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    copula: Optional[Callable],
    *args,
    **kwargs,
) -> Array:
    """Base function for denoising loss.

    Args:
        model_fn: Function that predicts the denoised output
        eps: Noise samples
        std: Standard deviation of the noise
        weight: Optional weight for the loss
        loss_mask: Optional mask for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument to add noise to
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        *args: Additional arguments passed to model_fn
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    x = args[argnums]

    if copula is not None:
        x, eps = copula(x, eps)

    x_noisy = x + std * eps
    if loss_mask is not None:
        x_noisy = jnp.where(loss_mask, x, x_noisy)

    new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]
    x_pred = model_fn(*new_args, **kwargs)

    loss = (x_pred - x) ** 2
    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    loss = jnp.sum(loss, axis=axis)

    if control_variate:
        raise NotImplementedError("Control variate is not implemented yet.")

    loss = loss * weight.reshape(loss.shape) if weight is not None else loss

    return loss


def build_denoising_loss(
    model_fn: ModelFn,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[Callable] = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a denoising loss function.

    Args:
        model_fn: Function that predicts the denoised output
        std: Standard deviation of the noise
        weight: Optional weight for the loss
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes inputs and returns a scalar loss value
    """

    def loss_fn(*args, rng=None, loss_mask=None, **kwargs):
        assert rng is not None, (
            "loss_fn does require rngs, pass them to function kwargs."
        )
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_loss(
            model_fn,
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
    model_fn: TimeDependentModelFn,
    mean_fn: TimeDependentFn,
    std_fn: TimeDependentFn,
    weight_fn: WeightFn,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[Callable] = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a time-dependent denoising loss function.

    Args:
        model_fn: Function that predicts the denoised output at time t
        mean_fn: Function that computes the mean at time t
        std_fn: Function that computes the standard deviation at time t
        weight_fn: Function that computes weights based on time
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time and inputs and returns a scalar loss value
    """

    def loss_fn(t, *args, rng=None, loss_mask=None, **kwargs):
        assert rng is not None, (
            "loss_fn does require rngs, pass them to function kwargs."
        )
        x = args[argnums]
        mean = mean_fn(t, x)
        std_t = std_fn(t, x)
        eps = jax.random.normal(rng, shape=x.shape)
        new_args = (t,) + args[:argnums] + (mean,) + args[argnums + 1 :]
        weight = weight_fn(t)

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_loss(
            model_fn,
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
