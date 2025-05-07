from typing import Optional

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

__all__ = [
    "build_denoising_score_matching_loss",
    "build_time_dependent_denoising_score_matching_loss",
]


def base_denoising_score_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    eps: Array,
    std: ArrayLike,
    weight: Optional[ArrayLike],
    axis: int,
    argnums: int,
    control_variate: bool,
    *args,
    **kwargs,
) -> Array:
    """Base function for denoising score matching loss.

    Args:
        model_fn: Function that predicts the score
        eps: Noise samples
        std: Standard deviation of the noise
        weight: Optional weight for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument to add noise to
        control_variate: Whether to use control variate for variance reduction
        *args: Additional arguments passed to model_fn
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
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
    model_fn: ModelFn, eps, std, axis, argnums, *args, **kwargs
) -> Array:
    """Compute Taylor expansion based control variate for variance reduction.

    Args:
        model_fn: Function that predicts the score
        eps: Noise samples
        std: Standard deviation of the noise
        axis: Axis along which to sum
        argnums: Index of the input argument
        *args: Additional arguments passed to model_fn
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of control variate values
    """
    s = model_fn(*args, **kwargs)
    term1 = 2 / std * jnp.sum(eps * s, axis=axis, keepdims=True)
    term2 = jnp.sum(eps**2, axis=axis, keepdims=True) / std**2
    # Multiply dimension of axis to get total number of elements
    # in the batch
    shape = args[argnums].shape
    d = jnp.prod(jnp.array([shape[a] for a in axis]))
    term3 = d / std**2

    cv = jnp.mean(term3 - term1 - term2, axis=axis)

    return cv


def control_variate_scaling(loss: Array, cv: Array) -> Array:
    """Compute optimal scaling for control variate.

    Args:
        loss: Array of loss values
        cv: Array of control variate values

    Returns:
        Optimal scaling factor
    """
    assert loss.shape == cv.shape, "Loss and control variate must have the same shape."
    cv_var = jnp.std(cv)
    loss_var = jnp.std(loss)
    cv_loss_covar = jnp.mean((loss - jnp.mean(loss)) * (cv - jnp.mean(cv)))

    beta = cv_loss_covar / (cv_var * loss_var)
    return beta


def build_denoising_score_matching_loss(
    model_fn: ModelFn,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a denoising score matching loss function.

    Args:
        model_fn: Function that predicts the score
        std: Standard deviation of the noise
        weight: Optional weight for the loss
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes inputs and returns a scalar loss value
    """

    def loss_fn(*args, rng=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        shape = args[argnums].shape
        eps = jax.random.normal(rng, shape=shape)

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_score_matching_loss(
            model_fn, eps, std, weight, _axis, argnums, control_variate, *args, **kwargs
        )

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_score_matching_loss(
    model_fn: TimeDependentModelFn,
    mean_fn: TimeDependentFn,
    std_fn: TimeDependentFn,
    weight_fn: Optional[WeightFn] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a time-dependent denoising score matching loss function.

    Args:
        model_fn: Function that predicts the score at time t
        mean_fn: Function that computes the mean at time t
        std_fn: Function that computes the standard deviation at time t
        weight_fn: Optional function that computes weights based on time
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time and inputs and returns a scalar loss value
    """

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

        _axis = kwargs.pop("axis", axis)

        loss = base_denoising_score_matching_loss(
            model_fn,
            eps,
            std_t,
            weight,
            _axis,
            argnums + 1,
            control_variate,
            *new_args,
            **kwargs,
        )

        return reduction_fn(loss)

    return loss_fn
