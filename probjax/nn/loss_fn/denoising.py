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

from probjax.nn.loss_fn.denoising_score_matching import (
    base_denoising_score_matching_loss,
)

__all__ = ["build_denoising_loss", "build_time_dependent_denoising_loss"]


def base_denoising_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    x0: Array,
    eps: Array,
    scale: ArrayLike,
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
        scale: Scaling factor for the data (only used in v-prediction)
        std: Standard deviation of the noise (sigma_t)
        weight: Optional weight for the loss
        loss_mask: Optional mask for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument to add noise to (x_noisy)
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        x0: Original clean data
        *args: Additional arguments passed to model_fn, where args[argnums] is x_noisy
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    # We assume x_noisy is already passed in args[argnums]
    x_pred = model_fn(*args, **kwargs)

    loss = (x_pred - x0) ** 2
    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    loss = jnp.sum(loss, axis=axis)

    if control_variate:
        raise NotImplementedError("Control variate is not implemented yet.")

    loss = loss * weight.reshape(loss.shape) if weight is not None else loss

    return loss


def base_eps_prediction_loss(
    model_fn: ModelFn,
    x0: Array,
    eps: Array,
    scale: ArrayLike,
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
    """Base function for epsilon prediction loss.

    Args:
        model_fn: Function that predicts the noise component
        eps: Noise samples
        scale: Scaling factor for the data (only used in v-prediction)
        std: Standard deviation of the noise (sigma_t)
        weight: Optional weight for the loss
        loss_mask: Optional mask for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument to add noise to (x_noisy)
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        x0: Original clean data
        *args: Additional arguments passed to model_fn, where args[argnums] is x_noisy
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    # Get epsilon prediction directly from model
    eps_pred = model_fn(*args, **kwargs)

    loss = (eps_pred - eps) ** 2

    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    loss = jnp.sum(loss, axis=axis)

    loss = loss * weight.reshape(loss.shape) if weight is not None else loss

    return loss


def base_v_prediction_loss(
    model_fn: ModelFn,
    x0: Array,
    eps: Array,
    scale: ArrayLike,
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
    """Base function for v-prediction loss.

    Args:
        model_fn: Function that predicts the v-component (combination of noise and data)
        eps: Noise samples
        scale: Scaling factor for x0 data
        std: Standard deviation of the noise (sigma_t)
        weight: Optional weight for the loss
        loss_mask: Optional mask for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument to add noise to (x_noisy)
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        x0: Original clean data
        *args: Additional arguments passed to model_fn, where args[argnums] is x_noisy
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """

    # Calculate alpha_t based on std (sigma_t)
    # Assuming alpha_t^2 + sigma_t^2 = 1 relationship
    alpha_t = scale
    sigma_t = std

    # Normalize by total variance for consistency
    total_variance = jnp.sqrt(alpha_t**2 + sigma_t**2)
    normalized_alpha = alpha_t / total_variance
    normalized_sigma = sigma_t / total_variance

    # Calculate v target according to the convention: alpha_t * epsilon - sigma_t * x0
    # Apply scale to x0 in v-prediction
    v_target = normalized_alpha * eps - normalized_sigma * x0

    # Get v prediction from model
    v_pred = model_fn(*args, **kwargs)

    loss = (v_pred - v_target) ** 2

    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))
    loss = jnp.sum(loss, axis=axis)

    loss = loss * weight.reshape(loss.shape) if weight is not None else loss

    return loss


def build_denoising_loss(
    model_fn: ModelFn,
    scale: ArrayLike,
    std: ArrayLike,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[Callable] = None,
    reduction_fn: ReductionFn = jnp.mean,
    prediction_target: str = "x0",
) -> LossFn:
    """Build a denoising loss function.

    Args:
        model_fn: Function that predicts the denoised output
        scale: Scaling factor for the data (used in v-prediction)
        std: Standard deviation of the noise
        weight: Optional weight for the loss
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        reduction_fn: Function to reduce the loss to a scalar
        prediction_target: Target type for prediction ("x0", "eps", "v", or "score")

    Returns:
        A loss function that takes inputs and returns a scalar loss value
    """

    def loss_fn(*args, rng=None, loss_mask=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        x = args[argnums]  # This is x0, the clean data
        shape = x.shape
        eps = jax.random.normal(rng, shape=shape)

        # Calculate alpha based on std (sigma)
        # Assuming alpha^2 + sigma^2 = 1 relationship
        alpha = jnp.sqrt(1.0 - std**2)

        # Create noisy input
        x_noisy = alpha * x + std * eps

        # Create new arguments with x_noisy replacing x
        new_args = args[:argnums] + (x_noisy,) + args[argnums + 1 :]

        _axis = kwargs.pop("axis", axis)

        if prediction_target == "x0":
            loss = base_denoising_loss(
                model_fn,
                x,
                eps,
                scale,
                std,
                weight,
                loss_mask,
                _axis,
                argnums,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "eps":
            loss = base_eps_prediction_loss(
                model_fn,
                x,
                eps,
                scale,
                std,
                weight,
                loss_mask,
                _axis,
                argnums,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "v":
            loss = base_v_prediction_loss(
                model_fn,
                x,
                eps,
                scale,
                std,
                weight,
                loss_mask,
                _axis,
                argnums,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "score":
            loss = base_denoising_score_matching_loss(
                model_fn,
                x,
                eps,
                scale,
                std,
                weight,
                loss_mask,
                _axis,
                argnums,
                *new_args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid prediction target: {prediction_target}")

        return reduction_fn(loss)

    return loss_fn


def build_time_dependent_denoising_loss(
    model_fn: TimeDependentModelFn,
    scale_fn: Callable[[ArrayLike], Array],
    std_fn: Callable[[ArrayLike], Array],
    weight_fn: WeightFn,
    argnums: int = 0,
    axis: int = -1,
    control_variate: bool = False,
    copula: Optional[Callable] = None,
    reduction_fn: ReductionFn = jnp.mean,
    prediction_target: str = "x0",
) -> LossFn:
    """Build a time-dependent denoising loss function.

    Args:
        model_fn: Function that predicts the denoised output at time t
        scale_fn: Function that computes the scaling factor at time t
        std_fn: Function that computes the standard deviation at time t
        weight_fn: Function that computes weights based on time
        argnums: Index of the input argument to add noise to
        axis: Axis along which to sum the loss
        control_variate: Whether to use control variate for variance reduction
        copula: Optional copula function for noise generation
        reduction_fn: Function to reduce the loss to a scalar
        prediction_target: Target type for prediction ("x0", "eps", "v", or "score")

    Returns:
        A loss function that takes time and inputs and returns a scalar loss value
    """

    def loss_fn(t, *args, rng=None, loss_mask=None, **kwargs):
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        x = args[argnums]  # This is x0, the clean data
        alpha_t = scale_fn(t)
        sigma_t = std_fn(t)
        eps = jax.random.normal(rng, shape=x.shape)

        # Create noisy x using alpha_t and sigma_t
        x_noisy = alpha_t * x + sigma_t * eps

        new_args = (t,) + args[:argnums] + (x_noisy,) + args[argnums + 1 :]
        weight = weight_fn(t)

        _axis = kwargs.pop("axis", axis)

        # Get scale directly from scale_fn
        scale = alpha_t

        if prediction_target == "x0":
            loss = base_denoising_loss(
                model_fn,
                x,
                eps,
                scale,
                sigma_t,
                weight,
                loss_mask,
                _axis,
                argnums + 1,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "eps":
            loss = base_eps_prediction_loss(
                model_fn,
                x,
                eps,
                scale,
                sigma_t,
                weight,
                loss_mask,
                _axis,
                argnums + 1,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "v":
            loss = base_v_prediction_loss(
                model_fn,
                x,
                eps,
                scale,
                sigma_t,
                weight,
                loss_mask,
                _axis,
                argnums + 1,
                control_variate,
                copula,
                *new_args,
                **kwargs,
            )
        elif prediction_target == "score":
            loss = base_denoising_score_matching_loss(
                model_fn,
                x,
                eps,
                scale,
                sigma_t,
                weight,
                loss_mask,
                _axis,
                argnums + 1,
                *new_args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid prediction target: {prediction_target}")

        return reduction_fn(loss)

    return loss_fn
