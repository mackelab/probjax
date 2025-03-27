from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["build_score_matching_loss", "build_time_dependent_score_matching_loss"]

from probjax.utils.protocols import LossFn, ModelFn, TimeDependentModelFn


def base_score_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    model_jac_fn: Callable,
    vmap_in_args: Sequence[int],
    axis: int,
    tikhonov: Optional[float],
    *args,
    **kwargs,
) -> Array:
    """Base function for score matching loss.

    Args:
        model_fn: Function that predicts the score
        model_jac_fn: Function that computes the Jacobian of the score
        vmap_in_args: Arguments to vectorize over
        axis: Axis along which to sum the loss
        tikhonov: Optional Tikhonov regularization parameter
        *args: Additional arguments passed to model_fn
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    score = model_fn(*args, **kwargs)
    _model_jac_fn = partial(model_jac_fn, **kwargs)
    _model_jac_fn = jax.vmap(_model_jac_fn, in_axes=vmap_in_args)
    jac_score = _model_jac_fn(*args, **kwargs)

    loss = 0.5 * jnp.sum(score**2, axis=axis) + jnp.trace(
        jac_score, axis1=axis - 1, axis2=axis
    )

    if tikhonov:
        loss += tikonov_regularization(jac_score, tikhonov, axis)
    return loss


def tikonov_regularization(
    jac_score: Array,
    tikhonov: float,
    axis: int,
) -> Array:
    """Compute Tikhonov regularization term.

    Args:
        jac_score: Jacobian of the score
        tikhonov: Regularization parameter
        axis: Axis along which to sum

    Returns:
        Regularization term
    """
    diag_jac = jnp.diagonal(jac_score, axis1=axis - 1, axis2=axis)
    regularizer = tikhonov * jnp.sum(diag_jac**2, axis=axis, keepdims=True)
    return regularizer


def build_score_matching_loss(
    model_fn: ModelFn,
    tikhonov: Optional[float] = None,
    reduction_fn: Callable = jnp.mean,
    jac_fn: Callable = partial(jax.jacfwd, argnums=0),
    axis: int = -1,
) -> LossFn:
    """Build a score matching loss function.

    Args:
        model_fn: Function that predicts the score
        tikhonov: Optional Tikhonov regularization parameter
        reduction_fn: Function to reduce the loss to a scalar
        jac_fn: Function to compute Jacobian
        axis: Axis along which to sum the loss

    Returns:
        A loss function that takes inputs and returns a scalar loss value
    """
    model_jac_fn = jac_fn(model_fn)

    def loss_fn(*args, rng=None, **kwargs):
        """Compute score matching loss.

        Args:
            *args: Arguments passed to model_fn
            rng: Random number generator key
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        vmap_in_args = (0,) * len(args)
        losses = base_score_matching_loss(
            model_fn,
            model_jac_fn,
            vmap_in_args,
            axis,
            tikhonov,
            *args,
            **kwargs,
        )

        return reduction_fn(losses)

    return loss_fn


def build_time_dependent_score_matching_loss(
    model_fn: TimeDependentModelFn,
    mean_fn: Callable,
    std_fn: Callable,
    axis: int = -1,
    tikhonov: Optional[float] = None,
    jac_fn: Callable = partial(jax.jacfwd, argnums=1),
    weight_fn: Optional[Callable] = None,
    reduction_fn: Callable = jnp.mean,
) -> LossFn:
    """Build a time-dependent score matching loss function.

    Args:
        model_fn: Function that predicts the score at time t
        mean_fn: Function that computes the mean at time t
        std_fn: Function that computes the standard deviation at time t
        axis: Axis along which to sum the loss
        tikhonov: Optional Tikhonov regularization parameter
        jac_fn: Function to compute Jacobian
        weight_fn: Optional function that computes weights based on time
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time and inputs and returns a scalar loss value
    """
    model_jac_fn = jac_fn(model_fn)

    def loss_fn(times: Array, xs_target: Array, *args, rng=None, **kwargs):
        """Compute time-dependent score matching loss.

        Args:
            times: Time values
            xs_target: Target values
            *args: Additional arguments passed to model_fn
            rng: Random number generator key
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        eps = jax.random.normal(rng, shape=xs_target.shape)

        xs_t = mean_t + std_t * eps

        weight = weight_fn(times) if weight_fn is not None else None
        vmap_in_args = (0,) * len(args)

        losses = base_score_matching_loss(
            model_fn,
            model_jac_fn,
            vmap_in_args,
            axis,
            tikhonov,
            times,
            xs_t,
            *args,
            **kwargs,
        )

        loss = reduction_fn(losses)
        return loss

    return loss_fn
