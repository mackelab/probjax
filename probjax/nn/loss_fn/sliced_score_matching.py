from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from probjax.utils.protocols import LossFn, ModelFn, TimeDependentModelFn, WeightFn

__all__ = [
    "build_sliced_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
]


def base_sliced_score_matching_loss(
    model_fn_and_jvp: ModelFn | TimeDependentModelFn,
    slice_dist: Callable,
    vmap_in_args: Sequence[int],
    weight: Optional[ArrayLike],
    axis: int,
    argnums: int,
    num_slices: int,
    rng,
    *args,
    **kwargs,
) -> Array:
    """Base function for sliced score matching loss.

    Args:
        model_fn_and_jvp: Function that predicts the score and its Jacobian-vector product
        slice_dist: Function that generates random slices
        vmap_in_args: Arguments to vectorize over
        weight: Optional weight for the loss
        axis: Axis along which to sum the loss
        argnums: Index of the input argument
        num_slices: Number of random slices to use
        rng: Random number generator key
        *args: Additional arguments passed to model_fn
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    x_shape = args[argnums].shape
    vs = slice_dist(rng, shape=(num_slices, *x_shape)).astype(jnp.float32)
    model_fn_and_jvp = partial(model_fn_and_jvp, **kwargs)
    # Over batches
    _model_fn_and_jvp = jax.vmap(model_fn_and_jvp, in_axes=(0,) + vmap_in_args)
    # Over slices
    _value_and_jvp = jax.vmap(_model_fn_and_jvp, in_axes=(0,) + (None,) * len(args))
    sliced_score, jac_trace, reg = _value_and_jvp(vs, *args)

    loss = 0.5 * sliced_score**2 + jac_trace
    if reg is not None:
        loss += reg

    loss = jnp.sum(loss, axis=axis)
    loss = loss * weight if weight is not None else loss

    return loss


def build_sliced_score_matching_loss(
    model_fn: ModelFn,
    num_slices: int,
    tikhonov: Optional[float] = None,
    weight: Optional[ArrayLike] = None,
    argnums: int = 0,
    axis: int = -1,
    slice_dist: Callable = jax.random.rademacher,
    reduction_fn: Callable = jnp.mean,
) -> LossFn:
    """Build a sliced score matching loss function.

    Args:
        model_fn: Function that predicts the score
        num_slices: Number of random slices to use
        tikhonov: Optional Tikhonov regularization parameter
        weight: Optional weight for the loss
        argnums: Index of the input argument
        axis: Axis along which to sum the loss
        slice_dist: Function that generates random slices
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes inputs and returns a scalar loss value
    """

    def value_and_jvp(v, *args, **kwargs):
        """Compute score and its Jacobian-vector product.

        Args:
            v: Random slice
            *args: Arguments passed to model_fn
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Tuple of (sliced score, Jacobian trace, regularization term)
        """
        _f = lambda x: model_fn(args[:argnums] + (x,) + args[argnums + 1 :], **kwargs)
        value, jvp = jax.jvp(_f, (args[argnums],), (v,))
        value = value[0] if isinstance(value, tuple) else value
        jvp = jvp[0] if isinstance(jvp, tuple) else jvp
        sliced_value = jnp.sum(value * v, axis)
        sliced_jvp = jnp.sum(jvp * v, axis)

        reg = tikhonov * jnp.sum((jvp * v) ** 2, axis) if tikhonov is not None else None
        return sliced_value, sliced_jvp, reg

    def loss_fn(*args, rng=None, **kwargs):
        """Compute sliced score matching loss.

        Args:
            *args: Arguments passed to model_fn
            rng: Random number generator key
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        assert (
            rng is not None
        ), "loss_fn does require rngs, pass them to function kwargs."
        vmap_in_args = (0,) * len(args)
        losses = base_sliced_score_matching_loss(
            value_and_jvp,
            slice_dist,
            vmap_in_args,
            weight,
            axis,
            argnums,
            num_slices,
            rng,
            *args,
            **kwargs,
        )

        return reduction_fn(losses)

    return loss_fn


def build_time_dependent_sliced_score_matching_loss(
    model: TimeDependentModelFn,
    mean_fn: Callable,
    std_fn: Callable,
    num_slices: int,
    weight_fn: Optional[WeightFn] = None,
    argnums: int = 1,
    axis: int = -1,
    slice_dist: Callable = jax.random.rademacher,
    reduction_fn: Callable = jnp.mean,
) -> LossFn:
    """Build a time-dependent sliced score matching loss function.

    Args:
        model: Function that predicts the score at time t
        mean_fn: Function that computes the mean at time t
        std_fn: Function that computes the standard deviation at time t
        num_slices: Number of random slices to use
        weight_fn: Optional function that computes weights based on time
        argnums: Index of the input argument
        axis: Axis along which to sum the loss
        slice_dist: Function that generates random slices
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time and inputs and returns a scalar loss value
    """

    def loss_fn(times: Array, xs_target: Array, *args, rng=None, **kwargs):
        """Compute time-dependent sliced score matching loss.

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
        rng_samples, rng_slices = jax.random.split(rng)
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        eps = jax.random.normal(rng_samples, shape=xs_target.shape)

        xs_target = mean_t + std_t * eps

        weight = weight_fn(times) if weight_fn is not None else None
        vmap_in_args = (0,) * len(args)

        loss = base_sliced_score_matching_loss(
            model,
            slice_dist,
            vmap_in_args,
            weight,
            axis,
            argnums,
            num_slices,
            rng_slices,
            times,
            xs_target,
            *args,
            **kwargs,
        )

        loss = reduction_fn(loss)
        return loss

    return loss_fn
