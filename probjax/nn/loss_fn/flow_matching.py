from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

import partial

from probjax.utils.protocols import (
    InterpolationFn,
    InterpolationNoiseFn,
    LossFn,
    ModelFn,
    ReductionFn,
    TimeDependentModelFn,
    WeightFn,
)


def base_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    interpolation_fn: InterpolationFn,
    interpolation_noise_fn: InterpolationNoiseFn | None,
    interpolation_grad: InterpolationNoiseFn,
    interpolation_noise_grad: InterpolationNoiseFn | None,
    weight_fn: WeightFn | None,
    metric_fn: Callable[[Array, Array], Array] | None,
    axis: int,
    t: Array,
    x0: Array,
    x1: Array,
    *args,
    rng: Optional[Array] = None,
    loss_mask: Optional[Array] = None,
    **kwargs,
) -> Array:
    """Base function for flow matching loss.

    Args:
        model_fn: Function that predicts the velocity field
        interpolation_fn: Function that interpolates between x0 and x1 at time t
        interpolation_noise_fn: Optional function that provides noise scale for interpolation
        interpolation_noise_grad: Optional function that computes gradient of noise scale
        interpolation_grad: Function that computes gradient of interpolation
        weight_fn: Optional function that computes weights based on time
        metric_fn: Optional function that computes the Riemannian metric tensor
        axis: Axis along which to sum the loss
        t: Time values
        x0: Starting points
        x1: Ending points
        *args: Additional arguments passed to model_fn
        rng: Random number generator key
        loss_mask: Optional mask for the loss
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    xt = interpolation_fn(x0, x1, t)
    if interpolation_noise_fn:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt += interpolation_noise_fn(x0, x1, t) * eps

    v_t = model_fn(t, xt, *args, **kwargs)
    u_t = interpolation_grad(x0, x1, t)

    if interpolation_noise_fn:
        u_t += interpolation_noise_grad(x0, x1, t) * eps

    # Compute loss using the metric if provided
    if metric_fn is not None:
        # Get the metric tensor at the current point
        metric = metric_fn(xt, t)
        # Compute the squared norm using the metric
        # v_t and u_t should be vectors in the tangent space
        diff = v_t - u_t
        if len(metric.shape) == 2:
            # If metric is a single matrix, broadcast it
            metric = jnp.expand_dims(metric, 0)
        # Compute (v-u)^T M (v-u) for each point
        loss = jnp.sum(diff * jnp.einsum('...ij,...j->...i', metric, diff), axis=axis)
    else:
        # Euclidean (L2) metric
        loss = jnp.sum((v_t - u_t) ** 2, axis=axis)

    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, 0.0)

    if weight_fn:
        loss = loss * weight_fn(t).reshape(loss.shape)

    return loss


def base_mean_flow_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    interpolation_fn: InterpolationFn,
    interpolation_noise_fn: Optional[InterpolationNoiseFn],
    interpolation_noise_grad: Optional[Callable],
    interpolation_grad: Callable,
    weight_fn: Optional[WeightFn],
    metric_fn: Optional[Callable[[Array, Array], Array]],
    axis: int,
    t: Array,
    r: Array,
    x0: Array,
    x1: Array,
    *args,
    rng: Optional[Array] = None,
    loss_mask: Optional[Array] = None,
    **kwargs,
) -> Array:
    """Base function for mean flow matching loss.

    Args:
        model_fn: Function that predicts the velocity field
        interpolation_fn: Function that interpolates between x0 and x1 at time t
        interpolation_noise_fn: Optional function that provides noise scale for interpolation
        interpolation_noise_grad: Optional function that computes gradient of noise scale
        interpolation_grad: Function that computes gradient of interpolation
        weight_fn: Optional function that computes weights based on time
        metric_fn: Optional function that computes the Riemannian metric tensor
        axis: Axis along which to sum the loss
        t: Time values
        r: Starting point
        x0: Starting points
        x1: Ending points
        *args: Additional arguments passed to model_fn
        rng: Random number generator key
        loss_mask: Optional mask for the loss
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Array of loss values
    """
    xt = interpolation_fn(x0, x1, t)
    if interpolation_noise_fn:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt += interpolation_noise_fn(x0, x1, t) * eps

    v_t = interpolation_grad(x0, x1, t)

    if interpolation_noise_fn:
        v_t += interpolation_noise_grad(x0, x1, t) * eps

    u_fn = partial(model_fn, *args, **kwargs)
    u_t, du_dt = jax.jvp(u_fn, (r,t, xt), (0.0, 1.0, v_t))

    v_t = v_t - (t-r) * du_dt
    v_t = jax.lax.stop_gradient(v_t)

    # Compute loss using the metric if provided
    if metric_fn is not None:
        # Get the metric tensor at the current point
        metric = metric_fn(xt, t)
        # Compute the squared norm using the metric
        # v_t and u_t should be vectors in the tangent space
        diff = v_t - u_t
        if len(metric.shape) == 2:
            # If metric is a single matrix, broadcast it
            metric = jnp.expand_dims(metric, 0)
        # Compute (v-u)^T M (v-u) for each point
        loss = jnp.sum(diff * jnp.einsum('...ij,...j->...i', metric, diff), axis=axis)
    else:
        # Euclidean (L2) metric
        loss = jnp.sum((v_t - u_t) ** 2, axis=axis)

    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, 0.0)

    if weight_fn:
        loss = loss * weight_fn(t).reshape(loss.shape)

    return loss

def build_flow_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: Optional[InterpolationNoiseFn] = None,
    weight_fn: Optional[WeightFn] = None,
    axis: int = -1,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a Euclidean flow matching loss function.

    Args:
        model_fn: Function that predicts the velocity field
        interpolation_fn: Function that interpolates between x0 and x1 at time t
        interpolation_noise_fn: Optional function that provides noise scale for interpolation
        weight_fn: Optional function that computes weights based on time
        axis: Axis along which to sum the loss
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time, x0, x1 and returns a scalar loss value
    """
    if interpolation_noise_fn:
        interpolation_noise_grad = jax.grad(
            lambda x_s, x_t, t: interpolation_noise_fn(x_s, x_t, t).sum(), argnums=2
        )
    else:
        interpolation_noise_grad = None

    # For default this is just x1-x0 !
    interpolation_grad = jax.grad(
        lambda x_s, x_t, t: interpolation_fn(x_s, x_t, t).sum(), argnums=2
    )

    def loss_fn(t, x0, x1, *args, rng=None, loss_mask=None, **kwargs):
        """Compute Euclidean flow matching loss.

        Args:
            t: Time values
            x0: Starting points
            x1: Ending points
            *args: Additional arguments passed to model_fn
            rng: Random number generator key
            loss_mask: Optional mask for the loss
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        loss = base_flow_matching_loss(
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad=interpolation_noise_grad,
            interpolation_grad=interpolation_grad,
            weight_fn=weight_fn,
            metric_fn=None,
            axis=axis,
            t=t,
            x0=x0,
            x1=x1,
            *args,
            rng=rng,
            loss_mask=loss_mask,
            **kwargs,
        )
        return reduction_fn(loss)

    return loss_fn


def build_mean_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: InterpolationNoiseFn | None = None,
    weight_fn: WeightFn | None = None,
    axis: int = -1,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a mean flow matching loss function.

    Args:
        model_fn: Function that predicts the velocity field
        interpolation_fn: Function that interpolates between x0 and x1 at time t
        interpolation_noise_fn: Optional function that provides noise scale for interpolation
        weight_fn: Optional function that computes weights based on time
        axis: Axis along which to sum the loss
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time, x0, x1 and returns a scalar loss value
    """
    if interpolation_noise_fn:
        interpolation_noise_grad = jax.grad(
            lambda x_s, x_t, t: interpolation_noise_fn(x_s, x_t, t).sum(), argnums=2
        )
    else:
        interpolation_noise_grad = None
    
    interpolation_grad = jax.grad(
        lambda x_s, x_t, t: interpolation_fn(x_s, x_t, t).sum(), argnums=2
    )

    def loss_fn(r,t, x0, x1, *args, rng=None, loss_mask=None, **kwargs):
        """Compute mean flow matching loss.

        Args:
            t: Time values
            x0: Starting points
            x1: Ending points
            *args: Additional arguments passed to model_fn
            rng: Random number generator key
            loss_mask: Optional mask for the loss
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        loss = base_mean_flow_matching_loss(
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad=interpolation_noise_grad,
            interpolation_grad=interpolation_grad,
            weight_fn=weight_fn,
            metric_fn=None,
            axis=axis,
            t=t,
            r=r,
            x0=x0,
            x1=x1,
            *args,
            rng=rng,
            loss_mask=loss_mask,
            **kwargs,
        )
        return reduction_fn(loss)
    
    return loss_fn

def build_riemannian_flow_matching_loss(
    model_fn: ModelFn | TimeDependentModelFn,
    metric_fn: Callable[[Array, Array], Array],
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: Optional[InterpolationNoiseFn] = None,
    weight_fn: Optional[WeightFn] = None,
    axis: int = -1,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a Riemannian flow matching loss function.

    Args:
        model_fn: Function that predicts the velocity field
        metric_fn: Function that computes the Riemannian metric tensor at a point.
            The function should take (x, t) as input and return a positive definite matrix.
        interpolation_fn: Function that interpolates between x0 and x1 at time t
        interpolation_noise_fn: Optional function that provides noise scale for interpolation
        weight_fn: Optional function that computes weights based on time
        axis: Axis along which to sum the loss
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time, x0, x1 and returns a scalar loss value
    """
    if interpolation_noise_fn:
        interpolation_noise_grad = jax.grad(
            lambda x_s, x_t, t: interpolation_noise_fn(x_s, x_t, t).sum(), argnums=2
        )
    else:
        interpolation_noise_grad = None

    # For default this is just x1-x0 !
    interpolation_grad = jax.grad(
        lambda x_s, x_t, t: interpolation_fn(x_s, x_t, t).sum(), argnums=2
    )

    def loss_fn(t, x0, x1, *args, rng=None, loss_mask=None, **kwargs):
        """Compute Riemannian flow matching loss.

        Args:
            t: Time values
            x0: Starting points
            x1: Ending points
            *args: Additional arguments passed to model_fn
            rng: Random number generator key
            loss_mask: Optional mask for the loss
            **kwargs: Additional keyword arguments passed to model_fn

        Returns:
            Scalar loss value
        """
        loss = base_flow_matching_loss(
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad=interpolation_noise_grad,
            interpolation_grad=interpolation_grad,
            weight_fn=weight_fn,
            metric_fn=metric_fn,
            axis=axis,
            t=t,
            x0=x0,
            x1=x1,
            *args,
            rng=rng,
            loss_mask=loss_mask,
            **kwargs,
        )
        return reduction_fn(loss)

    return loss_fn
