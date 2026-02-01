from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.utils.protocols import (
    LossFn,
    TimeDependentModelFn,
    WeightFn,
)


def base_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    weight_fn: WeightFn | None,
    adaptive_weight_p: float,
    adaptive_weight_eps: float,
    metric_fn: Callable[[Array, Array], Array] | None,
    axis: tuple[int, ...],
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
        schedule: Interpolation schedule providing path and velocity functions
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
    xt = schedule.interpolation_fn(t, x0, x1)

    noise_scale = schedule.interpolation_noise_fn(t, x0, x1)
    if noise_scale is not None:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt += noise_scale * eps

    v_t = model_fn(t, xt, *args, **kwargs)
    u_t = schedule.interpolation_velocity_fn(t, x0, x1)

    if noise_scale is not None:
        noise_velocity = schedule.interpolation_noise_velocity_fn(t, x0, x1)
        if noise_velocity is None:
            raise ValueError(
                "interpolation_noise_velocity_fn must be provided when noise is enabled."
            )
        u_t = u_t + noise_velocity * eps

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
        diff = v_t - u_t
        # Euclidean (L2) metric
        loss = jnp.sum(diff**2, axis=axis)

    if adaptive_weight_p > 0:
        weight = jax.lax.stop_gradient(
            1 / (jnp.sum(diff**2, axis=axis) + adaptive_weight_eps) ** adaptive_weight_p
        )
        loss = loss * weight

    if loss_mask is not None:
        loss = jnp.where(~loss_mask, loss, 0.0)

    if weight_fn:
        loss = loss * weight_fn(t).reshape(loss.shape)

    return loss


def build_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    weight_fn: Optional[WeightFn] = None,
    metric_fn: Callable[[Array, Array], Array] | None = None,
    reduction_fn: Callable = jnp.mean,
) -> LossFn:
    """Build a Euclidean flow matching loss function.

    Args:
        model_fn: Function that predicts the velocity field
        schedule: Interpolation schedule with explicit velocity functions
        weight_fn: Optional function that computes weights based on time
        metric_fn: Optional function that computes the Riemannian metric tensor
        axis: Axis along which to sum the loss
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time, x0, x1 and returns a scalar loss value
    """

    def loss_fn(
        t: Array,
        x0: Array,
        x1: Array,
        *args,
        rng: Optional[Array] = None,
        loss_mask: Optional[Array] = None,
        axis: int | tuple[int, ...] = -1,
        adaptive_weight_p: float = 0.0,
        adaptive_weight_eps: float = 1e-3,
        **kwargs,
    ):
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
        axis = axis if isinstance(axis, tuple) else (axis,)
        event_dims = len(axis)
        if x0.ndim > 1 + event_dims:
            raise ValueError(
                "x0 must have at most 1 batch dim + event_dims (len(axis)) dimensions"
            )
        if x1.ndim > 1 + event_dims:
            raise ValueError(
                "x1 must have at most 1 batch dim + event_dims (len(axis)) dimensions"
            )
        if t.ndim > 1 + event_dims and all(t.shape[i] == 1 for i in range(1, t.ndim)):
            raise ValueError(
                "t must have at most 1 batch dim + event_dims (len(axis)) dimensions"
            )

        # Compute the loss
        loss = base_flow_matching_loss(
            model_fn,
            schedule,
            weight_fn,
            adaptive_weight_p,
            adaptive_weight_eps,
            metric_fn,
            axis,
            t,
            x0,
            x1,
            *args,
            rng=rng,
            loss_mask=loss_mask,
            **kwargs,
        )
        return reduction_fn(loss)

    return loss_fn
