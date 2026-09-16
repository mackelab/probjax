"""Shared building blocks for flow matching and mean flow losses."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array


def _prepare_xt_ut(schedule, t: Array, x0: Array, x1: Array, rng) -> tuple:
    """Interpolate inputs and target velocity, injecting schedule noise."""
    xt = schedule.interpolation_fn(t, x0, x1)

    noise_scale = schedule.interpolation_noise_fn(t, x0, x1)
    if noise_scale is not None:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt = xt + noise_scale * eps

    u_t = schedule.interpolation_velocity_fn(t, x0, x1)

    if noise_scale is not None:
        noise_velocity = schedule.interpolation_noise_velocity_fn(t, x0, x1)
        if noise_velocity is None:
            raise ValueError(
                "interpolation_noise_velocity_fn must be provided when noise is "
                "enabled."
            )
        u_t = u_t + noise_velocity * eps

    return xt, u_t


def _reduce_diff(
    diff: Array,
    metric_fn: Callable[[Array, Array], Array] | None,
    xt: Array,
    t: Array,
    axis: tuple[int, ...],
    adaptive_weight_p: float,
    adaptive_weight_eps: float,
    loss_mask: Optional[Array],
    weight_fn,
) -> Array:
    """Reduce a residual into a loss: metric, adaptive weight, mask, time weight."""
    if metric_fn is not None:
        # Get the metric tensor at the current point
        metric = metric_fn(xt, t)
        # v_t and u_t should be vectors in the tangent space
        if len(metric.shape) == 2:
            # If metric is a single matrix, broadcast it
            metric = jnp.expand_dims(metric, 0)
        # Compute (v-u)^T M (v-u) for each point
        loss = jnp.sum(diff * jnp.einsum("...ij,...j->...i", metric, diff), axis=axis)
    else:
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


def _validate_time_batch_shapes(
    x0: Array, x1: Array, t: Array, axis
) -> tuple[int, ...]:
    """Validate batch/event dims for (t, x0, x1); return normalized axis tuple."""
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
    return axis
