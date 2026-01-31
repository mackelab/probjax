from typing import TYPE_CHECKING, Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.utils.protocols import (
    InterpolationFn,
    InterpolationNoiseFn,
    LossFn,
    ReductionFn,
    TimeDependentModelFn,
    WeightFn,
)

if TYPE_CHECKING:
    from probjax.nn.nets.flow_matching_configs import InterpolationScheduleProtocol

def base_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    interpolation_fn: InterpolationFn,
    interpolation_noise_fn: InterpolationNoiseFn | None,
    interpolation_grad_fn: InterpolationNoiseFn,
    interpolation_noise_grad_fn: InterpolationNoiseFn | None,
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
        interpolation_fn: Function ``interpolation_fn(t, x0, x1)`` describing the path
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
    xt = interpolation_fn(t, x0, x1)

    if interpolation_noise_fn:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt += interpolation_noise_fn(t, x0, x1) * eps

    v_t = model_fn(t, xt, *args, **kwargs)
    u_t = jax.vmap(interpolation_grad_fn)(t, x0, x1).reshape(v_t.shape)

    if interpolation_noise_fn:
        u_t += interpolation_noise_grad_fn(t, x0, x1).reshape(v_t.shape) * eps

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


def base_mean_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    interpolation_fn: InterpolationFn,
    interpolation_noise_fn: InterpolationNoiseFn | None,
    interpolation_grad_fn: InterpolationNoiseFn,
    interpolation_noise_grad_fn: InterpolationNoiseFn | None,
    weight_fn: WeightFn | None,
    adaptive_weight_p: float,
    adaptive_weight_eps: float,
    metric_fn: Callable[[Array, Array], Array] | None,
    axis: tuple[int, ...],
    r: Array,
    t: Array,
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
        interpolation_fn: Function ``interpolation_fn(t, x0, x1)`` describing the path
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
    xt = interpolation_fn(t, x0, x1)
    if interpolation_noise_fn:
        assert rng is not None, "rng is required when using interpolation_noise_fn"
        eps = jax.random.normal(rng, shape=xt.shape)
        xt += interpolation_noise_fn(t, x0, x1) * eps

    u_t = jax.vmap(interpolation_grad_fn)(t, x0, x1).reshape(xt.shape)

    if interpolation_noise_fn:
        u_t += interpolation_noise_grad_fn(t, x0, x1).reshape(eps.shape) * eps

    def v_fn(r, t, x):
        return model_fn(t, x, *args, r=r, **kwargs)

    v_t, dv_dt = jax.jvp(v_fn, (r, t, xt), (jnp.zeros_like(r), jnp.ones_like(t), u_t))

    u_t = u_t - (t - r) * dv_dt
    u_t = jax.lax.stop_gradient(u_t)
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
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: Optional[InterpolationNoiseFn] = None,
    interpolation_noise_grad_fn: Optional[Callable] = None,
    interpolation_grad_fn: Optional[Callable] = None,
    weight_fn: Optional[WeightFn] = None,
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

    # For default this is just x1-x0 !
    if interpolation_grad_fn is None:
        interpolation_grad_fn = jax.jacfwd(
            lambda t, x_s, x_t: interpolation_fn(t, x_s, x_t), argnums=0
        )
    else:
        interpolation_grad_fn = interpolation_grad_fn

    if interpolation_noise_fn and interpolation_noise_grad_fn is None:
        interpolation_noise_grad_fn = jax.jacfwd(
            lambda t, x_s, x_t: interpolation_noise_fn(t, x_s, x_t), argnums=0
        )

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
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad_fn=interpolation_noise_grad_fn,
            interpolation_grad_fn=interpolation_grad_fn,
            weight_fn=weight_fn,
            adaptive_weight_p=adaptive_weight_p,
            adaptive_weight_eps=adaptive_weight_eps,
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


def build_flow_matching_loss_from_schedule(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    **kwargs,
) -> LossFn:
    """Helper to build loss directly from an interpolation schedule."""
    return build_flow_matching_loss(
        model_fn=model_fn,
        interpolation_fn=schedule.interpolation_fn,
        interpolation_noise_fn=schedule.interpolation_noise_fn,
        **kwargs,
    )


def build_mean_flow_matching_loss_from_schedule(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    **kwargs,
) -> LossFn:
    """Helper to build mean flow loss directly from an interpolation schedule."""
    return build_mean_flow_matching_loss(
        model_fn=model_fn,
        interpolation_fn=schedule.interpolation_fn,
        interpolation_noise_fn=schedule.interpolation_noise_fn,
        **kwargs,
    )

def build_mean_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: InterpolationNoiseFn | None = None,
    interpolation_grad_fn: InterpolationFn | None = None,
    interpolation_noise_grad_fn: InterpolationNoiseFn | None = None,
    weight_fn: WeightFn | None = None,
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

    # For default this is just x1-x0 !
    if interpolation_grad_fn is None:
        interpolation_grad_fn = jax.jacfwd(
            lambda t, x_s, x_t: interpolation_fn(t, x_s, x_t), argnums=0
        )
    else:
        pass

    if interpolation_noise_fn and interpolation_noise_grad_fn is None:
        interpolation_noise_grad_fn = jax.jacfwd(
            lambda t, x_s, x_t: interpolation_noise_fn(t, x_s, x_t), argnums=0
        )

    def loss_fn(
        r,
        t,
        x0,
        x1,
        *args,
        rng=None,
        loss_mask=None,
        adaptive_weight_p: float = 0.0,
        adaptive_weight_eps: float = 1e-3,
        axis=-1,
        **kwargs,
    ):
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

        event_dims = 1 if isinstance(axis, int) else len(axis)
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

        loss = base_mean_flow_matching_loss(
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad_fn=interpolation_noise_grad_fn,
            interpolation_grad_fn=interpolation_grad_fn,
            weight_fn=weight_fn,
            adaptive_weight_p=adaptive_weight_p,
            adaptive_weight_eps=adaptive_weight_eps,
            metric_fn=None,
            axis=axis,
            r=r,
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


def build_riemannian_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    metric_fn: Callable[[Array, Array], Array],
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: Optional[InterpolationNoiseFn] = None,
    interpolation_noise_grad_fn: Optional[Callable] = None,
    interpolation_grad_fn: Optional[Callable] = None,
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
        interpolation_noise_grad_fn: Optional function that computes gradient of noise scale
        interpolation_grad_fn: Optional function that computes gradient of interpolation
        weight_fn: Optional function that computes weights based on time
        axis: Axis along which to sum the loss
        reduction_fn: Function to reduce the loss to a scalar

    Returns:
        A loss function that takes time, x0, x1 and returns a scalar loss value
    """

    # For default this is just x1-x0 !
    if interpolation_grad_fn is None:
        interpolation_grad_fn = jax.jacfwd(
            lambda t, x_s, x_t: interpolation_fn(t, x_s, x_t), argnums=0
        )

    if interpolation_noise_fn:
        if interpolation_noise_grad_fn is None:
            interpolation_noise_grad_fn = jax.jacfwd(
                lambda t, x_s, x_t: interpolation_noise_fn(t, x_s, x_t), argnums=0
            )

    def loss_fn(
        t: Array,
        x0: Array,
        x1: Array,
        *args,
        rng: Optional[Array] = None,
        loss_mask: Optional[Array] = None,
        **kwargs,
    ):
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
        event_dims = 1 if isinstance(axis, int) else len(axis)
        # Flatten the batch dimension
        x0 = jnp.reshape(x0, (-1, *x0.shape[event_dims:]))
        x1 = jnp.reshape(x1, (-1, *x1.shape[event_dims:]))
        t = jnp.reshape(t, (-1, *t.shape[event_dims:]))

        loss = base_flow_matching_loss(
            model_fn=model_fn,
            interpolation_fn=interpolation_fn,
            interpolation_noise_fn=interpolation_noise_fn,
            interpolation_noise_grad_fn=interpolation_noise_grad_fn,
            interpolation_grad_fn=interpolation_grad_fn,
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
