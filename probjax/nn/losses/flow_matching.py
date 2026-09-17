from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array

from probjax.nn.losses._flow_shared import _prepare_xt_ut, _reduce_diff, _validate_time_batch_shapes
from probjax.utils.protocols import InterpolationScheduleProtocol

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
    xt, u_t = _prepare_xt_ut(schedule, t, x0, x1, rng)

    v_t = model_fn(t, xt, *args, **kwargs)
    diff = v_t - u_t

    return _reduce_diff(
        diff,
        metric_fn,
        xt,
        t,
        axis,
        adaptive_weight_p,
        adaptive_weight_eps,
        loss_mask,
        weight_fn,
    )


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
        axis = _validate_time_batch_shapes(x0, x1, t, axis)

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
