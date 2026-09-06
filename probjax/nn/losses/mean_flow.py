from typing import TYPE_CHECKING, Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

if TYPE_CHECKING:
    from probjax.nn.generative.flow_matching.config import (
        InterpolationScheduleProtocol,
    )


from probjax.nn.losses._flow_shared import (
    _prepare_xt_ut,
    _reduce_diff,
    _validate_time_batch_shapes,
)
from probjax.utils.protocols import (
    LossFn,
    ReductionFn,
    TimeDependentModelFn,
    WeightFn,
)


def base_mean_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
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
    """Base function for mean flow matching loss."""
    xt, u_t = _prepare_xt_ut(schedule, t, x0, x1, rng)

    def v_fn(r, t, x):
        return model_fn(t, x, *args, r=r, **kwargs)

    v_t, dv_dt = jax.jvp(v_fn, (r, t, xt), (jnp.zeros_like(r), jnp.ones_like(t), u_t))

    u_t = u_t - (t - r) * dv_dt
    u_t = jax.lax.stop_gradient(u_t)
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


def build_mean_flow_matching_loss_from_schedule(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    **kwargs,
) -> LossFn:
    """Helper to build mean flow loss directly from an interpolation schedule."""
    return build_mean_flow_matching_loss(
        model_fn=model_fn,
        schedule=schedule,
        **kwargs,
    )


def build_mean_flow_matching_loss(
    model_fn: TimeDependentModelFn,
    schedule: "InterpolationScheduleProtocol",
    weight_fn: WeightFn | None = None,
    metric_fn: Callable[[Array, Array], Array] | None = None,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
    """Build a mean flow matching loss function."""

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
        axis = _validate_time_batch_shapes(x0, x1, t, axis)

        loss = base_mean_flow_matching_loss(
            model_fn,
            schedule,
            weight_fn,
            adaptive_weight_p,
            adaptive_weight_eps,
            metric_fn,
            axis,
            r,
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
