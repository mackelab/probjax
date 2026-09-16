from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.nn.losses._flow_shared import _prepare_xt_ut, _reduce_diff, _validate_time_batch_shapes
from probjax.utils.protocols import InterpolationScheduleProtocol

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
    imf: bool = True,
    **kwargs,
) -> Array:
    """Base function for mean flow matching loss.

    With ``imf=True`` (default) the Improved MeanFlow objective
    (arXiv:2512.02012) is used: a v-loss ``||V - u_t||^2`` with
    ``V = v_t + (t - r) * sg(dv_dt)`` where the JVP tangent is the
    network's own boundary-condition velocity ``v_bc = model_fn(t, xt,
    r=t)``. This estimates the marginal (not conditional) velocity, so
    the tangent carries much less variance, and the regression target
    ``u_t`` no longer depends on the network. Same optimum as the
    original objective, more stable optimization. With ``imf=False``
    the original MeanFlow objective is used instead.
    """
    xt, u_t = _prepare_xt_ut(schedule, t, x0, x1, rng)

    def v_fn(r, t, x):
        return model_fn(t, x, *args, r=r, **kwargs)

    if imf:
        tangent = jax.lax.stop_gradient(model_fn(t, xt, *args, r=t, **kwargs))
    else:
        tangent = u_t

    v_t, dv_dt = jax.jvp(
        v_fn, (r, t, xt), (jnp.zeros_like(r), jnp.ones_like(t), tangent)
    )

    if imf:
        pred = v_t + (t - r) * jax.lax.stop_gradient(dv_dt)
        target = u_t
    else:
        pred = v_t
        target = jax.lax.stop_gradient(u_t - (t - r) * dv_dt)

    diff = pred - target

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
    imf: bool = True,
) -> LossFn:
    """Build a mean flow matching loss function.

    Args:
        imf: Use the Improved MeanFlow v-loss objective (default) instead
            of the original MeanFlow u-loss. See
            :func:`base_mean_flow_matching_loss`.
    """
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
            imf=imf,
            **kwargs,
        )
        return reduction_fn(loss)

    return loss_fn
