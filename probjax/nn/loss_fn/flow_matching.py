from typing import Callable, Optional, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from probjax.nn.loss_fn.protocols import (
    InterpolationFn,
    InterpolationNoiseFn,
    LossFn,
    ReductionFn,
    VelocityModelFn,
    WeightFn,
)


# Flow matching objectives
def build_flow_matching_loss(
    model: VelocityModelFn,
    interpolation_fn: InterpolationFn = lambda t, x0, x1: (1 - t) * x0 + t * x1,
    interpolation_noise_fn: Optional[InterpolationNoiseFn] = None,
    weight_fn: Optional[WeightFn] = None,
    axis: int = -1,
    reduction_fn: ReductionFn = jnp.mean,
) -> LossFn:
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
        xt = interpolation_fn(x0, x1, t)
        if interpolation_noise_fn:
            assert (
                rng is not None
            ), "loss_fn does require rngs, pass them to function kwargs."
            eps = jax.random.normal(rng, shape=xt.shape)
            xt += interpolation_noise_fn(x0, x1, t) * eps

        v_t = model(t, xt, *args, **kwargs)
        u_t = interpolation_grad(x0, x1, t)

        if interpolation_noise_fn:
            u_t += interpolation_noise_grad(x0, x1, t) * eps

        loss = jnp.sum((v_t - u_t) ** 2, axis=axis)
        if loss_mask is not None:
            loss = jnp.where(~loss_mask, loss, jnp.zeros_like(loss))

        if weight_fn:
            loss = loss * weight_fn(t).reshape(loss.shape)

        return reduction_fn(loss)

    return loss_fn
