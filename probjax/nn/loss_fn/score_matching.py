from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, PyTree

from flax import nnx


__all__ = ["build_score_matching_loss", "build_time_dependent_score_matching_loss"]


def base_score_matching_loss(
    model_fn: Callable,
    model_jac_fn: Callable,
    vmap_in_args: Sequence[int],
    axis: int,
    tikhonov: Optional[float],
    *args,
    **kwargs,
):
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
):
    diag_jac = jnp.diagonal(jac_score, axis1=axis - 1, axis2=axis)
    regularizer = tikhonov * jnp.sum(diag_jac**2, axis=axis, keepdims=True)
    return regularizer


def build_score_matching_loss(
    model: nnx.Module | Callable,
    tikhonov: Optional[float] = None,
    reduction_fn: Callable = jnp.mean,
    update_params: Callable = nnx.update,
    jac_fn: Callable = partial(jax.jacfwd, argnums=0),
    axis: int = -1,
):
    model_fn = model
    model_jac_fn = jac_fn(model_fn)

    def loss_fn(params: PyTree, *args, rng=None, **kwargs):
        update_params(model, params)
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
    model: nnx.Module | Callable,
    mean_fn: Callable,
    std_fn: Callable,
    update_params: Callable = nnx.update,
    jac_fn: Callable = partial(jax.jacfwd, argnums=1),
    axis: int = -1,
):
    score_matching_loss = build_score_matching_loss(
        model, update_params=update_params, jac_fn=jac_fn, axis=axis
    )

    def loss_fn(
        params: PyTree, times: Array, xs_target: Array, *args, rng=None, **kwargs
    ):
        assert rng is not None, "loss_fn does require rngs, pass them to function kwargs."
        mean_t = mean_fn(times, xs_target)
        std_t = std_fn(times, xs_target)
        eps = jax.random.normal(rng, shape=xs_target.shape)

        xs_t = mean_t + std_t * eps

        loss = score_matching_loss(params, times, xs_t, *args, **kwargs)

        return loss

    return loss_fn
