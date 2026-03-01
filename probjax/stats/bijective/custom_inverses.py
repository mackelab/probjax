from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core import custom_inverse


@partial(custom_inverse, inv_argnum=1)
def affine_bijector(
    params: ArrayLike,
    x: ArrayLike,
    min_scale: float = 1e-3,
    **kwargs,
):
    del kwargs
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    loc, scale = loc.reshape(x.shape), scale.reshape(x.shape)
    scale = jnp.exp(scale) + min_scale
    return loc + scale * x


def _inv_and_logdet_affine_bijector(
    params: ArrayLike,
    y: ArrayLike,
    min_scale: float = 1e-3,
    **kwargs,
):
    del kwargs
    y = jnp.asarray(y)
    loc, scale = jnp.split(params, 2, axis=-1)
    loc, scale = loc.reshape(y.shape), scale.reshape(y.shape)
    scale = jnp.exp(scale) + min_scale
    x = (y - loc) / scale
    logdet = -jnp.sum(jnp.log(jnp.abs(scale)), axis=-1)
    return x, logdet


affine_bijector.definv_and_logdet(_inv_and_logdet_affine_bijector)


@partial(custom_inverse, inv_argnum=1)
def additive_bijector(
    params: ArrayLike,
    x: ArrayLike,
    **kwargs,
):
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    params = params.reshape(x.shape)
    return x + params


def _inv_and_logdet_additive_bijector(
    params: ArrayLike,
    y: ArrayLike,
    **kwargs,
):
    del kwargs
    y = jnp.asarray(y)
    params = jnp.asarray(params).reshape(y.shape)
    x = y - params
    logdet = jnp.zeros(y.shape[:-1], dtype=y.dtype)
    return x, logdet


additive_bijector.definv_and_logdet(_inv_and_logdet_additive_bijector)


@partial(custom_inverse, inv_argnum=1)
def rotate(R: ArrayLike, x: ArrayLike):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


__all__ = [
    "additive_bijector",
    "affine_bijector",
    "rotate",
]
