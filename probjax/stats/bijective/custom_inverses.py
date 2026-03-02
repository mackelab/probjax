from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core import custom_inverse


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


@partial(custom_inverse, inv_argnum=1)
def rotate(R: ArrayLike, x: ArrayLike):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


__all__ = [
    "additive_bijector",
    "affine_bijector",
    "rotate",
]
