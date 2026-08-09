"""Affine bijections in natural parameterization.

Like every module in this package these take ``x`` first and already-constrained
natural parameters afterwards; the caller (typically a bijector config in
:mod:`probjax.nn.generative.nflows.config`) is responsible for mapping an
unconstrained parameter vector onto them.

:func:`shift` and :func:`affine` are exactly invertible by the jaxpr inverse
interpreter, so they need no :class:`~probjax.core.custom_inverse` registration.
:func:`rotate` does, because the interpreter cannot know that ``R`` is orthogonal
(and hence that the inverse is ``R.T`` with zero log-determinant).
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core import custom_inverse

__all__ = [
    "affine",
    "rotate",
    "shift",
]


def shift(x: ArrayLike, loc: ArrayLike):
    """``y = x + loc``."""
    x = jnp.asarray(x)
    return x + jnp.asarray(loc).reshape(x.shape)


def affine(x: ArrayLike, loc: ArrayLike, scale: ArrayLike):
    """``y = loc + scale * x`` for strictly positive ``scale``."""
    x = jnp.asarray(x)
    loc = jnp.asarray(loc).reshape(x.shape)
    scale = jnp.asarray(scale).reshape(x.shape)
    return loc + scale * x


@partial(custom_inverse, inv_argnum=0)
def rotate(x: ArrayLike, R: ArrayLike):
    """``y = R @ x`` for an orthogonal ``R``."""
    return jnp.matmul(R, jnp.asarray(x).T).T


rotate.definv_and_logdet(lambda y, R: (jnp.matmul(R.T, jnp.asarray(y).T).T, 0.0))
