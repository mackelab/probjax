"""Shared helpers for the spherical distributions (Bingham, Watson).

The Bingham and Watson distributions both work on the unit sphere and
share the small geometric primitives: projecting onto the sphere, the
area of the sphere, and sampling from it.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

from probjax.utils.typing import Array, ArrayLike

__all__ = [
    "normalize_vector",
    "sphere_area",
    "sample_sphere",
    "sample_uniform_sphere",
]

#: Guards a division by a vanishing norm.
EPS = 1e-8


def normalize_vector(x: ArrayLike) -> Array:
    """Project vectors onto the unit sphere."""
    x_arr = jnp.asarray(x)
    norm = jnp.linalg.norm(x_arr, axis=-1, keepdims=True)
    return x_arr / (norm + EPS)


def sphere_area(dim: int) -> Array:
    """The surface area of the unit sphere in ``dim`` dimensions."""
    dim_arr = jnp.asarray(dim, dtype=jnp.float32)
    return jnp.exp(
        jnp.log(2.0) + 0.5 * dim_arr * jnp.log(jnp.pi) - gammaln(0.5 * dim_arr)
    )


def sample_sphere(rng, n_samples: int, dtype=jnp.float32) -> Array:
    """Sample ``n_samples`` random unit vectors on the 2-sphere."""
    xyz = random.normal(rng, shape=(n_samples, 3), dtype=dtype)
    norms = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    return xyz / (norms + EPS)


def sample_uniform_sphere(rng, dim: int) -> Array:
    """Sample one uniformly distributed unit vector in ``dim`` dimensions."""
    vec = random.normal(rng, shape=(dim,))
    return vec / (jnp.linalg.norm(vec) + EPS)
