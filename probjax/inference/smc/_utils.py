"""Shared particle helpers for reverse-time twisted SMC runners.

The generic particle helpers (``take_batch``, ``normalize_log_weights``) live
in :mod:`probjax.inference.particles`, which the filtering stack also uses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, PyTree


def random_like(rng, template: PyTree[Array]) -> PyTree[Array]:
    """Standard-normal noise matching a particle batch pytree."""
    leaves, treedef = jax.tree.flatten(template)
    keys = jax.random.split(rng, len(leaves))
    noise = [
        jax.random.normal(k, leaf.shape, getattr(leaf, "dtype", jnp.float32))
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return treedef.unflatten(noise)


def log_weighted_mean_exp(log_weights: Array, log_increments: Array) -> Array:
    """``log E_w[exp(d)]`` for normalized log-weights ``log_weights``."""
    return jax.scipy.special.logsumexp(log_weights + log_increments)


__all__ = [
    "random_like",
    "log_weighted_mean_exp",
]
