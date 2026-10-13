"""Shared particle helpers for reverse-time twisted SMC runners."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, PyTree, RngKey


def take_batch(particles: PyTree[Array], idx: Array) -> PyTree[Array]:
    """Gather particle batch rows (pytree-safe)."""
    return jax.tree.map(lambda leaf: leaf[idx], particles)


def random_like(key: RngKey, template: PyTree[Array]) -> PyTree[Array]:
    """Standard-normal noise matching a particle batch pytree."""
    leaves, treedef = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    noise = [
        jax.random.normal(k, leaf.shape, getattr(leaf, "dtype", jnp.float32))
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return treedef.unflatten(noise)


def log_weighted_mean_exp(log_weights: Array, log_increments: Array) -> Array:
    """``log E_w[exp(d)]`` for normalized log-weights ``log_weights``."""
    return jax.scipy.special.logsumexp(log_weights + log_increments)


def normalize_log_weights(log_unnorm: Array) -> Array:
    """Normalize unnormalized log-weights (single ``logsumexp``)."""
    return log_unnorm - jax.scipy.special.logsumexp(log_unnorm)


__all__ = [
    "take_batch",
    "random_like",
    "log_weighted_mean_exp",
    "normalize_log_weights",
]
