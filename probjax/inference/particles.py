"""Shared particle helpers for the SMC and filtering stacks.

Both families work on a leading particle axis: they gather particle rows and
normalize log-weights. The helpers live here, at the bottom of the inference
package, so neither family has to import the other.
"""

from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array

T = TypeVar("T")


def take_batch(particles: T, idx: Array) -> T:
    """Gather particle batch rows (pytree-safe)."""
    return jax.tree.map(lambda leaf: leaf[idx], particles)


def normalize_log_weights(log_unnorm: Array) -> Array:
    """Normalize unnormalized log-weights (single ``logsumexp``)."""
    return log_unnorm - jax.scipy.special.logsumexp(log_unnorm)


def nan_to_neg_inf(log_weights: Array) -> Array:
    """Treat NaN log weights as zero weight.

    ``jnp.nan_to_num(x, nan=-inf)`` would also clip ``-inf`` to the most
    negative float, which leaves an impossible particle a finite log weight.
    """
    return jnp.where(jnp.isnan(log_weights), -jnp.inf, log_weights)


def unique_fraction(ancestors: Array) -> Array:
    """Fraction of distinct parents among the resampled particles.

    The ESS measures the weights before resampling, so it cannot see the
    population collapse onto a few parents; this can. ``1 / N`` means every
    particle descends from one.
    """
    ordered = jnp.sort(ancestors)
    return (1 + jnp.sum(ordered[1:] != ordered[:-1])) / ancestors.shape[0]
