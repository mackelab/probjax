"""Shared particle helpers for the SMC and filtering stacks.

Both families work on a leading particle axis: they gather particle rows and
normalize log-weights. The helpers live here, at the bottom of the inference
package, so neither family has to import the other.
"""

from __future__ import annotations

from typing import TypeVar

import jax

from probjax.utils.typing import Array

T = TypeVar("T")


def take_batch(particles: T, idx: Array) -> T:
    """Gather particle batch rows (pytree-safe)."""
    return jax.tree.map(lambda leaf: leaf[idx], particles)


def normalize_log_weights(log_unnorm: Array) -> Array:
    """Normalize unnormalized log-weights (single ``logsumexp``)."""
    return log_unnorm - jax.scipy.special.logsumexp(log_unnorm)
