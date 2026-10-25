"""Observation-grid validation for temporal filtering.

Leaf module: only jax imports. :func:`particle_gibbs` validates its
``(ts, observations, observed)`` triple here.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def validate_observations(
    ts: Any, observations: Any, observed: Any
) -> tuple[Any, Any, Any]:
    """Coerce and validate ``(ts, observations, observed)``.

    Returns ``(ts, observations, observed)`` with ``ts`` a float 1-D array,
    ``observations`` an array sharing the leading time dimension, and
    ``observed`` a boolean mask over time (all-True when None).
    """
    ts = jnp.asarray(ts)
    ts = ts.astype(jnp.result_type(ts, 0.0))
    observations = jnp.asarray(observations)
    if ts.ndim != 1 or observations.ndim < 1 or observations.shape[0] != ts.shape[0]:
        raise ValueError("ts and observations must share their leading time dimension.")
    observed = (
        jnp.ones(ts.shape, dtype=bool) if observed is None else jnp.asarray(observed)
    )
    if observed.shape != ts.shape or observed.dtype != jnp.bool_:
        raise ValueError("observed must be a boolean mask with the same shape as ts.")
    return ts, observations, observed
