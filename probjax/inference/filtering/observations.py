"""Observation-grid validation for temporal filtering.

Leaf module: only jax imports. Both ``filtering.temporal`` and ``smc.temporal``
validate their ``(ts, observations, observed)`` triples here instead of
reaching into each other's privates.
"""

import jax.numpy as jnp


def validate_observations(ts, observations, observed):
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
