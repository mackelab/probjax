"""Prefix replay, population statistics and diagnostics for temporal SMC."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from probjax.inference.smc.ess import ess
from probjax.utils.typing import Array


def replay_filter(
    filt: Any, key: Any, theta: Any, t0: Any, data: Any, count: int
) -> tuple[Any, Array]:
    """Filter the first ``count`` observations of ``data = (ts, ys, observed)``.

    Returns the final filter state and the log likelihood. The loop stops at
    ``count``, so later observations in ``data`` are never read.
    """
    ts, observations, observed = data
    key, init_key = jax.random.split(key)
    initial = filt.init(init_key, theta, t0)
    zero = jnp.array(0.0, dtype=jnp.result_type(jax.tree.leaves(initial)[0], 0.0))

    def step(i, carry):
        key, state, logz = carry
        key, step_key = jax.random.split(key)
        state, info = filt.step(
            step_key, state, theta, ts[i], observations[i], observed[i]
        )
        return key, state, logz + info.log_likelihood

    _, state, logz = jax.lax.fori_loop(0, count, step, (key, initial, zero))
    return state, logz


def population_matrix(particles: Any) -> Array:
    return jax.vmap(lambda p: ravel_pytree(p)[0])(particles)


def population_covariance(
    particles: Any, log_weights: Array, ridge: float = 1e-5
) -> Array:
    x = population_matrix(particles)
    weights = jax.nn.softmax(log_weights)
    centered = x - jnp.sum(weights[:, None] * x, axis=0)
    cov = (centered.T * weights) @ centered
    return cov + ridge * jnp.eye(x.shape[1], dtype=x.dtype)


class PopulationDiagnostics(NamedTuple):
    ess: Array
    unique_particles: Array
    unique_ancestors: Array
    max_weight: Array


def population_diagnostics(
    particles: Any, log_weights: Array, ancestors: Array | None = None
) -> PopulationDiagnostics:
    """Weight ESS and exact population/ancestral diversity (O(N log N) sorting).

    Diagnostics do not imply independent posterior samples. ancestors may be
    composed origin labels for measuring collapse over many resampling steps.
    """
    x = population_matrix(particles)
    order = jnp.lexsort(x.T[::-1])
    unique = 1 + jnp.sum(jnp.any(jnp.diff(x[order], axis=0) != 0, axis=1))
    if ancestors is None:
        unique_ancestors = jnp.array(-1)  # unavailable
    else:
        labels = jnp.sort(ancestors)
        unique_ancestors = 1 + jnp.sum(jnp.diff(labels) != 0)
    return PopulationDiagnostics(
        ess(log_weights), unique, unique_ancestors, jax.nn.softmax(log_weights).max()
    )


class ReplicateDiagnostics(NamedTuple):
    mean: Array
    standard_error: Array
    num_replicates: Array


def summarize_replicates(estimates: Any) -> ReplicateDiagnostics:
    """Mean and Monte Carlo standard error across independent RUN estimates.

    Supply posterior expectation estimates (or evidence estimates on the desired
    scale), not individual correlated particles from one run.
    """
    values = jnp.asarray(estimates)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("At least two independent run estimates are required.")
    return ReplicateDiagnostics(
        values.mean(0),
        values.std(0, ddof=1) / jnp.sqrt(values.shape[0]),
        jnp.array(values.shape[0]),
    )
