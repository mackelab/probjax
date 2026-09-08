"""Shared replay, population statistics and diagnostics for temporal SMC."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.smc.base import map_fn
from blackjax.smc.ess import ess
from jax.flatten_util import ravel_pytree


def replay_filter(backend, key, theta, t0, data, count, *, differentiable=False):
    """Replay a prefix, returning terminal state, log likelihood, and prefix-minus-one.

    Dynamic loops avoid visiting future data. A masked static scan enables reverse
    mode for exact-likelihood MCMC; the ignored suffix never contributes a density.
    """
    key, init_key = jax.random.split(key)
    initial = backend.init(init_key, theta, t0)
    zero = jnp.array(0.0, dtype=jnp.result_type(jax.tree.leaves(initial)[0], 0.0))

    def step(i, carry):
        key, state, logz, _ = carry
        key, step_key = jax.random.split(key)
        state, info = backend.step(
            step_key, state, theta, data.ts[i], data.observations[i], data.observed[i]
        )
        return key, state, logz + info.log_likelihood, logz

    carry = (key, initial, zero, zero)
    if differentiable:

        def masked(carry, i):
            return jax.lax.cond(i < count, lambda: step(i, carry), lambda: carry), None

        carry, _ = jax.lax.scan(masked, carry, jnp.arange(data.ts.shape[0]))
    else:
        carry = jax.lax.fori_loop(0, count, step, carry)
    _, state, logz, previous = carry
    return state, logz, previous


def population_matrix(particles):
    return jax.vmap(lambda p: ravel_pytree(p)[0])(particles)


def population_covariance(particles, log_weights, ridge=1e-5):
    x = population_matrix(particles)
    weights = jax.nn.softmax(log_weights)
    centered = x - jnp.sum(weights[:, None] * x, axis=0)
    cov = (centered.T * weights) @ centered
    return cov + ridge * jnp.eye(x.shape[1], dtype=x.dtype)


class PopulationDiagnostics(NamedTuple):
    ess: jax.Array
    unique_particles: jax.Array
    unique_ancestors: jax.Array
    max_weight: jax.Array


def population_diagnostics(particles, log_weights, ancestors=None):
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


class LikelihoodDiagnostics(NamedTuple):
    log_likelihoods: jax.Array
    log_variance: jax.Array
    log_mean_likelihood: jax.Array


def likelihood_diagnostics(
    backend, key, theta, t0, data, count=None, *, num_replicates=8, batch_size=0
):
    """Replicate likelihood estimates at one fixed theta, not across theta values."""
    if num_replicates < 2:
        raise ValueError("At least two likelihood replicates are required.")
    count = data.ts.shape[0] if count is None else count
    estimates = map_fn(
        lambda k: replay_filter(backend, k, theta, t0, data, count)[1], batch_size
    )(jax.random.split(key, num_replicates))
    return LikelihoodDiagnostics(
        estimates,
        jnp.var(estimates, ddof=1),
        jax.scipy.special.logsumexp(estimates) - jnp.log(num_replicates),
    )


class ReplicateDiagnostics(NamedTuple):
    mean: jax.Array
    standard_error: jax.Array
    num_replicates: jax.Array


def summarize_replicates(estimates):
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
