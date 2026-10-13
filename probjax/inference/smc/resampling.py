"""Systematic resampling, vendored from unreleased BlackJAX main.

Same API as :func:`blackjax.smc.resampling.systematic`, with the boundary
fixes from ``50818d9c`` (route systematic/stratified queries through the
inverse-CDF helper using ``searchsorted(side="right")`` with queries clamped
strictly below the total mass via ``nextafter``, so float32 boundary queries
can't select zero-weight particles) and ``6f0edc21`` (skip zero-weight
resampling boundaries).

``blackjax.smc.resampling.multinomial``'s ``u=0`` collapse fix (``-log1p(-u)``)
is not vendored: probjax never calls multinomial.

Remove this module and delegate to BlackJAX once the pin reaches a release
containing those fixes.
"""

import jax
import jax.numpy as jnp
from blackjax.types import Array, PRNGKey

__all__ = ["systematic", "resample_systematic"]


def systematic(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    """Systematic resampling.

    Parameters
    ----------
    rng_key: PRNGKey
        PRNGKey to use in resampling.
    weights: Array
        Weights to resample.
    num_samples: int
        Number of particles to sample.

    Returns
    -------
    idx: Array
        Array of size `num_samples` to use for resampling.
    """
    return _systematic_or_stratified(rng_key, weights, num_samples, True)


def resample_systematic(key, log_weights, particles):
    weights = jnp.exp(log_weights)
    idx = systematic(key, weights, log_weights.shape[0])
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return jax.tree.map(lambda x: x[idx], particles), new_log_weights, idx


def _systematic_or_stratified(
    rng_key: PRNGKey, weights: Array, num_samples: int, is_systematic: bool
) -> Array:
    """Helper function for systematic and stratified resampling."""
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    return _inverse_cdf_indices(weights, u, num_samples)


def _inverse_cdf_indices(weights: Array, u: Array, num_samples: int) -> Array:
    """Map explicit systematic or stratified uniforms to particle indices."""
    queries = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    return _inverse_cdf(weights, queries)


def _inverse_cdf(weights: Array, queries: Array) -> Array:
    """Map queries to particle indices via inverse CDF.

    Uses half-open intervals so a zero-weight particle is never selected:
    queries are clipped strictly below the total mass.
    """
    cumsum = jnp.cumsum(weights)
    # Keep queries strictly below the total mass so a rounded-up query cannot
    # be clipped onto a trailing zero-weight particle.
    queries = jnp.minimum(queries, jnp.nextafter(cumsum[-1], -jnp.inf))
    idx = jnp.searchsorted(cumsum, queries, side="right")
    return jnp.clip(idx, 0, weights.shape[0] - 1)
