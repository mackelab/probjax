"""SMC² particle-count calibration and valid exchanges between compiled buckets."""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
from blackjax.smc.base import map_fn
from blackjax.smc.ess import ess

from probjax.inference.smc.temporal_utils import likelihood_diagnostics, replay_filter


class ParticleCountInfo(NamedTuple):
    recommended_count: jax.Array
    log_likelihood_variance: jax.Array
    at_capacity: jax.Array


def recommend_particle_count(
    current_count, log_likelihood_variance, buckets, *, target_variance=1.0
):
    """Choose a nondecreasing capacity bucket using the approximate variance ~ 1/N law.

    This is a cost heuristic, not a guarantee of mixing. Call likelihood_diagnostics
    at fixed parameters; variance across different theta values is not estimator noise.
    """
    if target_variance <= 0 or not buckets or any(a <= 0 for a in buckets):
        raise ValueError('Supply positive buckets and a positive target variance.')
    if tuple(sorted(set(buckets))) != tuple(buckets):
        raise ValueError('Buckets must be strictly increasing.')
    sizes = jnp.asarray(buckets)
    required = current_count * jnp.maximum(
        log_likelihood_variance / target_variance, 1.0
    )
    required = jnp.where(jnp.isfinite(required), required, sizes[-1] + 1)
    index = jnp.minimum(jnp.searchsorted(sizes, required), len(buckets) - 1)
    return ParticleCountInfo(
        sizes[index], log_likelihood_variance, required > sizes[-1]
    )


class ParticleExchangeInfo(NamedTuple):
    log_evidence_increment: jax.Array
    ess: jax.Array
    valid: jax.Array


def exchange_filter_population(key, state, new_backend, replay, *, batch_size=0):
    """Importance-exchange an SMC² population to a different inner particle count.

    Fresh filters are replayed under q_new. The extended-target correction is
    L_new/L_old; replacing filters without this correction changes the target.
    This function is JIT-compatible for each fixed old/new capacity pair. A host
    controller selects a cached compiled pair when a change in array shape is needed.
    No proposal state or user parameter values are discarded. No resampling occurs.
    The normalizing correction is included in the evidence estimate.
    """
    if new_backend.likelihood_kind != 'unbiased':
        raise ValueError(
            'Particle-count exchanges require an unbiased particle backend.'
        )
    n = state.log_weights.shape[0]
    if replay.ts.shape[0] == 0:
        raise ValueError('ReplayData must contain the assimilated prefix.')
    filters, logz, _ = map_fn(
        lambda args: replay_filter(
            new_backend, args[0], args[1], state.t0, replay, state.num_observations
        ),
        batch_size,
    )((jax.random.split(key, n), state.parameters))
    raw = state.log_weights + logz - state.log_likelihoods
    increment = jax.scipy.special.logsumexp(raw)
    prefix_ok = (state.num_observations <= replay.ts.shape[0]) & jnp.where(
        state.num_observations == 0,
        True,
        replay.ts[state.num_observations - 1] == state.t,
    )
    valid = jnp.isfinite(increment) & prefix_ok
    # Output has the new shape even on failure; check valid before committing.
    weights = jnp.where(valid, raw - increment, -jnp.inf)
    return state._replace(
        filter_states=filters,
        log_likelihoods=logz,
        log_weights=weights,
        log_evidence=state.log_evidence + increment,
    ), ParticleExchangeInfo(increment, jnp.where(valid, ess(weights), 0.0), valid)


class AdaptiveParticleResult(NamedTuple):
    state: object
    particle_count: int
    recommendation: ParticleCountInfo
    exchange_info: object
    key: object


def adapt_particle_count(
    key,
    state,
    backend_factory,
    current_count,
    replay,
    *,
    buckets=(64, 128, 256, 512),
    target_variance=1.0,
    num_parameters=4,
    num_replicates=8,
    batch_size=0,
):
    """Host-side automatic bucket selection; all expensive operations are compiled.

    backend_factory(N) supplies fixed-capacity particle filters. Diagnostics use
    independent runs at a weighted sample of parameter values, then the maximum
    estimated log-likelihood variance across these values. Changing capacity is
    explicit because a single JIT carry cannot change shape. For manual compiled
    control use likelihood_diagnostics, recommend_particle_count, and
    exchange_filter_population.
    An exchange failure raises and never returns an apparently usable population.
    """
    if current_count not in buckets or num_parameters < 1:
        raise ValueError(
            'current_count must be in buckets and num_parameters positive.'
        )
    if isinstance(state.num_observations, jax.core.Tracer):
        raise ValueError(
            'adapt_particle_count selects shapes on the host; '
            'use its compiled primitives inside JIT.'
        )
    key, select_key, diagnostic_key, exchange_key = jax.random.split(key, 4)
    indices = jax.random.categorical(
        select_key, state.log_weights, shape=(num_parameters,)
    )
    parameters = jax.tree.map(lambda x: x[indices], state.parameters)
    estimate = _diagnostic_kernel(
        backend_factory, current_count, num_replicates, batch_size
    )
    variance = estimate(
        jax.random.split(diagnostic_key, num_parameters),
        parameters,
        state.t0,
        replay,
        state.num_observations,
    ).max()
    recommendation = recommend_particle_count(
        current_count, variance, buckets, target_variance=target_variance
    )
    selected = int(recommendation.recommended_count)
    info = None
    if selected != current_count:
        state, info = _exchange_kernel(backend_factory, selected, batch_size)(
            exchange_key, state, replay
        )
        if not bool(info.valid):
            raise ValueError(
                'Particle-count exchange failed; retain the previous population.'
            )
    return AdaptiveParticleResult(state, selected, recommendation, info, key)


@lru_cache(maxsize=32)
def _diagnostic_kernel(factory, count, num_replicates, batch_size):
    backend = factory(count)
    return jax.jit(
        lambda keys, parameters, t0, data, n: jax.vmap(
            lambda key, p: (
                likelihood_diagnostics(
                    backend,
                    key,
                    p,
                    t0,
                    data,
                    n,
                    num_replicates=num_replicates,
                    batch_size=batch_size,
                ).log_variance
            )
        )(keys, parameters)
    )


@lru_cache(maxsize=32)
def _exchange_kernel(factory, count, batch_size):
    backend = factory(count)
    return jax.jit(
        lambda key, state, data: exchange_filter_population(
            key, state, backend, data, batch_size=batch_size
        )
    )
