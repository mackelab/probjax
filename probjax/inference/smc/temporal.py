"""Parameter SMC over physical time: SMC² with particle filters, exact with Kalman."""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.smc.base import map_fn
from blackjax.smc.ess import ess as effective_sample_size
from blackjax.smc.resampling import systematic
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from jax.flatten_util import ravel_pytree

from probjax.inference.filtering.temporal import (
    TemporalResult,
    _observations,
    _scan,
    temporal_filter,
)
from probjax.inference.smc.temporal_utils import population_covariance, replay_filter
from probjax.utils.guards import ensure_int

# Optimal-scaling constants for the default Gaussian random-walk proposal.
_TARGET_ACCEPTANCE = 0.234
_SCALE_FACTOR = 2.38


class TemporalSMCState(NamedTuple):
    parameters: Any
    filter_states: Any
    log_weights: Any
    log_likelihoods: Any
    log_evidence: Any
    t0: Any
    t: Any
    num_observations: Any
    proposal_scale: Any

    def posterior_samples(self, key, num_samples):
        """Draw ``num_samples`` parameters from the weighted population."""
        idx = jax.random.categorical(key, self.log_weights, shape=(num_samples,))
        return jax.tree.map(lambda x: x[idx], self.parameters)


class TemporalSMCInfo(NamedTuple):
    log_likelihood: Any
    ess: Any
    resampled: Any
    ancestors: Any
    acceptance_rate: Any
    valid: Any
    skipped: Any


class TemporalSMC(NamedTuple):
    init: Callable
    step: Callable


def temporal_smc(
    model,
    logprior_fn,
    *,
    num_particles=None,
    ess_threshold=0.5,
    num_moves=1,
    proposal_fn=None,
    batch_size=0,
):
    """SMC over parameters that assimilates one observation per step.

    Each parameter particle carries its own filter: a particle filter for a
    :class:`StateSpaceModel` (SMC², with ``num_particles`` latent particles) or
    a Kalman filter for a :class:`LinearGaussianModel` (exact). After every
    observation the outer weights are multiplied by the filters' likelihood
    increments. When the ESS drops below ``ess_threshold``, parameters are
    resampled together with their filters and moved by ``num_moves``
    pseudo-marginal Metropolis-Hastings steps, which re-filter the observed
    prefix for each proposal.

    Args:
        model: A :class:`StateSpaceModel`, :class:`LinearGaussianModel` or
            :class:`TemporalFilter`.
        logprior_fn: ``theta -> scalar`` log prior. Initial parameters passed to
            ``init`` must be draws from this prior.
        num_particles: Latent particles per filter (particle filters only).
        ess_threshold: Resample when the ESS falls below this fraction.
        num_moves: MH moves after each resampling; ``0`` disables them.
        proposal_fn: ``(key, theta) -> (theta', log_q_reverse_minus_forward)``.
            Defaults to a Gaussian random walk with the weighted population
            covariance and a scale tuned towards 23.4% acceptance.
        batch_size: Parameters processed at once; ``0`` vectorizes all.
    """
    filt = temporal_filter(model, num_particles)
    if filt.likelihood_kind not in ("exact", "unbiased"):
        raise ValueError("temporal_smc needs an exact or unbiased likelihood.")
    if not 0 <= ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in [0, 1].")
    ensure_int("num_moves", num_moves, min_value=0)
    adaptive = proposal_fn is None

    def init(key, parameters, t0=0.0):
        leaves = jax.tree.leaves(parameters)
        if not leaves or any(x.ndim < 1 for x in leaves):
            raise ValueError("Parameter leaves must have a leading particle axis.")
        n = leaves[0].shape[0]
        if n < 1 or any(x.shape[0] != n for x in leaves):
            raise ValueError("All parameter leaves must have the same nonempty batch.")
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        filter_states = map_fn(lambda a: filt.init(a[0], a[1], t0), batch_size)((
            jax.random.split(key, n),
            parameters,
        ))
        log_priors = map_fn(logprior_fn, batch_size)(parameters)
        if log_priors.shape != (n,):
            raise ValueError("logprior_fn must return a scalar per parameter.")
        zeros = jnp.zeros_like(log_priors)
        return TemporalSMCState(
            parameters,
            filter_states,
            zeros - jnp.log(n),
            zeros,
            jnp.zeros((), zeros.dtype),
            t0,
            t0,
            jnp.array(0, jnp.int32),
            jnp.ones((), zeros.dtype),
        )

    def rejuvenate(key, state, parameters, filter_states, log_likelihoods, data):
        count = state.num_observations + 1  # includes the new observation
        if adaptive:
            covariance = population_covariance(state.parameters, state.log_weights)
            factor = jnp.linalg.cholesky(covariance)

        def propose(key, theta):
            if not adaptive:
                return proposal_fn(key, theta)
            flat, unravel = ravel_pytree(theta)
            noise = factor @ jax.random.normal(key, flat.shape)
            scale = state.proposal_scale * _SCALE_FACTOR / jnp.sqrt(flat.size)
            return unravel(flat + scale * noise), 0.0

        def move(_, carry):
            key, theta, filter_state, log_likelihood, accepted = carry
            key, proposal_key, replay_key, accept_key = jax.random.split(key, 4)
            proposed, correction = propose(proposal_key, theta)
            proposed_state, proposed_ll = replay_filter(
                filt, replay_key, proposed, state.t0, data, count
            )
            log_ratio = (
                logprior_fn(proposed)
                - logprior_fn(theta)
                + proposed_ll
                - log_likelihood
                + correction
            )
            # The current estimate is kept on rejection: refreshing it would
            # change the pseudo-marginal target.
            current, (accept, _, _) = static_binomial_sampling(
                accept_key,
                jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio),
                (theta, filter_state, log_likelihood),
                (proposed, proposed_state, proposed_ll),
            )
            return (key, *current, accepted + accept)

        def moves(args):
            key, theta, filter_state, log_likelihood = args
            carry = (key, theta, filter_state, log_likelihood, jnp.array(0.0))
            _, theta, filter_state, log_likelihood, accepted = jax.lax.fori_loop(
                0, num_moves, move, carry
            )
            return theta, filter_state, log_likelihood, accepted / num_moves

        n = state.log_weights.shape[0]
        parameters, filter_states, log_likelihoods, rates = map_fn(moves, batch_size)((
            jax.random.split(key, n),
            parameters,
            filter_states,
            log_likelihoods,
        ))
        return parameters, filter_states, log_likelihoods, rates.mean()

    def step(key, state, ts, observations, observed=None):
        """Assimilate observation ``state.num_observations`` of the full data.

        Pass the whole dataset seen so far: the new observation is read at
        index ``state.num_observations`` and moves replay the prefix before it.
        """
        data = _observations(ts, observations, observed)
        ts, observations, observed = data
        count = state.num_observations
        n = state.log_weights.shape[0]
        index = jnp.minimum(count, ts.shape[0] - 1)
        t = ts[index]
        filter_key, resample_key, move_key = jax.random.split(key, 3)

        filter_states, infos = map_fn(
            lambda a: filt.step(
                a[0], a[1], a[2], t, observations[index], observed[index]
            ),
            batch_size,
        )((
            jax.random.split(filter_key, n),
            state.filter_states,
            state.parameters,
        ))
        # A NaN increment (e.g. a simulator diverging at extreme parameters)
        # gives that particle zero weight instead of poisoning the population.
        increments = jnp.where(
            jnp.isnan(infos.log_likelihood), -jnp.inf, infos.log_likelihood
        )
        log_likelihoods = state.log_likelihoods + increments
        log_weights = state.log_weights + increments
        increment = jax.scipy.special.logsumexp(log_weights)
        # The data must extend what the state has seen: its last assimilated
        # time is state.t.
        consistent = (count == 0) | (ts[jnp.maximum(count - 1, 0)] == state.t)
        valid = (count < ts.shape[0]) & (t > state.t) & consistent
        valid = valid & jnp.isfinite(increment)
        log_weights = log_weights - increment
        ess = effective_sample_size(log_weights)
        resampled = valid & (ess < ess_threshold * n)

        ancestors = jax.lax.cond(
            resampled,
            lambda: systematic(resample_key, jnp.exp(log_weights), n),
            lambda: jnp.arange(n),
        ).astype(jnp.int32)
        parameters, filter_states, log_likelihoods = jax.tree.map(
            lambda x: x[ancestors], (state.parameters, filter_states, log_likelihoods)
        )
        weighted = state._replace(log_weights=log_weights)
        log_weights = jnp.where(resampled, -jnp.log(n), log_weights)
        acceptance_rate = jnp.zeros((), log_weights.dtype)
        if num_moves > 0:
            parameters, filter_states, log_likelihoods, acceptance_rate = jax.lax.cond(
                resampled,
                lambda: rejuvenate(
                    move_key,
                    weighted,
                    parameters,
                    filter_states,
                    log_likelihoods,
                    data,
                ),
                lambda: (
                    parameters,
                    filter_states,
                    log_likelihoods,
                    acceptance_rate,
                ),
            )
        proposal_scale = state.proposal_scale
        if adaptive and num_moves > 0:
            tuned = update_scale_from_acceptance_rate(
                proposal_scale, acceptance_rate, _TARGET_ACCEPTANCE
            )
            proposal_scale = jnp.where(
                resampled, jnp.clip(tuned, 1e-3, 1e3), proposal_scale
            )

        next_state = TemporalSMCState(
            parameters,
            filter_states,
            log_weights,
            log_likelihoods,
            state.log_evidence + increment,
            state.t0,
            t,
            count + 1,
            proposal_scale,
        )
        # Atomic: an invalid step (impossible observation, nothing left to
        # assimilate) leaves the state unchanged.
        next_state = jax.lax.cond(valid, lambda: next_state, lambda: state)
        info = TemporalSMCInfo(
            jnp.where(valid, increment, 0.0),
            jnp.where(valid, ess, 0.0),
            resampled,
            ancestors,
            acceptance_rate,
            valid,
            jnp.array(False),
        )
        return next_state, info

    return TemporalSMC(init, step)


def run_temporal_smc(
    kernel, key, state, ts, observations, *, observed=None, history="none"
):
    """Assimilate every observation at ``ts`` that ``state`` has not seen yet.

    To continue a run, call again with the longer dataset: entries already
    assimilated are skipped without consuming random keys, so two calls give
    the same result as one. ``history`` is ``'none'``, ``'full'`` or a window
    size; ``result.log_likelihood`` covers only this call's observations.
    """
    data = _observations(ts, observations, observed)
    ts = data[0]
    zero = jnp.zeros((), state.log_weights.dtype)
    if ts.shape[0] == 0:
        return TemporalResult(state, zero, key, None)
    n = state.log_weights.shape[0]
    skip_info = TemporalSMCInfo(
        zero,
        zero,
        jnp.array(False),
        jnp.arange(n, dtype=jnp.int32),
        zero,
        jnp.array(True),
        jnp.array(True),
    )

    def step(key, state, i):
        return jax.lax.cond(
            i < state.num_observations,
            lambda: (state, skip_info),
            lambda: kernel.step(key, state, *data),
        )

    return _scan(step, key, state, ts, jnp.arange(ts.shape[0]), history)
