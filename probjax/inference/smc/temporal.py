"""Parameter SMC over physical time: SMC² with particle filters, exact with Kalman."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc.proposal import static_binomial_sampling
from jax.flatten_util import ravel_pytree

from probjax.inference.base import Kernel
from probjax.inference.particles import nan_to_neg_inf, take_batch
from probjax.inference.smc.ess import ess as effective_sample_size
from probjax.inference.smc.resampling import systematic
from probjax.inference.smc.temporal_utils import population_covariance, replay_filter
from probjax.utils.guards import ensure_int
from probjax.utils.typing import Array

# Optimal-scaling constants for the default Gaussian random-walk proposal.
_TARGET_ACCEPTANCE = 0.234
_SCALE_FACTOR = 2.38


class TemporalSMCState(NamedTuple):
    """A weighted population of parameters, each with its own filter.

    Attributes:
        parameters: pytree with a leading particle axis.
        filter_states: each parameter's filter state; their
            ``log_likelihood`` is ``log p(y_1:t | theta)`` (or its estimate).
        log_weights: normalised log weights.
        log_evidence: running estimate of ``log p(y_1:t)``.
        t0, t: initial time and time of the last assimilated observation.
        num_observations: observations assimilated so far.
        proposal_scale: scale of the adaptive random-walk move.
        data: ``(ts, ys, observed)`` buffers of the assimilated observations,
            which the moves replay.
    """

    parameters: Any
    filter_states: Any
    log_weights: Any
    log_evidence: Any
    t0: Any
    t: Any
    num_observations: Any
    proposal_scale: Any
    data: Any

    def posterior_samples(self, rng, num_samples: int) -> Any:
        """Draw ``num_samples`` parameters from the weighted population."""
        idx = jax.random.categorical(rng, self.log_weights, shape=(num_samples,))
        return take_batch(self.parameters, idx)


class TemporalSMCInfo(NamedTuple):
    log_evidence_increment: Any
    ess: Any
    resampled: Any
    ancestors: Any
    acceptance_rate: Any
    valid: Any


def _map(fn: Callable, xs: Any, batch_size: int) -> Any:
    if batch_size:
        return jax.lax.map(fn, xs, batch_size=batch_size)
    return jax.vmap(fn)(xs)


def temporal_smc(
    filter: Kernel,
    logprior_fn: Callable,
    *,
    max_observations: int,
    ess_threshold: float = 0.5,
    num_moves: int = 1,
    proposal_fn: Callable | None = None,
    batch_size: int = 0,
) -> Kernel:
    """SMC over parameters that assimilates one observation per step.

    Each parameter particle carries its own filter, built for one ``theta``:
    a :func:`~probjax.inference.particle_filter` gives SMC² (unbiased
    likelihoods), a :func:`~probjax.inference.kalman_filter` exact parameter
    SMC. After every observation the outer weights are multiplied by the
    filters' likelihood increments. When the ESS drops below
    ``ess_threshold``, parameters are resampled together with their filters
    and moved by ``num_moves`` pseudo-marginal Metropolis-Hastings steps,
    which re-filter the observations seen so far for each proposal.

    The kernel steps like a filter, ``step(key, state, None, t, y,
    observed=True)``, so ``run(key, kernel, state, None, len(ts), args=(ts,
    ys))`` assimilates a dataset, and running more steps continues it.

    Args:
        filter: Filter kernel for one ``theta`` (its params).
        logprior_fn: ``theta -> scalar`` log prior. Initial parameters passed
            to ``init`` must be draws from this prior.
        max_observations: Capacity of the buffer of assimilated observations;
            steps beyond it are rejected (``info.valid`` is false).
        ess_threshold: Resample when the ESS falls below this fraction.
        num_moves: MH moves after each resampling; ``0`` disables them.
        proposal_fn: ``(key, theta) -> (theta', log_q_reverse_minus_forward)``.
            Defaults to a Gaussian random walk with the weighted population
            covariance and a scale tuned towards 23.4% acceptance.
        batch_size: Parameters processed at once; ``0`` vectorizes all.

    Returns:
        A :class:`~probjax.inference.Kernel` with ``init(key, parameters,
        observation_like, t0=0.0)``; ``observation_like`` is any observation,
        which sets the buffer's shape.
    """
    if not 0 <= ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in [0, 1].")
    ensure_int("num_moves", num_moves, min_value=0)
    ensure_int("max_observations", max_observations, min_value=1)
    adaptive = proposal_fn is None

    def init(
        key, parameters: Any, observation_like: Any, t0: Any = 0.0
    ) -> TemporalSMCState:
        leaves = jax.tree.leaves(parameters)
        if not leaves or any(x.ndim < 1 for x in leaves):
            raise ValueError("Parameter leaves must have a leading particle axis.")
        n = leaves[0].shape[0]
        if n < 1 or any(x.shape[0] != n for x in leaves):
            raise ValueError("All parameter leaves must have the same nonempty batch.")
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, float))
        filter_states = _map(
            lambda a: filter.init(a[0], a[1], t0),
            (jax.random.split(key, n), parameters),
            batch_size,
        )
        log_priors = _map(logprior_fn, parameters, batch_size)
        if log_priors.shape != (n,):
            raise ValueError("logprior_fn must return a scalar per parameter.")
        dtype = filter_states.log_likelihood.dtype
        observation_like = jnp.asarray(observation_like)
        data = (
            jnp.zeros((max_observations,), t0.dtype),
            jnp.zeros(
                (max_observations,) + observation_like.shape, observation_like.dtype
            ),
            jnp.zeros((max_observations,), bool),
        )
        return TemporalSMCState(
            parameters,
            filter_states,
            jnp.full((n,), -jnp.log(n), dtype),
            jnp.zeros((), dtype),
            t0,
            t0,
            jnp.array(0, jnp.int32),
            jnp.ones((), dtype),
            data,
        )

    def rejuvenate(key, state: TemporalSMCState, parameters, filter_states, data):
        count = state.num_observations + 1  # includes the new observation
        # Only the adaptive proposal uses the factor; the placeholder keeps
        # its type concrete.
        factor: Array = jnp.zeros((0, 0))
        if adaptive:
            covariance = population_covariance(state.parameters, state.log_weights)
            factor = jnp.linalg.cholesky(covariance)

        def propose(key, theta: Any) -> tuple[Any, Any]:
            if not adaptive:
                assert proposal_fn is not None
                return proposal_fn(key, theta)
            flat, unravel = ravel_pytree(theta)
            noise = factor @ jax.random.normal(key, flat.shape)
            scale = state.proposal_scale * _SCALE_FACTOR / jnp.sqrt(flat.size)
            return unravel(flat + scale * noise), 0.0

        def move(_, carry: Any) -> Any:
            key, theta, filter_state, accepted = carry
            key, proposal_key, replay_key, accept_key = jax.random.split(key, 4)
            proposed, correction = propose(proposal_key, theta)
            proposed_state = replay_filter(
                filter, replay_key, proposed, state.t0, data, count
            )
            log_ratio = (
                logprior_fn(proposed)
                - logprior_fn(theta)
                + proposed_state.log_likelihood
                - filter_state.log_likelihood
                + correction
            )
            # The current estimate is kept on rejection: refreshing it would
            # change the pseudo-marginal target.
            log_p_accept: Any = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
            current, (accept, _, _) = static_binomial_sampling(
                accept_key,
                log_p_accept,
                (theta, filter_state),
                (proposed, proposed_state),
            )
            return (key, *current, accepted + accept)

        def moves(args: Any) -> Any:
            key, theta, filter_state = args
            carry = (key, theta, filter_state, jnp.array(0.0))
            _, theta, filter_state, accepted = jax.lax.fori_loop(
                0, num_moves, move, carry
            )
            return theta, filter_state, accepted / num_moves

        n = state.log_weights.shape[0]
        parameters, filter_states, rates = _map(
            moves, (jax.random.split(key, n), parameters, filter_states), batch_size
        )
        return parameters, filter_states, rates.mean()

    def step(
        key,
        state: TemporalSMCState,
        params: Any,
        t: Any,
        y: Any,
        observed: Any = True,
    ) -> tuple[TemporalSMCState, TemporalSMCInfo]:
        del params
        t = jnp.asarray(t, state.t.dtype)
        count = state.num_observations
        n = state.log_weights.shape[0]
        filter_key, resample_key, move_key = jax.random.split(key, 3)
        ts, ys, mask = state.data
        slot = jnp.minimum(count, max_observations - 1)
        data = (ts.at[slot].set(t), ys.at[slot].set(y), mask.at[slot].set(observed))

        filter_states, infos = _map(
            lambda a: filter.step(a[0], a[1], a[2], t, y, observed),
            (jax.random.split(filter_key, n), state.filter_states, state.parameters),
            batch_size,
        )
        # A NaN increment (e.g. a simulator diverging at extreme parameters)
        # gives that particle zero weight instead of poisoning the population.
        increments = nan_to_neg_inf(infos.log_likelihood_increment)
        log_weights = state.log_weights + increments
        increment = jax.scipy.special.logsumexp(log_weights)
        valid = (count < max_observations) & (t > state.t) & jnp.isfinite(increment)
        log_weights = log_weights - increment
        ess = effective_sample_size(log_weights)
        resampled = valid & (ess < ess_threshold * n)

        ancestors = jax.lax.cond(
            resampled,
            lambda: systematic(resample_key, jnp.exp(log_weights), n),
            lambda: jnp.arange(n),
        ).astype(jnp.int32)
        parameters, filter_states = take_batch(
            (state.parameters, filter_states), ancestors
        )
        weighted = state._replace(log_weights=log_weights)
        log_weights = jnp.where(resampled, -jnp.log(n), log_weights)
        acceptance_rate = jnp.zeros((), log_weights.dtype)
        if num_moves > 0:
            parameters, filter_states, acceptance_rate = jax.lax.cond(
                resampled,
                lambda: rejuvenate(move_key, weighted, parameters, filter_states, data),
                lambda: (parameters, filter_states, acceptance_rate),
            )
        proposal_scale = state.proposal_scale
        if adaptive and num_moves > 0:
            # Multiplicative Robbins-Monro step towards the target acceptance.
            tuned = proposal_scale * jnp.exp(acceptance_rate - _TARGET_ACCEPTANCE)
            proposal_scale = jnp.where(
                resampled, jnp.clip(tuned, 1e-3, 1e3), proposal_scale
            )

        next_state = TemporalSMCState(
            parameters,
            filter_states,
            log_weights,
            state.log_evidence + increment,
            state.t0,
            t,
            count + 1,
            proposal_scale,
            data,
        )
        # Atomic: an invalid step (impossible observation, a full buffer, a
        # time that does not advance) leaves the state unchanged.
        next_state = jax.lax.cond(valid, lambda: next_state, lambda: state)
        info = TemporalSMCInfo(
            jnp.where(valid, increment, 0.0),
            jnp.where(valid, ess, 0.0),
            resampled,
            ancestors,
            acceptance_rate,
            valid,
        )
        return next_state, info

    return Kernel(init, step, lambda state, params=None: params)
