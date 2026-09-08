"""Time-indexed parameter SMC and SMC² with optional observation bridges."""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.smc.base import map_fn
from blackjax.smc.ess import ess as effective_sample_size
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from jax.flatten_util import ravel_pytree

from probjax.inference.smc.base import make_mcmc_adapter
from probjax.inference.smc.temporal_utils import population_covariance, replay_filter


class ReplayData(NamedTuple):
    """Complete data prefix starting at t0; unused future observations are permitted."""

    ts: Any
    observations: Any
    observed: Any


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
    move_steps: Any
    lineages: Any


class TemporalSMCInfo(NamedTuple):
    log_likelihood: Any
    ess: Any
    resampled: Any
    ancestors: Any
    acceptance_rate: Any
    valid: Any
    num_tempering_steps: Any
    tempering_param: Any


class TemporalSMC(NamedTuple):
    init: Callable
    step: Callable
    requires_replay: bool = False


def temporal_smc(
    backend,
    logprior_fn,
    *,
    proposal_fn=None,
    num_rejuvenation_steps=1,
    ess_threshold=0.5,
    batch_size=0,
    adaptive_proposal=False,
    proposal_scale=1.0,
    adaptive_num_steps=False,
    max_rejuvenation_steps=16,
    target_accepted_moves=2.0,
    target_acceptance=0.234,
    tempering_ess=None,
    max_tempering_steps=64,
    mcmc_kernel=None,
    mcmc_parameters=None,
    mcmc_kernel_kwargs=None,
):
    """Build temporal SMC using exact or unbiased incremental likelihoods.

    Initial particles must be equally weighted prior draws. proposal_fn(key,theta)
    returns (candidate, log_q_reverse_minus_forward). adaptive_proposal=True
    instead uses a Gaussian random walk with weighted population covariance and
    an acceptance-tuned scale, frozen during each rejuvenation sweep.

    Exact backends may instead supply an existing probjax mcmc_kernel and its
    unbatched parameter dictionary (e.g. HMC). This uses differentiable prefix
    replay. No gradient-based move is applied to noisy particle likelihoods.

    tempering_ess optionally bridges each observation with conditional ESS steps.
    For stochastic backends these are EXTENDED-SPACE targets with retained filter
    likelihood estimates, not powers of the marginalized observation likelihood.
    The beta=1 endpoint is ordinary SMC². Prefix replay retains both L_previous
    and L_current so pseudo-marginal proposals preserve every bridge target.

    Each step is atomic: invalid replay/time or an unfinished bridge returns the
    input state with info.valid=False; increase max_tempering_steps and retry.
    All-impossible observations are reported as invalid (never silently uniform).
    batch_size bounds population mapping memory; zero uses full vectorization.
    """
    if backend.likelihood_kind not in ('exact', 'unbiased'):
        raise ValueError(
            'Parameter SMC requires an exact or unbiased likelihood backend.'
        )
    if not 0 <= ess_threshold <= 1:
        raise ValueError('ess_threshold must lie in [0, 1].')
    if not isinstance(num_rejuvenation_steps, int) or num_rejuvenation_steps < 0:
        raise ValueError('num_rejuvenation_steps must be a nonnegative integer.')
    if adaptive_num_steps and (
        not isinstance(max_rejuvenation_steps, int)
        or max_rejuvenation_steps < num_rejuvenation_steps
        or target_accepted_moves <= 0
    ):
        raise ValueError('Invalid adaptive move-count budget.')
    if tempering_ess is not None and not 0 < tempering_ess < 1:
        raise ValueError('tempering_ess must lie strictly between zero and one.')
    if not isinstance(max_tempering_steps, int) or max_tempering_steps < 1:
        raise ValueError('max_tempering_steps must be positive.')
    if proposal_scale <= 0 or not 0 < target_acceptance < 1:
        raise ValueError('proposal_scale and target_acceptance must be positive/valid.')
    if sum((proposal_fn is not None, adaptive_proposal, mcmc_kernel is not None)) > 1:
        raise ValueError(
            'Choose one proposal function, adaptive Gaussian, or MCMC kernel.'
        )
    if mcmc_kernel is not None:
        if backend.likelihood_kind != 'exact':
            raise ValueError(
                'Existing MCMC kernels require an exact likelihood backend.'
            )
        if mcmc_parameters is None:
            raise ValueError('mcmc_parameters must be supplied with an MCMC kernel.')
        mcmc_init, mcmc_step = make_mcmc_adapter(
            mcmc_kernel, **(mcmc_kernel_kwargs or {})
        )
    moves = (
        proposal_fn is not None or adaptive_proposal or mcmc_kernel is not None
    ) and num_rejuvenation_steps > 0

    if adaptive_num_steps and not moves:
        raise ValueError(
            'Adaptive move counts require an enabled rejuvenation proposal.'
        )

    def init(key, parameters, t0):
        leaves = jax.tree.leaves(parameters)
        if not leaves or any(x.ndim < 1 for x in leaves):
            raise ValueError('Parameter leaves must have a leading particle axis.')
        n = leaves[0].shape[0]
        if n < 1 or any(x.shape[0] != n for x in leaves):
            raise ValueError('All parameter leaves must have the same nonempty batch.')
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        states = map_fn(lambda args: backend.init(args[0], args[1], t0), batch_size)((
            jax.random.split(key, n),
            parameters,
        ))
        priors = map_fn(logprior_fn, batch_size)(parameters)
        if priors.shape != (n,):
            raise ValueError('logprior_fn must return a scalar per parameter.')
        zeros = jnp.zeros_like(priors)
        return TemporalSMCState(
            parameters,
            states,
            zeros - jnp.log(n),
            zeros,
            jnp.zeros((), priors.dtype),
            t0,
            t0,
            jnp.array(0, jnp.int32),
            jnp.asarray(proposal_scale),
            jnp.asarray(num_rejuvenation_steps, dtype=jnp.int32),
            jnp.arange(n, dtype=jnp.int32),
        )

    def step(key, state, t, observation, observed=True, replay=None):
        t = jnp.asarray(t, state.t.dtype)
        count = state.num_observations + 1
        replay_valid = jnp.array(True)
        if moves:
            if replay is None:
                raise ValueError(
                    'Rejuvenation requires ReplayData for the full observed prefix.'
                )
            if replay.ts.ndim != 1 or replay.ts.shape[0] == 0:
                raise ValueError(
                    'ReplayData cannot be empty when assimilating observations.'
                )
            if (
                replay.observations.shape[0] != replay.ts.shape[0]
                or replay.observed.shape != replay.ts.shape
            ):
                raise ValueError('ReplayData arrays must share their time dimension.')
            replay_valid = (count <= replay.ts.shape[0]) & (replay.ts[count - 1] == t)
            if not isinstance(replay_valid, jax.core.Tracer) and not bool(replay_valid):
                raise ValueError(
                    'ReplayData must contain the full prefix through this time.'
                )
        n = state.log_weights.shape[0]
        key, predict_key = jax.random.split(key)
        states, infos = map_fn(
            lambda args: backend.step(
                args[0], args[1], args[2], t, observation, observed
            ),
            batch_size,
        )((jax.random.split(predict_key, n), state.filter_states, state.parameters))
        likelihoods = state.log_likelihoods + infos.log_likelihood

        def rejuvenate(
            key, parameters, states, likelihoods, previous_ll, beta, covariance, scale
        ):
            def one(args):
                key, theta, current, ll, previous = args

                def target(p):
                    _, new, old = replay_filter(
                        backend, key, p, state.t0, replay, count, differentiable=True
                    )
                    return logprior_fn(p) + old + beta * (new - old)

                def move(_, carry):
                    key, theta, current, ll, previous, accepted = carry
                    key, proposal_key, replay_key, accept_key = jax.random.split(key, 4)
                    if mcmc_kernel is not None:
                        init_key, kernel_key = jax.random.split(proposal_key)
                        chain = mcmc_init(theta, target, rng_key=init_key)
                        chain, info = mcmc_step(
                            kernel_key, chain, target, **mcmc_parameters
                        )
                        theta = chain.position
                        current, ll, previous = replay_filter(
                            backend, replay_key, theta, state.t0, replay, count
                        )
                        accepted += info.acceptance_rate
                    else:
                        if adaptive_proposal:
                            flat, unravel = ravel_pytree(theta)
                            noise = jax.random.normal(proposal_key, flat.shape)
                            proposed = unravel(
                                flat
                                + scale
                                * (2.38 / jnp.sqrt(flat.size))
                                * (jnp.linalg.cholesky(covariance) @ noise)
                            )
                            correction = 0.0
                        else:
                            proposed, correction = proposal_fn(proposal_key, theta)
                        candidate, new_ll, old_ll = replay_filter(
                            backend, replay_key, proposed, state.t0, replay, count
                        )
                        ratio = (
                            logprior_fn(proposed)
                            - logprior_fn(theta)
                            + old_ll
                            - previous
                            + beta * ((new_ll - old_ll) - (ll - previous))
                            + correction
                        )
                        (theta, current, ll, previous), (accept, _, _) = (
                            static_binomial_sampling(
                                accept_key,
                                jnp.where(jnp.isnan(ratio), -jnp.inf, ratio),
                                (theta, current, ll, previous),
                                (proposed, candidate, new_ll, old_ll),
                            )
                        )
                        accepted += accept
                    return key, theta, current, ll, previous, accepted

                _, theta, current, ll, previous, accepted = jax.lax.fori_loop(
                    0,
                    state.move_steps,
                    move,
                    (key, theta, current, ll, previous, jnp.array(0.0)),
                )
                return theta, current, ll, previous, accepted / state.move_steps

            parameters, states, ll, old, rates = map_fn(one, batch_size)((
                jax.random.split(key, n),
                parameters,
                states,
                likelihoods,
                previous_ll,
            ))
            return parameters, states, ll, old, rates.mean()

        def stage(carry):
            (
                key,
                parameters,
                states,
                ll,
                old,
                weights,
                logz,
                beta,
                stages,
                accepted,
                any_resampled,
                parents,
                scale,
                valid,
                last_ess,
            ) = carry
            key, resample_key, move_key = jax.random.split(key, 3)
            increments = ll - old
            if tempering_ess is None:
                delta = 1.0 - beta
            else:

                def objective(delta):
                    power = jnp.where(delta == 0.0, 0.0, delta * increments)
                    log_cess = (
                        jnp.log(n)
                        + 2 * jax.scipy.special.logsumexp(weights + power)
                        - jax.scipy.special.logsumexp(weights + 2 * power)
                    )
                    return log_cess - jnp.log(n * tempering_ess)

                delta = dichotomy(objective, jnp.array(0.0), 1.0 - beta)
            beta_next = jnp.minimum(beta + delta, 1.0)
            raw = weights + delta * increments
            increment = jax.scipy.special.logsumexp(raw)
            valid = valid & jnp.isfinite(increment) & jnp.isfinite(delta) & (delta > 0)
            weights = jnp.where(valid, raw - increment, -jnp.inf)
            last_ess = jnp.where(valid, effective_sample_size(weights), 0.0)
            resampled = valid & ((last_ess < ess_threshold * n) | (beta_next < 1.0))
            covariance = (
                population_covariance(parameters, weights)
                if adaptive_proposal
                else jnp.zeros((1, 1))
            )
            idx = jax.lax.cond(
                resampled,
                lambda: systematic(resample_key, jnp.exp(weights), n),
                lambda: jnp.arange(n, dtype=jnp.int32),
            )
            parameters, states, ll, old, parents = jax.tree.map(
                lambda x: x[idx], (parameters, states, ll, old, parents)
            )
            weights = jnp.where(resampled, -jnp.log(n), weights)
            rate = jnp.array(0.0)
            if moves:
                parameters, states, ll, old, rate = jax.lax.cond(
                    resampled,
                    lambda: rejuvenate(
                        move_key,
                        parameters,
                        states,
                        ll,
                        old,
                        beta_next,
                        covariance,
                        scale,
                    ),
                    lambda: (parameters, states, ll, old, rate),
                )
            if adaptive_proposal:
                scale = jnp.where(
                    resampled,
                    jnp.clip(
                        update_scale_from_acceptance_rate(
                            scale, rate, target_acceptance
                        ),
                        1e-3,
                        1e3,
                    ),
                    scale,
                )
            return (
                key,
                parameters,
                states,
                ll,
                old,
                weights,
                logz + increment,
                beta_next,
                stages + 1,
                accepted + rate,
                any_resampled | resampled,
                parents,
                scale,
                valid,
                last_ess,
            )

        carry = (
            key,
            state.parameters,
            states,
            likelihoods,
            state.log_likelihoods,
            state.log_weights,
            jnp.zeros_like(state.log_evidence),
            jnp.array(0.0),
            jnp.array(0),
            jnp.array(0.0),
            jnp.array(False),
            jnp.arange(n, dtype=jnp.int32),
            state.proposal_scale,
            replay_valid & (t > state.t),
            jnp.array(0.0),
        )
        # A single untempered step avoids a while-loop around the common path.
        if tempering_ess is None:
            carry = stage(carry)
        else:
            carry = jax.lax.while_loop(
                lambda c: (c[7] < 1.0) & (c[8] < max_tempering_steps) & c[13],
                stage,
                carry,
            )
        (
            _,
            parameters,
            states,
            ll,
            _,
            weights,
            increment,
            beta,
            stages,
            acceptance,
            resampled,
            parents,
            scale,
            valid,
            last_ess,
        ) = carry
        valid = valid & (beta >= 1.0)
        next_state = TemporalSMCState(
            parameters,
            states,
            weights,
            ll,
            state.log_evidence + increment,
            state.t0,
            t,
            count,
            scale,
            jnp.where(
                adaptive_num_steps & resampled,
                jnp.clip(
                    jnp.ceil(
                        target_accepted_moves
                        / jnp.maximum(acceptance / jnp.maximum(stages, 1), 1e-6)
                    ),
                    1,
                    max_rejuvenation_steps,
                ).astype(jnp.int32),
                state.move_steps,
            ),
            state.lineages[parents],
        )
        next_state = jax.lax.cond(valid, lambda: next_state, lambda: state)
        return next_state, TemporalSMCInfo(
            jnp.where(valid, increment, 0.0),
            last_ess,
            resampled,
            parents,
            acceptance / jnp.maximum(stages, 1),
            valid,
            stages,
            beta,
        )

    return TemporalSMC(init, step, moves)


def run_temporal_smc(
    kernel, key, state, ts, observations, *, observed=None, history="none", replay=None
):
    """Run parameter SMC; return TemporalResult with optional population history.

    For a fresh state, replay defaults to this run's data. When resuming a stream
    with rejuvenation, pass ReplayData starting at state.t0, not just the new chunk.
    History='none' stores only the final population; integer windows are bounded.
    """
    from probjax.inference.filtering.temporal import (
        TemporalFilter,
        _observations,
        run_temporal_filter,
    )

    ts, observations, observed = _observations(ts, observations, observed)
    if replay is None:
        if (
            kernel.requires_replay
            and not isinstance(state.num_observations, jax.core.Tracer)
            and int(state.num_observations) != 0
        ):
            raise ValueError(
                "Resuming with rejuvenation requires full-prefix ReplayData."
            )
        replay = ReplayData(ts, observations, observed)
    else:
        replay = ReplayData(*_observations(*replay))
    adapter = TemporalFilter(
        kernel.init,
        lambda key, state, theta, t, y, mask: kernel.step(
            key, state, t, y, mask, replay=replay
        ),
        "exact",  # runner does not inspect this; this is an outer evidence increment.
    )
    return run_temporal_filter(
        adapter, key, state, None, ts, observations, observed=observed, history=history
    )
