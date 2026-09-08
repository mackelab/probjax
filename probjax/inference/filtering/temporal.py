"""Incremental filtering on a physical time grid.

A backend consumes one interval and returns a *normalized* predictive likelihood
increment. Times in a run are observation times AFTER the initial state's time;
the initial distribution is supplied by the caller (and may already be filtered).
"""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.particle_filter import ParticleFilter


class TemporalFilter(NamedTuple):
    """Pure ``init(key, theta, t0)`` and ``step(key, state, theta, t, y, mask)``.

    ``step`` returns (state, info); info.log_likelihood is the incremental log
    likelihood. ``likelihood_kind`` describes the likelihood, not its logarithm:
    'exact', 'unbiased', or 'approximate'. SMC² requires an unbiased likelihood.
    """

    init: Callable
    step: Callable
    likelihood_kind: str


def kalman_backend(initial_fn, transition_fn, observation_fn):
    """Adapt the dense Kalman filter for exact linear-Gaussian likelihoods.

    initial_fn(theta, t0) -> (mean, covariance)
    transition_fn(theta, t_previous, t_next) -> (Phi, Q)
    observation_fn(theta, t_next) -> (C, R)
    """

    def init(key, theta, t0):
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        mean, cov = initial_fn(theta, t0)
        return kalman_filter.init(mean, cov, t0)

    def step(key, state, theta, t, observation, observed=True):
        t = jnp.asarray(t, dtype=state.t.dtype)
        kernel = kalman_filter(
            lambda old, new: transition_fn(theta, old, new),
            lambda new: observation_fn(theta, new),
        )
        return jax.lax.cond(
            observed,
            lambda: kernel(state, t, observation, key),
            lambda: kernel(state, t, None, key),
        )

    return TemporalFilter(init, step, "exact")


def particle_backend(
    initial_fn,
    transition_fn,
    log_likelihood_fn,
    *,
    ess_threshold=0.5,
    proposal_fn=None,
    proposal_logdensity_fn=None,
    transition_logdensity_fn=None,
):
    """Adapt the particle filter, using discrete systematic resampling.

    initial_fn(key, theta, t0) -> particles (N, D), sampled from the initial law
    transition_fn(key, theta, particles, t_previous, t_next) -> particles
    log_likelihood_fn(theta, particles, observation, t_next) -> (N,)

    Optional proposals use the transition sampler's signature plus ``observation``
    as the final argument. Both densities take
    (theta, new_particles, old_particles, t_previous, t_next), with the proposal
    density also taking observation. Missing observations use the model transition.
    All three proposal arguments must be supplied together. Proposals must cover
    the support of the transition/observation target.
    """
    supplied = (proposal_fn, proposal_logdensity_fn, transition_logdensity_fn)
    if any(fn is not None for fn in supplied) and not all(
        fn is not None for fn in supplied
    ):
        raise ValueError("A proposal requires its density and the transition density.")
    if not 0 <= ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in [0, 1].")

    def init(key, theta, t0):
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        particles = initial_fn(key, theta, t0)
        if particles.ndim != 2 or particles.shape[0] < 1:
            raise ValueError(
                "Particle initializers must return a nonempty (N, D) array."
            )
        return ParticleFilter.init(particles, jnp.asarray(t0))

    def step(key, state, theta, t, observation, observed=True):
        t = jnp.asarray(t, dtype=state.t.dtype)
        transition = lambda key, x, t: transition_fn(key, theta, x, state.t, t)
        likelihood = lambda x, y, t: log_likelihood_fn(theta, x, y, t)
        kwargs = {}
        if proposal_fn is not None:
            kwargs = dict(
                proposal_transition_fn=lambda key, x, t: proposal_fn(
                    key, theta, x, state.t, t, observation
                ),
                proposal_logdensity_fn=lambda new, old, t: proposal_logdensity_fn(
                    theta, new, old, state.t, t, observation
                ),
                transition_logdensity_fn=lambda new, old, t: transition_logdensity_fn(
                    theta, new, old, state.t, t
                ),
            )
        kernel = ParticleFilter(
            likelihood,
            transition,
            resample_criterion=lambda ess: ess < ess_threshold,
            **kwargs,
        )
        predict = ParticleFilter(
            likelihood, transition, resample_criterion=lambda ess: ess < ess_threshold
        )
        return jax.lax.cond(
            observed,
            lambda: kernel(state, t, observation, key),
            lambda: predict(state, t, None, key),
        )

    return TemporalFilter(init, step, "unbiased")


class TemporalTrace(NamedTuple):
    """Stored states at ``ts``, with their immediately preceding initial state.

    Windowed traces condition on the filtering distribution at the window start;
    they do not provide smoothed estimates for states discarded from the window.
    """

    initial_state: Any
    ts: Any
    states: Any
    infos: Any


class TemporalResult(NamedTuple):
    state: Any
    log_likelihood: Any
    key: Any
    trace: Any


def _observations(ts, observations, observed):
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


def run_temporal_filter(
    backend, key, state, theta, ts, observations, *, observed=None, history="full"
):
    """Scan an incremental filter; ``history`` is 'full', 'none', or a window size.

    The grid must increase from state.t. A positive integer keeps a bounded ring
    buffer; 'none' stores no time history. No growing arrays enter the scan carry.
    Streaming reproduces a run by repeatedly splitting ``key, step_key`` and
    calling backend.step. Returned log_likelihood covers only this run's intervals.
    """
    ts, observations, observed = _observations(ts, observations, observed)
    n = ts.shape[0]
    if history not in ("full", "none") and (
        not isinstance(history, int) or isinstance(history, bool) or history < 1
    ):
        raise ValueError("history must be 'full', 'none', or a positive integer.")

    def advance(carry, data):
        key, state, logz = carry
        key, step_key = jax.random.split(key)
        t, y, mask = data
        state, info = backend.step(step_key, state, theta, t, y, mask)
        return (key, state, logz + info.log_likelihood), (state, info)

    zero = jnp.asarray(0.0, dtype=jnp.result_type(jax.tree.leaves(state)[0], 0.0))
    carry = (key, state, zero)
    if history == "none":

        def discard(carry, data):
            carry, _ = advance(carry, data)
            return carry, None

        (key, final, logz), _ = jax.lax.scan(
            discard, carry, (ts, observations, observed)
        )
        return TemporalResult(final, logz, key, None)
    if history == "full" or n == 0:
        (key, final, logz), (states, infos) = jax.lax.scan(
            advance, carry, (ts, observations, observed)
        )
        return TemporalResult(final, logz, key, TemporalTrace(state, ts, states, infos))

    capacity = min(history, n)
    info_shape = jax.eval_shape(
        backend.step, key, state, theta, ts[0], observations[0], observed[0]
    )[1]
    allocate = lambda x: jnp.zeros((capacity,) + x.shape, x.dtype)
    states = jax.tree.map(allocate, state)
    infos = jax.tree.map(allocate, info_shape)

    def window_step(carry, data):
        running, boundary, states, infos = carry
        i, t, y, mask = data
        slot = i % capacity
        old = jax.tree.map(lambda x: x[slot], states)
        boundary = jax.lax.cond(i >= capacity, lambda: old, lambda: boundary)
        running, (next_state, info) = advance(running, (t, y, mask))
        states = jax.tree.map(lambda x, v: x.at[slot].set(v), states, next_state)
        infos = jax.tree.map(lambda x, v: x.at[slot].set(v), infos, info)
        return (running, boundary, states, infos), None

    ((key, final, logz), boundary, states, infos), _ = jax.lax.scan(
        window_step,
        (carry, state, states, infos),
        (jnp.arange(n), ts, observations, observed),
    )
    order = (jnp.arange(capacity) + n) % capacity
    states, infos = jax.tree.map(lambda x: x[order], (states, infos))
    return TemporalResult(
        final, logz, key, TemporalTrace(boundary, ts[-capacity:], states, infos)
    )
