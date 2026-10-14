"""State-space models and incremental filtering on a physical time grid.

A model is filtered as a :class:`TemporalFilter`, which consumes one interval at a
time and returns a *normalized* predictive likelihood increment. Observation times
are AFTER the initial time ``t0``; condition the initial law on any observation at
``t0`` yourself.
"""

import jax
import jax.numpy as jnp

from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.observations import validate_observations
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.inference.filtering.state_space import (
    LinearGaussianModel,
    StateSpaceModel,
    TemporalFilter,
    TemporalResult,
    TemporalTrace,
)

__all__ = [
    "StateSpaceModel",
    "LinearGaussianModel",
    "TemporalFilter",
    "TemporalTrace",
    "TemporalResult",
    "temporal_filter",
    "run_temporal_filter",
    "scan_history",
]


def temporal_filter(model, num_particles=None, ess_threshold=0.5):
    """Build the filter for ``model``: Kalman for linear-Gaussian, else particles.

    A :class:`TemporalFilter` is returned unchanged.
    """
    if isinstance(model, TemporalFilter):
        return model
    if isinstance(model, LinearGaussianModel):
        return _kalman_filter(model)
    if num_particles is None:
        raise ValueError("A StateSpaceModel needs num_particles.")
    return _particle_filter(model, num_particles, ess_threshold)


def _kalman_filter(model):
    def init(key, theta, t0):
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        mean, cov = model.initial(theta)
        return kalman_filter.init(mean, cov, t0)

    def step(key, state, theta, t, observation, observed=True):
        t = jnp.asarray(t, dtype=state.t.dtype)
        kernel = kalman_filter(
            lambda old, new: model.transition(theta, old, new),
            lambda new: model.observation(theta, new),
        )
        return jax.lax.cond(
            observed,
            lambda: kernel(state, t, observation, key),
            lambda: kernel(state, t, None, key),
        )

    return TemporalFilter(init, step, "exact")


def _particle_filter(model, num_particles, ess_threshold):
    if not 0 <= ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in [0, 1].")

    def init(key, theta, t0):
        t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, 0.0))
        keys = jax.random.split(key, num_particles)
        particles = jax.vmap(model.init, (0, None))(keys, theta)
        return ParticleFilter.init(particles, t0)

    def step(key, state, theta, t, observation, observed=True):
        t = jnp.asarray(t, dtype=state.t.dtype)

        def transition(key, particles, t):
            keys = jax.random.split(key, num_particles)
            step = jax.vmap(model.step, (0, None, 0, None, None))
            return step(keys, theta, particles, state.t, t)

        def likelihood(particles, y, t):
            return jax.vmap(model.log_likelihood, (None, 0, None, None))(
                theta, particles, y, t
            )

        kernel = ParticleFilter(
            likelihood, transition, resample_criterion=lambda ess: ess < ess_threshold
        )
        return jax.lax.cond(
            observed,
            lambda: kernel(state, t, observation, key),
            lambda: kernel(state, t, None, key),
        )

    return TemporalFilter(init, step, "unbiased")


def run_temporal_filter(
    key,
    model,
    theta,
    ts,
    observations,
    *,
    t0=0.0,
    num_particles=None,
    observed=None,
    history="full",
):
    """Filter ``model`` at fixed ``theta`` over observations at ``ts`` after ``t0``.

    ``model`` is a :class:`StateSpaceModel` (needs ``num_particles``), a
    :class:`LinearGaussianModel` or a :class:`TemporalFilter`. ``history`` is
    ``'full'``, ``'none'`` or a window size: a positive integer keeps a ring
    buffer of the last intervals. ``observed`` masks missing observations, whose
    values may be NaN.
    """
    ts, observations, observed = validate_observations(ts, observations, observed)
    filt = temporal_filter(model, num_particles)
    key, init_key = jax.random.split(key)
    state = filt.init(init_key, theta, t0)

    def step(key, state, data):
        t, y, mask = data
        return filt.step(key, state, theta, t, y, mask)

    return scan_history(step, key, state, ts, (ts, observations, observed), history)


def scan_history(step, key, state, ts, xs, history):
    """Scan ``step(key, state, x) -> (state, info)`` over ``xs``, keeping ``history``.

    Steps returning ``info.skipped`` neither consume a key nor add likelihood, so
    resuming a run over a longer data array reproduces a single run.
    """
    n = ts.shape[0]
    if history not in ("full", "none") and (
        not isinstance(history, int) or isinstance(history, bool) or history < 1
    ):
        raise ValueError("history must be 'full', 'none', or a positive integer.")

    def advance(carry, x):
        key, state, logz = carry
        next_key, step_key = jax.random.split(key)
        state, info = step(step_key, state, x)
        if hasattr(info, "skipped"):
            next_key = jax.lax.cond(info.skipped, lambda: key, lambda: next_key)
        return (next_key, state, logz + info.log_likelihood), (state, info)

    zero = jnp.asarray(0.0, dtype=jnp.result_type(jax.tree.leaves(state)[0], 0.0))
    carry = (key, state, zero)
    if history == "none":
        (key, final, logz), _ = jax.lax.scan(
            lambda c, x: (advance(c, x)[0], None), carry, xs
        )
        return TemporalResult(final, logz, key, None)
    if history == "full" or n == 0:
        (key, final, logz), (states, infos) = jax.lax.scan(advance, carry, xs)
        return TemporalResult(final, logz, key, TemporalTrace(state, ts, states, infos))

    capacity = min(history, n)
    first = jax.tree.map(lambda x: x[0], xs)
    info_shape = jax.eval_shape(step, key, state, first)[1]
    allocate = lambda x: jnp.zeros((capacity,) + x.shape, x.dtype)
    states = jax.tree.map(allocate, state)
    infos = jax.tree.map(allocate, info_shape)

    def window_step(carry, data):
        running, boundary, states, infos = carry
        i, x = data
        slot = i % capacity
        old = jax.tree.map(lambda s: s[slot], states)
        boundary = jax.lax.cond(i >= capacity, lambda: old, lambda: boundary)
        running, (next_state, info) = advance(running, x)
        states = jax.tree.map(lambda s, v: s.at[slot].set(v), states, next_state)
        infos = jax.tree.map(lambda s, v: s.at[slot].set(v), infos, info)
        return (running, boundary, states, infos), None

    ((key, final, logz), boundary, states, infos), _ = jax.lax.scan(
        window_step, (carry, state, states, infos), (jnp.arange(n), xs)
    )
    order = (jnp.arange(capacity) + n) % capacity
    states, infos = jax.tree.map(lambda x: x[order], (states, infos))
    return TemporalResult(
        final, logz, key, TemporalTrace(boundary, ts[-capacity:], states, infos)
    )
