"""The bootstrap particle filter and its guided variant."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from probjax.inference.base import Kernel
from probjax.inference.filtering.gaussian import as_time, unobserved
from probjax.inference.filtering.state_space import StateSpaceModel
from probjax.inference.particles import take_batch
from probjax.inference.smc.ess import ess
from probjax.inference.smc.resampling import systematic
from probjax.utils.typing import Array, ArrayLike


class ParticleFilterState(NamedTuple):
    """A weighted particle approximation of the filtering distribution.

    ``log_likelihood`` accumulates an unbiased estimate of ``p(y_1:t | theta)``
    on the log scale.
    """

    particles: Any
    log_weights: Array
    t: Array
    log_likelihood: Array


class ParticleFilterInfo(NamedTuple):
    ancestors: Array
    log_likelihood_increment: Array
    ess: Array


def particle_filter(
    model: StateSpaceModel,
    num_particles: int,
    *,
    ess_threshold: float = 0.5,
    proposal: Optional[Callable] = None,
    unbiased_gradients: bool = False,
) -> Kernel:
    """Particle filter for a state-space model.

    Each step moves the particles through the transition (or ``proposal``),
    weights them by the observation likelihood and resamples when the ESS
    falls below ``ess_threshold``. ``state.log_likelihood`` is an unbiased
    estimate of the likelihood (on the log scale), so the filter can drive a
    pseudo-marginal sampler over ``theta``.

    Args:
        model: The model, written for one particle.
        num_particles: Number of particles.
        ess_threshold: Resample when the ESS falls below this fraction.
        proposal: ``(key, theta, x, y, t_previous, t) -> (x_next, log_q)``,
            a proposal that can look at the observation; needs
            ``model.transition_logdensity``. On steps without an observation
            the transition is used.
        unbiased_gradients: Pass gradients of the weights through resampling
            (a stop-gradient correction), for fitting ``theta`` by gradient.

    Returns:
        A :class:`~probjax.inference.Kernel` with ``init(key, theta, t0=0.0)``
        and ``step(key, state, theta, t, y, observed=True)``.
    """
    if not 0 <= ess_threshold <= 1:
        raise ValueError("ess_threshold must lie in [0, 1].")
    if proposal is not None and model.transition_logdensity is None:
        raise ValueError("A proposal needs model.transition_logdensity.")
    transition = jax.vmap(model.step, (0, None, 0, None, None))
    likelihood = jax.vmap(model.log_likelihood, (None, 0, None, None))

    def init(key: Any, theta: Any, t0: ArrayLike = 0.0) -> ParticleFilterState:
        keys = jax.random.split(key, num_particles)
        particles = jax.vmap(model.init, (0, None))(keys, theta)
        dtype = jnp.result_type(jax.tree.leaves(particles)[0], float)
        return ParticleFilterState(
            particles,
            jnp.full((num_particles,), -jnp.log(num_particles), dtype),
            as_time(t0),
            jnp.zeros((), dtype),
        )

    def propose(key, theta, particles, y, t_previous, t, observed):
        keys = jax.random.split(key, num_particles)
        moved = transition(keys, theta, particles, t_previous, t)
        if proposal is None:
            return moved, jnp.zeros(num_particles)
        guided, log_q = jax.vmap(proposal, (0, None, 0, None, None, None))(
            keys, theta, particles, y, t_previous, t
        )
        log_p = jax.vmap(model.transition_logdensity, (None, 0, 0, None, None))(
            theta, guided, particles, t_previous, t
        )
        particles = jax.tree.map(
            lambda a, b: jnp.where(observed, a, b), guided, moved
        )
        return particles, jnp.where(observed, log_p - log_q, 0.0)

    def step(
        key: Any,
        state: ParticleFilterState,
        theta: Any,
        t: ArrayLike,
        y: ArrayLike,
        observed: Any = True,
    ) -> Tuple[ParticleFilterState, ParticleFilterInfo]:
        move_key, resample_key = jax.random.split(key)
        t = jnp.asarray(t, state.t.dtype)
        y = unobserved(observed, jnp.asarray(y))
        particles, log_correction = propose(
            move_key, theta, state.particles, y, state.t, t, observed
        )
        log_increments = log_correction + jnp.where(
            observed, likelihood(theta, particles, y, t), 0.0
        )
        log_weights = state.log_weights + log_increments
        # The incoming weights are normalised, so this estimates p(y_t | y_<t).
        increment = logsumexp(log_weights)
        log_weights = log_weights - increment
        effective = ess(log_weights)

        def resample(_):
            ancestors = systematic(resample_key, jnp.exp(log_weights), num_particles)
            new_log_weights = jnp.full_like(log_weights, -jnp.log(num_particles))
            if unbiased_gradients:
                kept = log_weights[ancestors]
                new_log_weights = new_log_weights + kept - jax.lax.stop_gradient(kept)
            return ancestors, new_log_weights

        def keep(_):
            return jnp.arange(num_particles, dtype=jnp.int32), log_weights

        ancestors, log_weights = jax.lax.cond(
            effective < ess_threshold * num_particles, resample, keep, None
        )
        particles = take_batch(particles, ancestors)
        increment = increment.astype(state.log_likelihood.dtype)
        new_state = ParticleFilterState(
            particles,
            log_weights.astype(state.log_weights.dtype),
            t,
            state.log_likelihood + increment,
        )
        return new_state, ParticleFilterInfo(ancestors, increment, effective)

    return Kernel(init, step, lambda state, theta=None: theta)
