from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from probjax.inference.filtering.base import FilterAPI, as_time
from probjax.inference.particles import normalize_log_weights
from probjax.inference.smc.ess import ess
from probjax.inference.smc.resampling import (
    resample_systematic,
)
from probjax.utils.jaxutils import leading_dim
from probjax.utils.typing import Array, ArrayLike, RngKey


class ParticleFilterState(NamedTuple):
    particles: Any
    log_weights: Array
    t: Optional[Array]


class ParticleFilterInfo(NamedTuple):
    ancestors: Array
    log_likelihood: Array
    ess: Array
    is_observed: bool


def init(
    particles: Any,
    t: Optional[ArrayLike] = None,
    log_weights: Optional[ArrayLike] = None,
) -> ParticleFilterState:
    # Particles may be a pytree with a leading particle axis.
    num_particles = leading_dim(particles)
    if log_weights is None:
        weights = jnp.full((num_particles,), fill_value=-jnp.log(num_particles))
    else:
        weights = jnp.asarray(log_weights)
        assert weights.shape[0] == num_particles, (
            "The number of particles and log weights must be the same."
        )
        weights = normalize_log_weights(weights)
    return ParticleFilterState(particles, weights, as_time(t))


def resample_when_ess_below(ess: float, ess_threshold: float = 0.5) -> bool:
    return ess < ess_threshold


def build_kernel(
    log_likelihood_fn: Callable,
    transition_fn: Callable,
    transition_logdensity_fn: Optional[Callable] = None,
    proposal_transition_fn: Optional[Callable] = None,
    proposal_logdensity_fn: Optional[Callable] = None,
    resample_criterion: Callable = resample_when_ess_below,
    resample_fn: Callable = resample_systematic,
    unbiased_gradients: bool = False,
):
    """Build a particle filter kernel.

    Args:
        log_likelihood_fn (Callable): Log likelihood function of the model.
        transition_fn (Callable): Transition function p(x_t|x_{t-1}) of the model.
        transition_logdensity_fn (Optional[Callable]): Computes logdensity function of
            the transition function. Defaults to None.
        proposal_transition_fn (Optional[Callable]): Transition based on a proposal.
            Defaults to None.
        proposal_logdensity_fn (Optional[Callable]): Proposal density function.
            Defaults to None.
        resample_criterion (Callable): Criterion to decide when to resample. Defaults
            to resample_when_ess_below.
        resample_fn (Callable): Resampling function. Defaults to resample_systematic.
        unbiased_gradients (bool): Whether to use unbiased gradients. Defaults to
            False.
    """

    def kernel(
        state: ParticleFilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[RngKey] = None,
    ) -> tuple[ParticleFilterState, ParticleFilterInfo]:
        assert rng_key is not None, (
            "You must provide a random key for the particle filter kernel."
        )
        # Unpack state
        particles = state.particles
        log_weights = state.log_weights
        rng_key, rng_key_predict, rng_key_resample = jax.random.split(rng_key, 3)

        # Predict new particles
        if proposal_logdensity_fn is None:
            new_particles = transition_fn(rng_key_predict, particles, t)
        else:
            assert proposal_transition_fn is not None, (
                "A proposal transition is required alongside the proposal density."
            )
            new_particles = proposal_transition_fn(rng_key_predict, particles, t)

        # Update weights
        is_observed = observed is not None
        if is_observed:
            # Update step
            log_weights = log_likelihood_fn(new_particles, observed, t) + log_weights
            if (
                transition_logdensity_fn is not None
                and proposal_logdensity_fn is not None
            ):
                log_weights = log_weights + (
                    transition_logdensity_fn(new_particles, particles, t)
                    - proposal_logdensity_fn(new_particles, particles, t)
                )
            log_normalizer = jax.scipy.special.logsumexp(log_weights)
            log_weights = log_weights - log_normalizer
            # Incoming weights sum to one: logsumexp already estimates p(y_t|y_<t).
            log_likelihood = log_normalizer

        else:
            if (
                transition_logdensity_fn is not None
                and proposal_logdensity_fn is not None
            ):
                log_weights += transition_logdensity_fn(
                    new_particles, particles, t
                ) - proposal_logdensity_fn(new_particles, particles, t)
                log_normalizer = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_normalizer
                log_likelihood = log_normalizer
            else:
                log_likelihood = jnp.array(0.0)

        # Resample if necessary
        effective_samples_size = ess(log_weights)
        do_resample = resample_criterion(effective_samples_size / log_weights.shape[0])

        def resample(key, log_weights, particles):
            new_particles, new_log_weights, idx = resample_fn(
                key, log_weights, particles
            )
            if unbiased_gradients:
                new_log_weights += log_weights[idx] - jax.lax.stop_gradient(
                    log_weights[idx]
                )

            return new_particles, new_log_weights, idx

        def no_resample(key, log_weights, particles):
            return (
                particles,
                log_weights,
                jnp.arange(log_weights.shape[0], dtype=jnp.int32),
            )

        new_particles, new_log_weights, ancestors = jax.lax.cond(
            do_resample,
            resample,
            no_resample,
            rng_key_resample,
            log_weights,
            new_particles,
        )

        new_state = ParticleFilterState(new_particles, new_log_weights, as_time(t))
        info = ParticleFilterInfo(
            ancestors, log_likelihood, effective_samples_size, is_observed
        )

        return new_state, info

    return kernel


class ParticleFilter(FilterAPI):
    """Particle filter inference algorithm.

    This class implements the particle filter algorithm. The particle filter is a
    sequential Monte Carlo method that approximates the filtering distribution of a
    state-space model. The particle filter is a generalization of the Kalman filter
    to non-linear and non-Gaussian models.

    To build a particle filter, you need to provide the following functions:
    Args:
        log_likelihood_fn (Callable): Log likelihood function of the model p(y_t|x_t).
        transition_fn (Callable): Transition function p(x_t|x_{t-1}) of the model.
        transition_logdensity_fn (Optional[Callable]): Computes logdensity function of
            the transition function. Defaults to None.
        proposal_transition_fn (Optional[Callable]): Transition based on a proposal.
            Defaults to None.
        proposal_logdensity_fn (Optional[Callable]): Proposal density function.
            Defaults to None.
        resample_criterion (Callable): Criterion to decide when to resample. Defaults
            to resample_when_ess_below.
        resample_fn (Callable): Resampling function. Defaults to resample_systematic.
        unbiased_gradients (bool): Whether to use unbiased gradients. Defaults to
            False.
    """

    init = init
    build_kernel = build_kernel

    @staticmethod
    def default_unpack(state: ParticleFilterState, info: ParticleFilterInfo) -> Any:
        return state.particles
