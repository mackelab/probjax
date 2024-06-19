import blackjax

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.flatten_util import ravel_pytree


from probjax.inference.filtering.base import FilterState, FilterInfo, FilterKernel
from probjax.inference.smc.resampling import resample_systematic, resample_multinomial, resample_residual, resample_ot
from blackjax.smc.ess import ess


class ParticleFilterState(NamedTuple):
    particles: ArrayLike
    log_weights: ArrayLike

class ParticleFilterInfo(NamedTuple):
    ancestors: ArrayLike
    logZ: ArrayLike
    ess: ArrayLike
    t: int | float | ArrayLike
    is_observed: bool


def init(particles: ArrayLike) -> ParticleFilterState:
    print(particles.shape)
    # Initialize state for a particle filter
    num_particles = particles.shape[0]
    log_weights = jax.tree_map(lambda x: jnp.full_like(x, fill_value=-jnp.log(num_particles)), particles)
    return ParticleFilterState(particles, log_weights)

def resample_when_ess_below(ess: float,ess_threshold: float= 0.5):
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
    """ Build a particle filter kernel.

    Args:
        log_likelihood_fn (Callable): Log likelihood function of the model.
        transition_fn (Callable): Transition function p(x_t|x_{t-1}) of the model.
        transition_logdensity_fn (Optional[Callable], optional): Computes logdensity function of the transition function. Defaults to None.
        proposal_transition_fn (Optional[Callable], optional): Transition based on a proposal. Defaults to None.
        proposal_logdensity_fn (Optional[Callable], optional): Proposal density_fn. Defaults to None.
        resample_criterion (Callable, optional): _description_. Defaults to resample_when_ess_below.
        resample_fn (Callable, optional): _description_. Defaults to systematic.
        unbiased_gradients (bool, optional): _description_. Defaults to True.
    """

    def kernel(
        state: ParticleFilterState,
        t: Optional[float | int] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[ArrayLike] = None,
    ):
        assert rng_key is not None, "You must provide a random key for the particle filter kernel."
        # Unpack state
        particles = state.particles
        log_weights = state.log_weights
        rng_key, rng_key_predict, rng_key_resample = jax.random.split(rng_key, 3)
        log_num_particles = jnp.log(particles.shape[0])

        # Predict new particles
        if proposal_logdensity_fn is None:
            new_particles = transition_fn(rng_key_predict, particles, t)
        else:
            new_particles = proposal_transition_fn(rng_key_predict, particles, t)

        # Update weights
        is_observed = observed is not None
        if is_observed:
            # Update step
            log_weights = log_likelihood_fn(new_particles, observed, t) + log_weights
            if transition_logdensity_fn is not None and proposal_logdensity_fn is not None:
                log_weights = log_weights + (transition_logdensity_fn(new_particles, particles, t) - proposal_logdensity_fn(new_particles, particles,t))
            log_normalizer = jax.scipy.special.logsumexp(log_weights)
            log_weights = log_weights - log_normalizer
            logZ = log_normalizer - log_num_particles

        else:
            if transition_logdensity_fn is not None and proposal_logdensity_fn is not None:
                log_weights += (transition_logdensity_fn(new_particles, particles,t) - proposal_logdensity_fn(new_particles, particles,t))
                log_normalizer = jax.scipy.special.logsumexp(log_weights)
                log_weights = log_weights - log_normalizer
            else:
                new_log_weights = log_weights
            logZ = 0. # Without observation we don't have a logZ

        # Resample if necessary
        effective_samples_size = ess(log_weights)
        do_resample = resample_criterion(effective_samples_size/log_weights.shape[0])

        def resample(key, log_weights, particles):
            new_particles, new_log_weights, idx = resample_fn(key, log_weights, particles)
            if unbiased_gradients:
                new_log_weights += (log_weights[idx] - jax.lax.stop_gradient(log_weights[idx]))

            return new_particles, new_log_weights, idx

        def no_resample(key, log_weights, particles):
            return particles, log_weights, jnp.arange(particles.shape[0])

        new_particles, new_log_weights, ancestors = jax.lax.cond(do_resample, resample, no_resample, rng_key_resample, log_weights, new_particles)

        new_state = ParticleFilterState(new_particles, new_log_weights)
        info = ParticleFilterInfo(ancestors, logZ, effective_samples_size, t, is_observed)

        return new_state, info

    return kernel


class ParticleFilter(FilterKernel):
    
    init_fn = staticmethod(init)
    build_kernel = staticmethod(build_kernel)
    
    def __init__(self, log_likelihood_fn: Callable, transition_fn: Callable, transition_logdensity_fn: Optional[Callable] = None, proposal_transition_fn: Optional[Callable] = None, proposal_logdensity_fn: Optional[Callable] = None, resample_criterion: Callable = resample_when_ess_below, resample_fn: Callable = resample_systematic, unbiased_gradients: bool = False):
        self._kernel = build_kernel(
            log_likelihood_fn,
            transition_fn,
            transition_logdensity_fn,
            proposal_transition_fn,
            proposal_logdensity_fn,
            resample_criterion,
            resample_fn,
            unbiased_gradients
        )
