import blackjax

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt

from blackjax.smc.resampling import systematic, multinomial, residual
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
    # Initialize state for a particle filter
    log_weights = jax.tree_map(jnp.zeros_like, particles)
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
    resample_fn: Callable = systematic,
    unbiased_gradients: bool = True,
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
        rng_key,
        state: ParticleFilterState,
        t: Optional[float | int] = None,
        observed: Optional[ArrayLike] = None,
    ):
        # Unpack state
        particles = state.particles
        log_weights = state.log_weights
        rng_key, rng_key_predict, rng_key_resample = jax.random.split(rng_key, 3)
        
        # Predict
        if proposal_logdensity_fn is None:
            new_particles = transition_fn(rng_key_predict, particles, t)
        else:
            new_particles = proposal_transition_fn(rng_key_predict, particles, t)
        is_observed = observed is not None
 
        if is_observed:
            # Update step
            log_weights = log_likelihood_fn(new_particles, observed, t) + log_weights
            if transition_logdensity_fn is not None and proposal_logdensity_fn is not None:
                log_weights = log_weights + (transition_logdensity_fn(new_particles, particles, t) - proposal_logdensity_fn(new_particles, particles,t))
            logZ = jax.scipy.special.logsumexp(log_weights)
            log_weights = log_weights - logZ
            effective_samples_size = ess(log_weights)
            do_resample = resample_criterion(effective_samples_size/log_weights.shape[0])
            
            def resample(key, log_weights, particles):
                idx = resample_fn(key, jnp.exp(log_weights), particles.shape[0])
                new_particles = particles[idx]
                if unbiased_gradients:
                    new_log_weights = jnp.zeros_like(log_weights)  + (log_weights[idx] - jax.lax.stop_gradient(log_weights[idx]))
                else:
                    new_log_weights = jnp.zeros_like(log_weights)
                return new_particles, new_log_weights, idx
            
            def no_resample(key, log_weights, particles):
                return particles, log_weights, jnp.arange(particles.shape[0])
            
            new_particles, new_log_weights, ancestors = jax.lax.cond(do_resample, resample, no_resample, rng_key_resample, log_weights, new_particles)
        else:
            new_log_weights = log_weights
            if transition_logdensity_fn is not None and proposal_logdensity_fn is not None:
                new_log_weights = new_log_weights + (transition_logdensity_fn(new_particles, particles,t) - proposal_logdensity_fn(new_particles, particles,t))
            logZ = 0.
            ancestors = jnp.arange(particles.shape[0])
            effective_samples_size = None 
            
        
        new_state = ParticleFilterState(new_particles, new_log_weights)
        info = ParticleFilterInfo(ancestors, logZ, effective_samples_size, t, is_observed)
        
        return new_state, info
    
    return kernel

