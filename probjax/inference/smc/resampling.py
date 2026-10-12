import jax.numpy as jnp
from blackjax.smc.resampling import systematic


def resample_systematic(key, log_weights, particles):
    weights = jnp.exp(log_weights)
    idx = systematic(key, weights, log_weights.shape[0])
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return particles[idx], new_log_weights, idx
