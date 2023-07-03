import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .exponential_family import ExponentialFamily
from .constraints import real, positive, unit_interval, unit_integer_interval

__all__ = ["Bernoulli"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import bernoulli


@register_pytree_node_class
class Bernoulli(ExponentialFamily):
    
    arg_constraints = {"probs": unit_interval}
    support = unit_integer_interval

    def __init__(self, probs: Array):
        self.probs = probs

    def sample(self, key, sample_shape=()):
        return random.bernoulli(key, self.probs, shape=sample_shape)

    def log_prob(self, value: Array) -> Array:
        return bernoulli.logpmf(value, self.probs)

    def mean(self) -> Array:
        return self.probs

    def variance(self) -> Array:
        return self.probs * (1 - self.probs)

    def entropy(self) -> Array:
        return (
            jnp.log(2)
            - self.probs * jnp.log(self.probs)
            - (1 - self.probs) * jnp.log(1 - self.probs)
        )

    def cdf(self, value: Array) -> Array:
        return bernoulli.cdf(value, self.probs)

    def icdf(self, value: Array) -> Array:
        return bernoulli.ppf(value, self.probs)
