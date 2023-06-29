import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution
from .constraints import real, positive, unit_interval, interval

__all__ = ["TruncatedNormal"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import truncnorm


@register_pytree_node_class
class TruncatedNormal(Distribution):
    arg_constraints = {"loc": real, "scale": positive, "a": real, "b": real}

    def __init__(self, loc: Array, scale: Array, a: Array, b: Array):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        self.loc, self.scale, self.a, self.b = jnp.broadcast_arrays(loc, scale, a, b)

    def log_prob(self, value: Array) -> Array:
        return truncnorm.logpdf(value, self.a, self.b, loc=self.loc, scale=self.scale)
    
    def sample(self, key: Array, sample_shape: Array = ()) -> Array:
        standardize_lower = (self.a - self.loc) / self.scale
        standardize_upper = (self.b - self.loc) / self.scale
        return random.truncated_normal(key, standardize_lower, standardize_upper, shape=sample_shape) * self.scale + self.loc
    

    def cdf(self, value: Array) -> Array:
        return truncnorm.cdf(value, self.a, self.b, loc=self.loc, scale=self.scale)
