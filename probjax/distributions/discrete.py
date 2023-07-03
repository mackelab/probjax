import jax.numpy as jnp
import numpy as np
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .constraints import finit_set, simplex, real

__all__ = ["Discrete"]

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Discrete:
    arg_constraints = {"values": real, "probs": simplex}

    def __init__(self, values: Array, probs: Array | None = None):
        self.values = jnp.asarray(values)
        self.support = finit_set(self.values)

        if probs is None:
            self.probs = jnp.ones_like(values) / np.prod(values.shape)
        else:
            self.probs = jnp.asarray(probs)

    def sample(self, key, sample_shape=()):
        return random.choice(key, self.values, p=self.probs, shape=sample_shape)

    def log_prob(self, value: Array) -> Array:
        return jnp.where(value == self.values, jnp.log(self.probs), -jnp.inf)

    def mean(self) -> Array:
        return jnp.sum(self.values * self.probs)

    def variance(self) -> Array:
        m = self.mean()
        return jnp.sum((self.values - m) ** 2 * self.probs)

    def entropy(self) -> Array:
        return -jnp.sum(self.probs * jnp.log(self.probs))

    def cdf(self, value: Array) -> Array:
        return jnp.cumsum(self.probs)

    def icdf(self, value: Array) -> Array:
        return jnp.searchsorted(self.cdf(self.values), value)
