import jax.numpy as jnp
import numpy as np
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution
from .exponential_family import ExponentialFamily
from .constraints import finit_set, simplex, real, unit_interval, unit_integer_interval

from jax.scipy.stats import bernoulli

__all__ = ["Discrete", "Bernoulli"]

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Bernoulli(ExponentialFamily):
    arg_constraints = {"probs": unit_interval}
    support = unit_integer_interval

    def __init__(self, probs: Array):
        self.probs = probs
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(key, self.probs, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return bernoulli.logpmf(value, self.probs)

    @property
    def mean(self) -> Array:
        return self.probs

    @property
    def variance(self) -> Array:
        return self.probs * (1 - self.probs)

    @property
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


@register_pytree_node_class
class Discrete(Distribution):
    arg_constraints = {"values": real, "probs": simplex}

    def __init__(self, values: Array, probs: Array | None = None):
        self.values = jnp.asarray(values)
        self.support = finit_set(self.values)

        if probs is None:
            self.probs = jnp.ones_like(values) / np.prod(values.shape)
        else:
            self.probs = jnp.asarray(probs)

        super().__init__(batch_shape=values.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.choice(key, self.values, p=self.probs, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return jnp.where(value == self.values, jnp.log(self.probs), -jnp.inf)

    @property
    def mean(self) -> Array:
        return jnp.sum(self.values * self.probs)

    @property
    def variance(self) -> Array:
        m = self.mean
        return jnp.sum((self.values - m) ** 2 * self.probs)

    @property
    def entropy(self) -> Array:
        return -jnp.sum(self.probs * jnp.log(self.probs))

    def cdf(self, value: Array) -> Array:
        return jnp.cumsum(self.probs)

    def icdf(self, value: Array) -> Array:
        return jnp.searchsorted(self.cdf(self.values), value)
