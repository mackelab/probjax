import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution
from .exponential_family import ExponentialFamily
from .constraints import (
    finit_set,
    simplex,
    real,
    unit_interval,
    unit_integer_interval,
    positive_integer,
)

from jax.scipy.stats import bernoulli, binom, poisson, geom, multinomial

__all__ = [
    "Empirical",
    "Dirac",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Geometric",
    "Categorical",
]

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Bernoulli(ExponentialFamily):
    arg_constraints = {"probs": unit_interval}
    support = unit_integer_interval

    def __init__(self, probs: Array):
        self.probs = jnp.asarray(probs)
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


class Categorical(ExponentialFamily):
    arg_constraints = {"probs": simplex}

    def __init__(self, probs: Array):
        self.probs = jax.nn.softmax(probs)
        shape = self.probs.shape
        if len(shape) > 1:
            batch_shape = shape[:-1]
            event_shape = ()
        else:
            batch_shape = ()
            event_shape = ()

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.categorical(key, self.probs, shape=shape, axis=-1)

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value).astype(jnp.int32)
        value = value[..., None]
        probs, value = jnp.broadcast_arrays(self.probs, value)
        return jnp.squeeze(jnp.log(jnp.take_along_axis(probs, value, axis=-1)), -1)

    @property
    def mean(self) -> Array:
        return jnp.sum(self.probs * jnp.arange(self.probs.shape[-1]), axis=-1)

    @property
    def variance(self) -> Array:
        return jnp.sum(
            self.probs * (jnp.arange(self.probs.shape[-1]) - self.mean) ** 2, axis=-1
        )

    @property
    def entropy(self) -> Array:
        return -jnp.sum(self.probs * jnp.log(self.probs), axis=-1)


@register_pytree_node_class
class Binomial(ExponentialFamily):
    arg_constraints = {"n": positive_integer, "probs": unit_interval}

    def __init__(self, n: int, probs: Array):
        self.n = int(n)
        self.probs = probs
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape + (self.n,)
        return random.bernoulli(key, self.probs, shape=shape).sum(axis=-1)

    def log_prob(self, value: Array) -> Array:
        return binom.logpmf(value, self.n, self.probs)

    @property
    def mean(self) -> Array:
        return self.n * self.probs

    @property
    def variance(self) -> Array:
        return self.n * self.probs * (1 - self.probs)

    @property
    def entropy(self) -> Array:
        return (
            jnp.log(2)
            - self.probs * jnp.log(self.probs)
            - (1 - self.probs) * jnp.log(1 - self.probs)
        )


@register_pytree_node_class
class Poisson(ExponentialFamily):
    arg_constraints = {"rate": positive_integer}

    def __init__(self, rate: Array):
        self.rate = rate
        super().__init__(batch_shape=rate.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.poisson(key, self.rate, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return poisson.logpmf(value, self.rate)

    @property
    def mean(self) -> Array:
        return self.rate

    @property
    def variance(self) -> Array:
        return self.rate

    @property
    def entropy(self) -> Array:
        return self.rate * (1 - jnp.log(self.rate))

    def cdf(self, value: Array) -> Array:
        return poisson.cdf(value, self.rate)

    def icdf(self, value: Array) -> Array:
        return poisson.ppf(value, self.rate)


@register_pytree_node_class
class Geometric(ExponentialFamily):
    arg_constraints = {"probs": unit_interval}

    def __init__(self, probs: Array):
        self.probs = probs
        super().__init__(batch_shape=probs.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.geometric(key, self.probs, shape=shape)

    def log_prob(self, value: Array) -> Array:
        return geom.logpmf(value, self.probs)

    @property
    def mean(self) -> Array:
        return (1 - self.probs) / self.probs

    @property
    def variance(self) -> Array:
        return (1 - self.probs) / self.probs**2

    @property
    def entropy(self) -> Array:
        return -self.probs * jnp.log(self.probs) - (1 - self.probs) * jnp.log(
            1 - self.probs
        )

    def cdf(self, value: Array) -> Array:
        return geom.cdf(value, self.probs)

    def icdf(self, value: Array) -> Array:
        return geom.ppf(value, self.probs)


# Dirac delta distribution
@register_pytree_node_class
class Dirac(Distribution):
    arg_constraints = {"value": real}

    def __init__(self, value: Array):
        self.value = value
        self.support = finit_set(self.value)
        super().__init__(batch_shape=value.shape, event_shape=())

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return jnp.broadcast_to(self.value, shape)

    def log_prob(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value == true_value, 0.0, -jnp.inf)

    @property
    def mean(self) -> Array:
        return self.value

    @property
    def variance(self) -> Array:
        return jnp.zeros(self.batch_shape)

    @property
    def entropy(self) -> Array:
        return jnp.zeros(self.batch_shape)

    def cdf(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value >= true_value, 1.0, 0.0)

    def icdf(self, value: Array) -> Array:
        true_value = jnp.broadcast_to(self.value, value.shape)
        return jnp.where(value >= 1.0, true_value, jnp.inf)


# Empirical distribution
@register_pytree_node_class
class Empirical(Distribution):
    arg_constraints = {"values": real, "probs": simplex}

    def __init__(self, values: Array, probs: Array | None = None):
        self.values = jnp.asarray(values)
        self.support = finit_set(self.values)

        self.num_values = self.values.shape[0]
        batch_shape = ()
        event_shape = self.values.shape[1:]

        if probs is None:
            self.probs = None
        else:
            assert probs.shape == values.shape
            self.probs = jnp.asarray(probs)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.choice(key, self.values, shape=shape, p=self.probs)

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value)
        value = value[..., None]
        mask = jnp.equal(value, self.values)
        log_probs = jnp.where(mask, jnp.log(self.probs), -jnp.inf)
        return jnp.sum(log_probs, axis=-1)

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
