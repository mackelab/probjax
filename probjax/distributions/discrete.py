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


@register_pytree_node_class
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
        value = jax.nn.one_hot(value, self.probs.shape[-1])
        log_probs = jax.scipy.special.xlogy(value, self.probs).sum(axis=-1)
        return log_probs

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

        # Reinterpret the values as a batch of independent distributions
        self.num_values = self.values.shape[0]
        # Rest is interpreted as batch shape
        if values.ndim == 1:
            batch_shape = ()
            event_shape = ()
        else:
            batch_shape = self.values.shape[1:]
            event_shape = ()

        if probs is None:
            self.probs = None
        else:
            #assert probs.shape == values.shape, "probs shape mismatch"
            self.probs = jnp.asarray(probs)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        base_index = jnp.arange(0, self.num_values)
        base_index = jnp.broadcast_to(base_index, self.probs.shape)
        index = random.choice(
            key, base_index, shape=shape, p=self.probs
        )
        samples = jnp.take_along_axis(self.values, index, axis=0)
        return samples

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value)
        mask = jnp.equal(value[:, None], self.values)
        indices = jnp.argmax(mask, axis=-self.values.ndim)
        valid = jnp.any(mask, axis=-self.values.ndim)
        if self.probs is not None:
            probs, indices = jnp.broadcast_arrays(self.probs, indices)
            log_probs = jnp.take_along_axis(jnp.log(probs), indices, axis=-1)
            log_probs = jnp.where(valid, log_probs, -jnp.inf)
        else:
            log_probs = jnp.where(valid, -jnp.log(self.num_values), -jnp.inf)
        return log_probs

    @property
    def mean(self) -> Array:
        return jnp.sum(self.values * self.probs)
    
    @property
    def mode(self) -> Array:
        return self.values[jnp.argmax(self.probs)]

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
