import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution

__all__ = ["Normal"]


class Uniform(Distribution):

    arg_constraints = {"low": None, "high": None}
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        super().__init__(batch_shape=jnp.shape(low), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        return random.uniform(key, sample_shape, minval=self.low, maxval=self.high)

    def log_prob(self, value: Array) -> Array:
        return jnp.log(
            jnp.where(
                (value >= self.low) & (value <= self.high),
                1.0 / (self.high - self.low),
                0.0,
            )
        )

    def cdf(self, x: Array) -> Array:
        return jnp.where(
            x < self.low,
            0.0,
            jnp.where(x > self.high, 1.0, (x - self.low) / (self.high - self.low)),
        )

    def icdf(self, q: Array) -> Array:
        return self.low + q * (self.high - self.low)
