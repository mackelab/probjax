import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution

__all__ = ["Normal"]


class Normal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        scale (float or ndarray): standard deviation of the distribution
            (often referred to as sigma)
    """

    arg_constraints = {"loc": None, "scale": None}

    def __init__(self, loc: Array, scale: Array):

        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        self.loc, self.scale = jnp.broadcast_arrays(loc, scale)

        super().__init__(batch_shape=loc.shape, event_shape=())

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def stddev(self) -> Array:
        return self.scale

    @property
    def variance(self) -> Array:
        return jnp.power(self.stddev, 2)

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.loc.shape
        eps = random.normal(key, shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        # compute the variance
        var = self.scale**2
        log_scale = jnp.log(self.scale)
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - jnp.log(jnp.sqrt(2 * jnp.pi))
        )

    def cdf(self, value):
        return 0.5 * (
            1 + erf((value - self.loc) * self.scale.reciprocal() / jnp.sqrt(2))
        )

    def icdf(self, value):
        return self.loc + self.scale * erfinv(2 * value - 1) * jnp.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.scale)
