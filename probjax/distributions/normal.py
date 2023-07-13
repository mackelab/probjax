import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .exponential_family import ExponentialFamily
from .constraints import real, positive, unit_interval

__all__ = ["Normal"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import norm


@register_pytree_node_class
class Normal(ExponentialFamily):
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

    arg_constraints = {"loc": real, "scale": positive}
    support = real

    def __init__(self, loc: Array | float, scale: Array | float):
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
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        return jnp.squeeze(norm.logpdf(value, self.loc, self.scale))

    def cdf(self, value):
        return norm.cdf(value, self.loc, self.scale)

    def icdf(self, value):
        return norm.ppf(value, self.loc, self.scale)

    def entropy(self):
        return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.scale)
