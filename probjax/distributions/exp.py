import jax.numpy as jnp
from jax import random
from jax.scipy.stats import expon as jax_expon

from jaxtyping import Array

from .exponential_family import ExponentialFamily
from .constraints import real, positive

__all__ = ["Exponential"]


from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Exponential(ExponentialFamily):
    r"""
    Creates an exponential distribution parameterized by the rate `lam`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Exponential(jnp.array([0.5]))
        >>> m.sample(key)  # exponential distribution with rate=0.5
        array([0.97394896], dtype=float32)

    Args:
        lam (float or ndarray): rate parameter lambda (inverse scale)
    """

    arg_constraints = {"lam": positive}
    support = positive

    def __init__(self, lam: Array):
        lam = jnp.asarray(lam)
        self.lam = lam

        super().__init__(batch_shape=lam.shape, event_shape=())

    @property
    def rate(self) -> Array:
        return self.lam

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.exponential(key, shape) / self.lam

    def log_prob(self, value):
        return jax_expon.logpdf(value * self.lam)
