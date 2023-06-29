import jax.numpy as jnp
from jax import random
from jax.scipy.stats import gamma as jax_gamma

from jaxtyping import Array

from .exponential_family import ExponentialFamily
from .constraints import real, positive

__all__ = ["Gamma"]


from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Gamma(ExponentialFamily):
    r"""
    Creates a gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Gamma(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # gamma distribution with shape=2 and rate=3
        array([1.3750159], dtype=float32)

    Args:
        alpha (float or ndarray): shape parameter alpha
        beta (float or ndarray): rate parameter beta
    """

    arg_constraints = {"alpha": positive, "beta": positive}
    support = positive

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def concentration(self) -> Array:
        return self.alpha

    @property
    def rate(self) -> Array:
        return self.beta

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(key, self.alpha, shape) / self.beta

    def log_prob(self, value):
        return jax_gamma.logpdf(value * self.beta, self.alpha)

    def cdf(self, value):
        return jax_gamma.cdf(value * self.beta, self.alpha)
