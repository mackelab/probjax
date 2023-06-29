import jax.numpy as jnp
from jax import random
from jax.scipy.stats import beta as jax_beta
from jax.scipy.special import gammaln, digamma

from jaxtyping import Array

from .exponential_family import ExponentialFamily
from .constraints import real, positive, unit_interval

__all__ = ["Beta"]

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Beta(ExponentialFamily):
    r"""
    Creates a beta distribution parameterized by concentration parameters `alpha` and `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Beta(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # beta distribution with alpha=2 and beta=3
        array([0.5302244], dtype=float32)

    Args:
        alpha (float or ndarray): concentration parameter alpha
        beta (float or ndarray): concentration parameter beta
    """

    arg_constraints = {"alpha": positive, "beta": positive}
    support = unit_interval

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def concentration1(self) -> Array:
        return self.alpha

    @property
    def concentration0(self) -> Array:
        return self.beta

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.beta(key, self.alpha, self.beta, shape)

    def log_prob(self, value):
        return jax_beta.logpdf(value, self.alpha, self.beta)

    def cdf(self, value):
        return jax_beta.cdf(value, self.alpha, self.beta)


    def entropy(self):
        alpha, beta = self.alpha, self.beta
        return (
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1) * digamma(alpha)
            + (beta - 1) * digamma(beta)
            - (alpha + beta - 2) * digamma(alpha + beta)
        )
