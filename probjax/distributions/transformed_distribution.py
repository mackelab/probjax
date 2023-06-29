
import jax.numpy as jnp

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from jax import random
from jax.lax import scan

from .distribution import Distribution

__all__ = ["TransformedDistribution"]

class TransformedDistribution(Distribution):
    """
    Creates a transformed distribution by applying an arbitrary callable transformation
    to a base distribution.

    Args:
        base_dist: Base distribution object.
        transformation: Callable transformation that takes samples from the base distribution
            and returns transformed samples.
    """

    def __init__(self, base_dist, transformation):
        self.base_dist = base_dist
        self.transformation = transformation

        batch_shape = base_dist.batch_shape
        event_shape = base_dist.event_shape

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def arg_constraints(self):
        return self.base_dist.arg_constraints

    @property
    def support(self):
        return self.base_dist.support

    def rsample(self, key, sample_shape=()):
        samples = self.base_dist.rsample(key, sample_shape)
        return self.transformation(samples)

    def log_prob(self, value):
        transformed_value = self.transformation.inv(value)
        log_prob = self.base_dist.log_prob(transformed_value)
        return log_prob - jnp.sum(jnp.log(jnp.abs(self.transformation.log_abs_det_jacobian(value))), axis=-1)

    def cdf(self, value):
        transformed_value = self.transformation.inv(value)
        return self.base_dist.cdf(transformed_value)

    def icdf(self, value):
        transformed_value = self.base_dist.icdf(value)
        return self.transformation(transformed_value)

    def entropy(self):
        transformed_entropy = self.base_dist.entropy()
        return transformed_entropy - jnp.sum(jnp.log(jnp.abs(self.transformation.log_abs_det_jacobian(self.transformation.inv(self.base_dist.mean)))), axis=-1)
