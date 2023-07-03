import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from jax import random
from jax.lax import scan

from .distribution import Distribution
from .constraints import real, positive, unit_interval

__all__ = ["Independent"]


class Independent(Distribution):
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Base distribution object.
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    def __init__(self, base_dist: Distribution, reinterpreted_batch_ndims: int):
        if not isinstance(base_dist, Distribution):
            raise ValueError("base_dist must be an instance of ExponentialFamily")

        self.base_dist = base_dist
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        batch_shape = base_dist.batch_shape[:reinterpreted_batch_ndims]
        event_shape = (
            base_dist.batch_shape[reinterpreted_batch_ndims:] + base_dist.event_shape
        )

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def arg_constraints(self):
        return self.base_dist.arg_constraints

    @property
    def support(self):
        return self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def rsample(self, key, sample_shape=()):
        return self.base_dist.rsample(key, sample_shape)

    def sample(self, key, sample_shape=()):
        return self.base_dist.sample(key, sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)

        # Sum the log probabilities along the event dimensions
        axis = tuple(range(-self.reinterpreted_batch_ndims, 0))
        return jnp.sum(log_prob, axis=axis)

    def entropy(self):
        entropy = self.base_dist.entropy()

        # Sum the entropies along the event dimensions
        axis = tuple(range(-self.reinterpreted_batch_ndims, 0))
        return jnp.sum(entropy, axis=axis)
