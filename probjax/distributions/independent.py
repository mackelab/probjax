import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from jax import random
from jax.lax import scan

from typing import Sequence, Union

from .distribution import Distribution
from .constraints import real, positive, unit_interval

__all__ = ["Independent"]

# Transforms a batch of independent distributions into a single mulitvariate product distribution.


class Independent(Distribution):
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Base distribution object.
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    def __init__(
        self,
        base_dist: Union[Distribution, Sequence[Distribution]],
        reinterpreted_batch_ndims: int,
    ):
        if isinstance(base_dist, Distribution):
            self.base_dist = [base_dist]
            self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

            batch_shape = base_dist.batch_shape[:-reinterpreted_batch_ndims]
            event_shape = (
                base_dist.batch_shape[-reinterpreted_batch_ndims:]
                + base_dist.event_shape
            )
        else:
            batch_shapes = [b.batch_shape for b in base_dist]
            event_shapes = [b.event_shape for b in base_dist]
            assert all(
                b == batch_shapes[0] for b in batch_shapes
            ), "Batch shapes must be equal"
            assert all(
                e == event_shapes[0] for e in event_shapes
            ), "Event shapes must be equl"

            batch_shape = batch_shapes[0][:-reinterpreted_batch_ndims]
            event_shape = batch_shapes[0][-reinterpreted_batch_ndims:] + event_shapes[0]

        self.reduce_axis = tuple(range(-self.reinterpreted_batch_ndims, 0))

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def mean(self):
        return jnp.concatenate([b.mean for b in self.base_dist], axis=-1)

    @property
    def median(self):
        return jnp.concatenate([b.median for b in self.base_dist], axis=-1)

    @property
    def mode(self):
        return jnp.concatenate([b.mode for b in self.base_dist], axis=-1)

    @property
    def variance(self):
        return jnp.concatenate([b.variance for b in self.base_dist], axis=-1)

    def rsample(self, key, sample_shape=()):
        return jnp.concatenate(
            [b.rsample(key, sample_shape) for b in self.base_dist],
            axis=-1,
        )

    def sample(self, key, sample_shape=()):
        return jnp.concatenate(
            [b.sample(key, sample_shape) for b in self.base_dist], axis=-1
        )

    def log_prob(self, value):
        log_prob = jnp.concatenate([b.log_prob(value) for b in self.base_dist], axis=-1)

        # Sum the log probabilities along the event dimensions
        return jnp.sum(log_prob, axis=self.reduce_axis)

    def entropy(self):
        entropy = jnp.concatenate([b.entropy() for b in self.base_dist], axis=-1)

        # Sum the entropies along the event dimensions
        return jnp.sum(entropy, axis=self.reduce_axis)
