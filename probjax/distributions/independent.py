from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .distribution import Distribution

__all__ = ["Independent"]

from jax.tree_util import register_pytree_node_class

# Transforms a batch of independent distributions into a single mulitvariate product distribution.


@register_pytree_node_class
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
        # Determine batch_shape and event_shape using the helper function
        batch_shape, event_shape, split_dims = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )

        if isinstance(base_dist, Distribution):
            # Single distribution case
            self.base_dist = [base_dist]
        else:
            self.base_dist = base_dist

        self.split_dims = split_dims
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        # for p in self.base_dist:
        #     p._batch_shape = batch_shape
        # p._event_shape = event_shape // len(self.base_dist)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def mean(self):
        return jnp.stack([b.mean for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def median(self):
        return jnp.stack([b.median for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def mode(self):
        # The mode does change and is not equal to the mode of the base distribution
        raise NotImplementedError()

    @property
    def variance(self):
        return jnp.stack([b.variance for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    def rsample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        concat_dim = (
            -max(len(self.event_shape), 1)
            if self.reinterpreted_batch_ndims > 0
            else -(len(self.event_shape) + len(self.batch_shape))
        )
        samples = jnp.concatenate(
            [p.rsample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            axis=concat_dim,
        )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def sample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))

        concat_dim = (
            -max(len(self.event_shape), 1)
            if self.reinterpreted_batch_ndims > 0
            else -(len(self.event_shape) + len(self.batch_shape))
        )
        samples = jnp.concatenate(
            [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            axis=concat_dim,
        )

        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        if len(self.base_dist) == 1:
            log_prob = self.base_dist[0].log_prob(value)
        else:
            split_dim = (
                -max(len(self.event_shape), 1)
                if self.reinterpreted_batch_ndims > 0
                else -(len(self.event_shape) + len(self.batch_shape))
            )
            split_value = jnp.split(value, self.split_dims[:-1], axis=split_dim)
            log_prob = jnp.concatenate(
                [
                    jnp.expand_dims(b.log_prob(v), axis=split_dim)
                    for b, v in zip(self.base_dist, split_value)
                ],
                axis=split_dim,
            )

        # Sum the log probabilities along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            sum_dim = tuple(range(-self.reinterpreted_batch_ndims, 0))
            log_prob = jnp.sum(log_prob, axis=sum_dim)

            if len(self.event_shape) > 0:
                return log_prob.reshape(
                    value.shape[: -(len(self.event_shape) + len(self.batch_shape))]
                    + self.batch_shape
                )
            else:
                return log_prob.reshape(value.shape + self.batch_shape)
        else:
            if len(self.event_shape) > 0:
                return log_prob.reshape(value.shape[: -len(self.event_shape)])
            else:
                return log_prob.reshape(value.shape)

    def entropy(self):
        entropy = jnp.stack([b.entropy() for b in self.base_dist], axis=-1)

        # Sum the entropies along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                entropy,
                axis=tuple(
                    range(-self.reinterpreted_batch_ndims, -len(self.event_shape))
                ),
            )
        else:
            return entropy

    def __repr__(self) -> str:
        return f"Independent({self.base_dist}, reinterpreted_batch_ndims={self.reinterpreted_batch_ndims})"

        # Each distribution will be registered as a PyTree

    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.base_dist)
        return (
            tuple(flat_components),
            [tree_components, self.reinterpreted_batch_ndims],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components, reinterpreted_batch_ndims = aux_data
        return cls(
            jax.tree_util.tree_unflatten(tree_components, children),
            reinterpreted_batch_ndims,
        )


# Should behave as follows
# If reinterpreted_batch_ndims = 0, then:
# - (b1, e) x (b2, e) = (b1+b2, e)
# If reinterpreted_batch_ndims = 1, then:


def determine_shapes(
    base_dist: Union[Distribution, Sequence[Distribution]],
    reinterpreted_batch_ndims: int,
):
    if isinstance(base_dist, Distribution):
        # Single distribution case
        base_dist = [base_dist]

    # Extract batch shapes and event shapes from the list of base distributions
    batch_shapes = [b.batch_shape for b in base_dist]
    event_shapes = [b.event_shape for b in base_dist]

    batch_ndims = [len(b) for b in batch_shapes]
    event_ndims = [len(e) for e in event_shapes]

    assert (
        reinterpreted_batch_ndims >= 0
    ), "reinterpreted_batch_ndims must be non-negative."
    assert (
        all([b == batch_ndims[0] for b in batch_ndims])
    ), "Batch dimensions and event dimensions must be equal for all base distributions."
    assert (
        all([e == event_ndims[0] for e in event_ndims])
    ), "Batch dimensions and event dimensions must be equal for all base distributions."
    assert (
        all(reinterpreted_batch_ndims <= len(b) for b in batch_shapes)
        or all(reinterpreted_batch_ndims <= len(e) for e in event_shapes)
    ), "reinterpreted_batch_ndims must be greater than or equal to the batch shape of the base distribution."

    split_dims_batch = [b[0] if len(b) > 0 else 0 for b in batch_shapes]
    split_dims_event = [e[0] if len(e) > 0 else 0 for e in event_shapes]
    first_batch_shape_sum = sum(split_dims_batch)
    first_event_shape_sum = sum(split_dims_event)

    other_batch_shapes = [b[1:] for b in batch_shapes]
    other_event_shapes = [e[1:] for e in event_shapes]

    # other batch shapes must be equal to the other batch shapes
    assert (
        all([other_batch_shapes[0] == b for b in other_batch_shapes])
    ), "All batch shapes at index larger than 0 must be equal to the other batch shapes"
    assert (
        all([other_event_shapes[0] == e for e in other_event_shapes])
    ), "All event shapes at index larger than 0 must be equal to the other event shapes"

    # Joint batch_shape and event_shape
    if first_batch_shape_sum > 0:
        batch_shape = [first_batch_shape_sum] + list(other_batch_shapes[0])
    else:
        if reinterpreted_batch_ndims == 0:
            batch_shape = [len(batch_shapes)]
        else:
            batch_shape = []

    if first_event_shape_sum > 0:
        event_shape = [first_event_shape_sum] + list(other_event_shapes[0])
    else:
        event_shape = list(other_event_shapes[0])

    # Reinterpreted batch dimensions
    if reinterpreted_batch_ndims > 0:
        event_shape = batch_shape[-reinterpreted_batch_ndims:] + list(event_shape)
        batch_shape = batch_shape[:-reinterpreted_batch_ndims]
        split_dims = [max(s, 1) for s in split_dims_event]
    else:
        event_shape = event_shapes[0]
        split_dims = [max(s, 1) for s in split_dims_batch]

    # Cummulatively sum the split_dims
    split_dims = list(np.cumsum(split_dims))

    return tuple(batch_shape), tuple(event_shape), tuple(split_dims)
