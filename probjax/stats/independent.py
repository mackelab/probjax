from typing import Sequence, Union, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, PRNGKeyArray, ArrayLike

from .base import rv_generic, rv_frozen
from .constraints import Constraint, distribution, non_negative_integer

__all__ = ["independent"]


def determine_shapes(
    base_dist: Union[rv_frozen, Sequence[rv_frozen]],
    reinterpreted_batch_ndims: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Helper function to determine shapes for Independent distribution."""
    if isinstance(base_dist, rv_frozen):
        # Single distribution case
        base_dist = [base_dist]

    # Extract batch shapes and event shapes from the list of base distributions
    batch_shapes = [b.batch_shape for b in base_dist]
    event_shapes = [b.event_shape for b in base_dist]

    batch_ndims = [len(b) for b in batch_shapes]
    event_ndims = [len(e) for e in event_shapes]

    assert reinterpreted_batch_ndims >= 0, (
        "reinterpreted_batch_ndims must be non-negative."
    )
    assert all([b == batch_ndims[0] for b in batch_ndims]), (
        "Batch dimensions and event dimensions must be equal for all base distributions."
    )
    assert all([e == event_ndims[0] for e in event_ndims]), (
        "Batch dimensions and event dimensions must be equal for all base distributions."
    )
    assert all(reinterpreted_batch_ndims <= len(b) for b in batch_shapes) or all(
        reinterpreted_batch_ndims <= len(e) for e in event_shapes
    ), (
        "reinterpreted_batch_ndims must be greater than or equal to the batch shape of"
        "the base distribution."
    )

    split_dims_batch = [b[0] if len(b) > 0 else 0 for b in batch_shapes]
    split_dims_event = [e[0] if len(e) > 0 else 0 for e in event_shapes]
    first_batch_shape_sum = sum(split_dims_batch)
    first_event_shape_sum = sum(split_dims_event)

    other_batch_shapes = [b[1:] for b in batch_shapes]
    other_event_shapes = [e[1:] for e in event_shapes]

    # other batch shapes must be equal to the other batch shapes
    assert all([other_batch_shapes[0] == b for b in other_batch_shapes]), (
        "All batch shapes at index larger than 0 must be equal to the other batch shapes"
    )
    assert all([other_event_shapes[0] == e for e in other_event_shapes]), (
        "All event shapes at index larger than 0 must be equal to the other event shapes"
    )

    # Joint batch_shape and event_shape
    if first_batch_shape_sum > 0:
        batch_shape = [first_batch_shape_sum] + list(other_batch_shapes[0])
    else:
        batch_shape = [len(batch_shapes)] if reinterpreted_batch_ndims == 0 else []

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


class independent(rv_generic):
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Frozen base distribution object(s).
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    parameters = {
        "base_dist": distribution,
        "reinterpreted_batch_ndims": non_negative_integer,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def support(
        cls,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Support of the distribution."""
        if isinstance(base_dist, rv_frozen):
            return base_dist.support()
        else:
            return tuple(d.support() for d in base_dist)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...],
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Random variates of given shape."""
        if isinstance(base_dist, rv_frozen):
            base_dist = [base_dist]

        keys = random.split(rng, len(base_dist))
        concat_dim = (
            -max(len(shape), 1)
            if reinterpreted_batch_ndims > 0
            else -(len(shape) + len(base_dist[0].batch_shape))
        )
        samples = jnp.concatenate(
            [d.rvs(k, shape) for k, d in zip(keys, base_dist)],
            axis=concat_dim,
        )
        return samples

    @classmethod
    def logpdf(
        cls,
        x: ArrayLike,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Log of the probability density function at x of the given RV."""
        if isinstance(base_dist, rv_frozen):
            log_prob = base_dist.logpdf(x)
        else:
            split_dim = (
                -max(len(x.shape), 1)
                if reinterpreted_batch_ndims > 0
                else -(len(x.shape) + len(base_dist[0].batch_shape))
            )
            split_value = jnp.split(
                x, [d.event_shape[0] for d in base_dist[:-1]], axis=split_dim
            )
            log_prob = jnp.concatenate(
                [
                    jnp.expand_dims(d.logpdf(v), axis=split_dim)
                    for d, v in zip(base_dist, split_value)
                ],
                axis=split_dim,
            )

        # Sum the log probabilities along the event dimensions
        if reinterpreted_batch_ndims > 0:
            sum_dim = tuple(range(-reinterpreted_batch_ndims, 0))
            log_prob = jnp.sum(log_prob, axis=sum_dim)

        return log_prob

    @classmethod
    def pdf(
        cls,
        x: ArrayLike,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Probability density function at x of the given RV."""
        return jnp.exp(cls.logpdf(x, base_dist, reinterpreted_batch_ndims))

    @classmethod
    def mean(
        cls,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Mean of the distribution."""
        if isinstance(base_dist, rv_frozen):
            base_dist = [base_dist]

        means = jnp.stack([d.mean() for d in base_dist], axis=-1)
        batch_shape, event_shape, _ = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )
        return means.reshape(batch_shape + event_shape)

    @classmethod
    def var(
        cls,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Variance of the distribution."""
        if isinstance(base_dist, rv_frozen):
            base_dist = [base_dist]

        variances = jnp.stack([d.var() for d in base_dist], axis=-1)
        batch_shape, event_shape, _ = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )
        return variances.reshape(batch_shape + event_shape)

    @classmethod
    def entropy(
        cls,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Entropy of the RV."""
        if isinstance(base_dist, rv_frozen):
            base_dist = [base_dist]

        entropies = jnp.stack([d.entropy() for d in base_dist], axis=-1)
        if reinterpreted_batch_ndims > 0:
            return jnp.sum(
                entropies,
                axis=tuple(
                    range(-reinterpreted_batch_ndims, -len(base_dist[0].event_shape))
                ),
            )
        else:
            return entropies

    @classmethod
    def cdf(
        cls,
        x: ArrayLike,
        base_dist: Union[rv_frozen, Sequence[rv_frozen]],
        reinterpreted_batch_ndims: int,
    ):
        """Cumulative distribution function of the RV."""
        if isinstance(base_dist, rv_frozen):
            return base_dist.cdf(x)
        else:
            split_dim = (
                -max(len(x.shape), 1)
                if reinterpreted_batch_ndims > 0
                else -(len(x.shape) + len(base_dist[0].batch_shape))
            )
            split_value = jnp.split(
                x, [d.event_shape[0] for d in base_dist[:-1]], axis=split_dim
            )
            cdfs = jnp.concatenate(
                [
                    jnp.expand_dims(d.cdf(v), axis=split_dim)
                    for d, v in zip(base_dist, split_value)
                ],
                axis=split_dim,
            )
            if reinterpreted_batch_ndims > 0:
                sum_dim = tuple(range(-reinterpreted_batch_ndims, 0))
                cdfs = jnp.prod(cdfs, axis=sum_dim)
            return cdfs
