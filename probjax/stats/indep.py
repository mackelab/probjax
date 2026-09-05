"""
Independent Distribution (:mod:`probjax.stats.indep`)
====================================================

This module contains the Independent distribution, which treats a distribution as a
batch of independent distributions.
"""

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import random

from probjax.utils.typing import RngKey

from .base import rv_continuous_frozen, rv_generic
from .constraints import distribution, non_negative_integer

__all__ = ["indep"]


def determine_shapes(
    base_dist: Union[rv_continuous_frozen, Sequence[rv_continuous_frozen]],
    reinterpreted_batch_ndims: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Helper function to determine shapes for Independent distribution."""
    if isinstance(base_dist, rv_continuous_frozen):
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
        "Batch dimensions must be equal for all base distributions."
    )
    assert all([e == event_ndims[0] for e in event_ndims]), (
        "Event dimensions must be equal for all base distributions."
    )

    # Reinterpret batch dimensions as event dimensions where applicable.
    new_event_shapes = []
    new_batch_shapes = []

    for b_shape, e_shape in zip(batch_shapes, event_shapes, strict=False):
        if len(b_shape) > 0:
            # Reinterpret batch dimensions as event dimensions
            new_event_shape = b_shape + e_shape
            new_batch_shape = ()
        else:
            new_event_shape = e_shape
            new_batch_shape = b_shape

        new_event_shapes.append(new_event_shape)
        new_batch_shapes.append(new_batch_shape)

    # Concatenate event shapes for multiple distributions
    if len(new_event_shapes) > 1:
        # Sum the first dimension of each event shape
        first_dims = [e[0] if len(e) > 0 else 0 for e in new_event_shapes]
        first_dim_sum = sum(first_dims)

        # Take the rest of the dimensions from the first event shape
        other_dims = new_event_shapes[0][1:] if len(new_event_shapes[0]) > 0 else ()

        # Combine into final event shape
        event_shape = (first_dim_sum,) + other_dims
    else:
        event_shape = new_event_shapes[0]

    # Batch shape is empty since we've reinterpreted all batch dimensions
    batch_shape = ()

    # For splitting: use the first dimension of each event shape
    split_dims = [e[0] if len(e) > 0 else 1 for e in new_event_shapes]
    # Only need split points between distributions
    split_indices = [sum(split_dims[: i + 1]) for i in range(len(split_dims) - 1)]

    return (
        tuple(batch_shape),
        tuple(event_shape),
        tuple(split_dims),
        tuple(split_indices),
    )


class rv_frozen_indep(rv_continuous_frozen):
    """Frozen independent distribution."""

    def __init__(self, dist, base_dists, reinterpreted_batch_ndims, **kwargs):
        super().__init__(dist, base_dists, reinterpreted_batch_ndims, **kwargs)
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self.split_dims = split_dims
        self.split_indices = split_indices

    def _compute_batch_and_event_shape(
        self, base_dists, reinterpreted_batch_ndims, **kwargs
    ):
        batch_shape, event_shape, _, _ = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        return batch_shape, event_shape


class indep_gen(rv_generic):
    """Independent random variable.

    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Parameters
    ----------
    *base_dists : rv_continuous_frozen
        Base distribution(s) to make independent.
    reinterpreted_batch_ndims : int, optional
        The number of batch dimensions that should be considered as event dimensions.
        Default is 1.
    """

    parameters = {
        "base_dists": distribution,
        "reinterpreted_batch_ndims": non_negative_integer,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, *base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Create a frozen independent distribution."""
        return self.freeze(
            base_dists=base_dists,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            **kwargs,
        )

    def freeze(self, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Freeze the independent distribution with the given parameters."""
        return rv_frozen_indep(
            self,
            base_dists=base_dists,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            **kwargs,
        )

    @classmethod
    def support(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Support of the independent distribution."""
        if len(base_dists) == 1:
            return base_dists[0].support()
        else:
            return tuple(d.support() for d in base_dists)

    @classmethod
    def pdf(cls, x, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Probability density function of the independent distribution."""
        return jnp.exp(cls.logpdf(x, base_dists, reinterpreted_batch_ndims, **kwargs))

    @classmethod
    def logpdf(cls, x, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Log of the probability density function of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )

        # Split the input along the last dimension
        split_value = jnp.split(x, split_indices, axis=-1)

        # Compute logpdf for each base distribution
        logpdf = sum(d.logpdf(v) for d, v in zip(base_dists, split_value, strict=False))
        # Sum up to be of shape reinterpreted_batch_ndins
        for _ in range(reinterpreted_batch_ndims):
            logpdf = jnp.sum(logpdf, axis=-1)
        return logpdf

    @classmethod
    def cdf(cls, x, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Cumulative distribution function of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )

        # Split the input along the last dimension
        split_value = jnp.split(x, split_indices, axis=-1)

        # Compute CDF for each base distribution
        cdf = jnp.prod([
            d.cdf(v) for d, v in zip(base_dists, split_value, strict=False)
        ])

        # Product up to be of shape reinterpreted_batch_ndims
        for _ in range(reinterpreted_batch_ndims):
            cdf = jnp.prod(cdf, axis=-1)
        return cdf

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        base_dists=None,
        reinterpreted_batch_ndims=1,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        keys = random.split(rng, len(base_dists))

        # Generate samples for each base distribution
        samples = jnp.concatenate(
            [
                d.dist._rvs_impl(k, shape=shape, **d._call_kwds)
                for k, d in zip(keys, base_dists, strict=False)
            ],
            axis=-1,
        )
        return samples

    @classmethod
    def mean(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Mean of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        if len(base_dists) == 1:
            return base_dists[0].mean(*kwargs)
        else:
            means = jnp.stack([d.mean(*kwargs) for d in base_dists], axis=-1)
            return means.reshape(batch_shape + event_shape)

    @classmethod
    def var(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Variance of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        if len(base_dists) == 1:
            return base_dists[0].var(*kwargs)
        else:
            variances = jnp.stack([d.var(*kwargs) for d in base_dists], axis=-1)
            return variances.reshape(batch_shape + event_shape)

    @classmethod
    def entropy(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Entropy of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        if len(base_dists) == 1:
            return base_dists[0].entropy(*kwargs)
        else:
            entropies = jnp.stack([d.entropy(*kwargs) for d in base_dists], axis=-1)
            return entropies.reshape(batch_shape + event_shape)

    @classmethod
    def mode(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Mode of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        if len(base_dists) == 1:
            return base_dists[0].mode(*kwargs)
        else:
            modes = jnp.stack([d.mode(*kwargs) for d in base_dists], axis=-1)
            return modes.reshape(batch_shape + event_shape)

    @classmethod
    def fit(cls, data, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Fit the independent distribution to data.

        Parameters
        ----------
        data : ArrayLike
            The data to fit the distribution to.
        base_dists : Sequence[rv_continuous_frozen]
            The base distributions to fit.
        reinterpreted_batch_ndims : int, optional
            The number of batch dimensions that should be considered as event
            dimensions.
            Default is 1.
        **kwargs
            Additional keyword arguments passed to each base distribution's fit method.

        Returns
        -------
        Sequence[rv_continuous_frozen]
            The fitted base distributions.
        """
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )

        # Split the data along the last dimension
        split_data = jnp.split(data, split_indices, axis=-1)

        # Fit each base distribution
        fitted_dists = [
            d.fit(dat, **kwargs) for d, dat in zip(base_dists, split_data, strict=False)
        ]

        return fitted_dists


indep = indep_gen(name="indep")
