"""
Independent Distribution (:mod:`probjax.stats.independent`)
==================================================

This module contains the Independent distribution, which treats a distribution as a batch of independent distributions.
"""

from typing import Sequence, Union, Tuple, Optional, List
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, PRNGKeyArray, ArrayLike

from .base import rv_generic, rv_continuous_frozen
from .constraints import Constraint, distribution, non_negative_integer

__all__ = ["independent"]


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

    # For each distribution, reinterpret batch dimensions as event dimensions if applicable
    new_event_shapes = []
    new_batch_shapes = []

    for b_shape, e_shape in zip(batch_shapes, event_shapes):
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


class rv_frozen_independent(rv_continuous_frozen):
    """Frozen independent distribution."""

    def __init__(self, dist, base_dists, reinterpreted_batch_ndims, **kwargs):
        super().__init__(dist, **kwargs)
        self.args = (base_dists,)
        self.kwds = {"reinterpreted_batch_ndims": reinterpreted_batch_ndims}
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self.split_dims = split_dims
        self.split_indices = split_indices
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.base_dists = base_dists


class independent_gen(rv_generic):
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
        return rv_frozen_independent(
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
        logpdf = sum(d.logpdf(v) for d, v in zip(base_dists, split_value))
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
        cdf = jnp.prod([d.cdf(v) for d, v in zip(base_dists, split_value)])

        # Product up to be of shape reinterpreted_batch_ndims
        for _ in range(reinterpreted_batch_ndims):
            cdf = jnp.prod(cdf, axis=-1)
        return cdf

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        base_dists=None,
        reinterpreted_batch_ndims=1,
        **kwargs,
    ):
        """Random variates of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        keys = random.split(rng, len(base_dists))

        # Generate samples for each base distribution
        samples = jnp.concatenate(
            [d.rvs(k, shape) for k, d in zip(keys, base_dists)],
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
        variances = jnp.stack([d.var() for d in base_dists], axis=-1)
        # Sum up to be of shape reinterpreted_batch_ndims
        for _ in range(reinterpreted_batch_ndims):
            variances = jnp.sum(variances, axis=-1)
        return variances

    @classmethod
    def entropy(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Entropy of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        entropies = jnp.stack([d.entropy() for d in base_dists], axis=-1)
        # Sum up to be of shape reinterpreted_batch_ndims
        for _ in range(reinterpreted_batch_ndims):
            entropies = jnp.sum(entropies, axis=-1)
        return entropies

    @classmethod
    def mode(cls, base_dists, reinterpreted_batch_ndims=1, **kwargs):
        """Mode of the independent distribution."""
        batch_shape, event_shape, split_dims, split_indices = determine_shapes(
            base_dists, reinterpreted_batch_ndims
        )
        modes = jnp.stack([d.mode() for d in base_dists], axis=-1)
        # Sum up to be of shape reinterpreted_batch_ndims
        for _ in range(reinterpreted_batch_ndims):
            modes = jnp.sum(modes, axis=-1)
        return modes

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
            The number of batch dimensions that should be considered as event dimensions.
            Default is 1.
        **kwargs
            Additional keyword arguments passed to the fit method of each base distribution.

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
        fitted_dists = [d.fit(dat, **kwargs) for d, dat in zip(base_dists, split_data)]

        return fitted_dists


independent = independent_gen(name="independent")
