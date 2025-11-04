"""
Transformed Distribution (:mod:`probjax.stats.transformed`)
=========================================================

This module implements transformed distributions that apply a bijective transformation
to a base distribution.
"""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.core import inverse_and_logabsdet
from probjax.stats.base import rv_continuous
from probjax.stats.constraints import distribution, real
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["transformed"]


class transformed_gen(rv_continuous):
    """A transformed distribution that applies a bijective transformation to a base distribution."""

    parameters = {
        "base_dist": distribution,
        "bijector": callable,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, base_dist, bijector, **kwds):
        """Parse arguments for the transformed distribution."""
        return (base_dist, bijector), kwds

    @classmethod
    def support(cls, base_dist, bijector, **kwds):
        """Support of the transformed distribution."""
        return real

    @classmethod
    @functools.cache
    def _get_vmapped_bijector(cls, batch_shape, bijector):
        """Get the vmapped bijector."""
        for _ in range(len(batch_shape)):
            bijector = jax.vmap(bijector)
        return bijector

    @classmethod
    @functools.cache
    def _get_vmapped_inverse_and_logdet(
        cls, batch_shape, bijector, inv_and_logdet=None
    ):
        """Get the vmapped inverse and log determinant."""
        if inv_and_logdet is None:
            inv_and_logdet = inverse_and_logabsdet(bijector)
        for _ in range(len(batch_shape)):
            inv_and_logdet = jax.vmap(inv_and_logdet)
        return inv_and_logdet

    @classmethod
    def logpdf(cls, x: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Log probability density function of the transformed distribution."""
        event_shape = base_dist.event_shape
        x = x.reshape(-1, *event_shape)
        inverse_and_logdet = cls._get_vmapped_inverse_and_logdet(
            x.shape[:1], bijector, inverse_and_logdet
        )
        inv_value, log_det = inverse_and_logdet(x)
        return base_dist.logpdf(inv_value) + log_det

    @classmethod
    def pdf(cls, x: ArrayLike, base_dist, bijector, **kwds):
        """Probability density function of the transformed distribution."""
        return jnp.exp(cls.logpdf(x, base_dist, bijector, **kwds))

    @classmethod
    def cdf(cls, x: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Cumulative distribution function of the transformed distribution."""
        batch_shape = base_dist.batch_shape
        bijector = cls._get_vmapped_bijector(batch_shape, bijector)
        inverse_and_logdet = cls._get_vmapped_inverse_and_logdet(
            batch_shape, bijector, inverse_and_logdet
        )
        inv_value, _ = inverse_and_logdet(x)
        return base_dist.cdf(inv_value)

    @classmethod
    def ppf(cls, q: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Percent point function of the transformed distribution."""
        batch_shape = base_dist.batch_shape
        bijector = cls._get_vmapped_bijector(batch_shape, bijector)
        inverse_and_logdet = cls._get_vmapped_inverse_and_logdet(
            batch_shape, bijector, inverse_and_logdet
        )
        inv_value, _ = inverse_and_logdet(q)
        return bijector(inv_value)

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        base_dist=None,
        bijector=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the transformed distribution."""
        batch_shape = base_dist.batch_shape
        batch_shape = shape + batch_shape
        bijector = cls._get_vmapped_bijector(batch_shape, bijector)
        samples = base_dist.rvs(rng, shape=shape)
        return bijector(samples)

    @classmethod
    def mean(cls, base_dist, bijector, **kwds):
        """Mean of the transformed distribution."""
        raise NotImplementedError("Mean not implemented for transformed distribution")

    @classmethod
    def var(cls, base_dist, bijector, **kwds):
        """Variance of the transformed distribution."""
        raise NotImplementedError(
            "Variance not implemented for transformed distribution"
        )

    @classmethod
    def entropy(cls, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Entropy of the transformed distribution."""
        raise NotImplementedError(
            "Entropy not implemented for transformed distribution"
        )

    @classmethod
    def mode(cls, base_dist, bijector, **kwds):
        """Mode of the transformed distribution."""
        raise NotImplementedError("Mode not implemented for transformed distribution")


transformed = transformed_gen(name="transformed")
