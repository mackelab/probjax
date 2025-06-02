"""
Transformed Distribution (:mod:`probjax.stats.transformed`)
=========================================================

This module implements transformed distributions that apply a bijective transformation
to a base distribution.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import functools

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.core import inverse_and_logabsdet
from probjax.stats.base import rv_continuous, rv_continuous_frozen
from probjax.stats.constraints import distribution

__all__ = ["transformed", "transformed_frozen"]


class transformed(rv_continuous):
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
    def _get_support(cls, base_dist, bijector, **kwds):
        """Get the support of the transformed distribution."""
        # For a bijective transformation, the support is the image of the base distribution's support
        # under the transformation. Since we can't easily compute this in general,
        # we return the full real line as a conservative estimate.
        return (-jnp.inf, jnp.inf)

    @classmethod
    def _get_batch_shape(cls, base_dist, bijector, **kwds):
        """Get the batch shape of the transformed distribution."""
        return base_dist.batch_shape

    @classmethod
    def _get_event_shape(cls, base_dist, bijector, **kwds):
        """Get the event shape of the transformed distribution."""
        return base_dist.event_shape

    @classmethod
    def support(cls, base_dist, bijector, **kwds):
        """Get the support of the transformed distribution."""
        return cls._get_support(base_dist, bijector, **kwds)

    @classmethod
    def logpdf(cls, x: ArrayLike, base_dist, bijector, **kwds):
        """Log probability density function of the transformed distribution."""
        inv_and_logdet = inverse_and_logabsdet(bijector)
        inv_value, log_det = inv_and_logdet(x)
        return base_dist.logpdf(inv_value) + log_det

    @classmethod
    def cdf(cls, x: ArrayLike, base_dist, bijector, **kwds):
        """Cumulative distribution function of the transformed distribution."""
        inv_and_logdet = inverse_and_logabsdet(bijector)
        inv_value, _ = inv_and_logdet(x)
        return base_dist.cdf(inv_value)

    @classmethod
    def ppf(cls, q: ArrayLike, base_dist, bijector, **kwds):
        """Percent point function of the transformed distribution."""
        inv_value = base_dist.ppf(q)
        return bijector(inv_value)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        base_dist=None,
        bijector=None,
        **kwds,
    ):
        """Random variates of the transformed distribution."""
        samples = base_dist.rvs(rng, shape)
        return bijector(samples)

    @classmethod
    def mean(cls, base_dist, bijector, **kwds):
        """Mean of the transformed distribution."""
        # For a bijective transformation, the mean is the image of the base distribution's mean
        # under the transformation.
        base_mean = base_dist.mean()
        return bijector(base_mean)

    @classmethod
    def var(cls, base_dist, bijector, **kwds):
        """Variance of the transformed distribution."""
        # For a bijective transformation, we need to compute the variance numerically
        # since it's not easily computable in general.
        samples = base_dist.rvs(jax.random.PRNGKey(0), (10000,))
        transformed_samples = bijector(samples)
        return jnp.var(transformed_samples, axis=0)

    @classmethod
    def entropy(cls, base_dist, bijector, **kwds):
        """Entropy of the transformed distribution."""
        inv_and_logdet = inverse_and_logabsdet(bijector)
        mean = base_dist.mean()
        _, log_det = inv_and_logdet(mean)
        return base_dist.entropy() - jnp.sum(log_det)

    @classmethod
    def mode(cls, base_dist, bijector, **kwds):
        """Mode of the transformed distribution."""
        # For a bijective transformation, the mode is the image of the base distribution's mode
        base_mode = base_dist.mode()
        return bijector(base_mode)

    def freeze(self, base_dist, bijector, **kwds):
        """Freeze the transformed distribution with the given parameters."""
        return transformed_frozen(self, base_dist, bijector, **kwds)


class transformed_frozen(rv_continuous_frozen):
    """Frozen transformed distribution."""

    def __init__(self, dist, base_dist, bijector, **kwds):
        super().__init__(dist, base_dist=base_dist, bijector=bijector, **kwds)
        self.base_dist = base_dist
        self.bijector = bijector
        self._inv_and_logdet = inverse_and_logabsdet(bijector)

    def pdf(self, x: ArrayLike):
        """Probability density function of the frozen transformed distribution."""
        return self.dist.pdf(x, self.base_dist, self.bijector, **self.kwds)

    def logpdf(self, x: ArrayLike):
        """Log probability density function of the frozen transformed distribution."""
        return self.dist.logpdf(x, self.base_dist, self.bijector, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen transformed distribution."""
        return self.dist.cdf(x, self.base_dist, self.bijector, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen transformed distribution."""
        return self.dist.ppf(q, self.base_dist, self.bijector, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen transformed distribution."""
        return self.dist.rvs(rng, shape, self.base_dist, self.bijector, **self.kwds)

    def mean(self):
        """Mean of the frozen transformed distribution."""
        return self.dist.mean(self.base_dist, self.bijector, **self.kwds)

    def var(self):
        """Variance of the frozen transformed distribution."""
        return self.dist.var(self.base_dist, self.bijector, **self.kwds)

    def entropy(self):
        """Entropy of the frozen transformed distribution."""
        return self.dist.entropy(self.base_dist, self.bijector, **self.kwds)

    def mode(self):
        """Mode of the frozen transformed distribution."""
        return self.dist.mode(self.base_dist, self.bijector, **self.kwds)

    def support(self):
        """Get the support of the frozen transformed distribution."""
        return self.dist.support(self.base_dist, self.bijector, **self.kwds)
