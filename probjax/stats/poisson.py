"""
Poisson Distribution (:mod:`probjax.stats.poisson`)
================================================

This module implements the Poisson distribution.
"""

from typing import Any, Dict, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import poisson
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import positive_integer

__all__ = ["poisson", "poisson_frozen"]


class poisson(rv_discrete):
    """A Poisson distribution."""

    parameters = {
        "rate": positive_integer,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, rate, **kwds):
        """Parse arguments for the Poisson distribution."""
        return (rate,), kwds

    @classmethod
    def _get_support(cls, rate, **kwds):
        """Get the support of the Poisson distribution."""
        return (0, jnp.inf)

    @classmethod
    def _get_batch_shape(cls, rate, **kwds):
        """Get the batch shape of the Poisson distribution."""
        return rate.shape

    @classmethod
    def _get_event_shape(cls, rate, **kwds):
        """Get the event shape of the Poisson distribution."""
        return ()

    @classmethod
    def pmf(cls, k: ArrayLike, rate, **kwds):
        """Probability mass function of the Poisson distribution."""
        return jnp.exp(cls.logpmf(k, rate, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, rate, **kwds):
        """Log probability mass function of the Poisson distribution."""
        return poisson.logpmf(k, rate)

    @classmethod
    def cdf(cls, k: ArrayLike, rate, **kwds):
        """Cumulative distribution function of the Poisson distribution."""
        return poisson.cdf(k, rate)

    @classmethod
    def ppf(cls, q: ArrayLike, rate, **kwds):
        """Percent point function of the Poisson distribution."""
        return poisson.ppf(q, rate)

    @classmethod
    def rvs(cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), rate=None, **kwds):
        """Random variates of the Poisson distribution."""
        return random.poisson(rng, rate, shape=shape)

    @classmethod
    def mean(cls, rate, **kwds):
        """Mean of the Poisson distribution."""
        return rate

    @classmethod
    def var(cls, rate, **kwds):
        """Variance of the Poisson distribution."""
        return rate

    @classmethod
    def entropy(cls, rate, **kwds):
        """Entropy of the Poisson distribution."""
        return rate * (1 - jnp.log(rate))

    def freeze(self, rate, **kwds):
        """Freeze the Poisson distribution with the given parameters."""
        return poisson_frozen(self, rate, **kwds)


class poisson_frozen(rv_discrete_frozen):
    """Frozen Poisson distribution."""

    def __init__(self, dist, rate, **kwds):
        super().__init__(dist, rate=rate, **kwds)
        self.rate = rate

    def pmf(self, k: ArrayLike):
        """Probability mass function of the frozen Poisson distribution."""
        return self.dist.pmf(k, self.rate, **self.kwds)

    def logpmf(self, k: ArrayLike):
        """Log probability mass function of the frozen Poisson distribution."""
        return self.dist.logpmf(k, self.rate, **self.kwds)

    def cdf(self, k: ArrayLike):
        """Cumulative distribution function of the frozen Poisson distribution."""
        return self.dist.cdf(k, self.rate, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Poisson distribution."""
        return self.dist.ppf(q, self.rate, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Poisson distribution."""
        return self.dist.rvs(rng, shape, self.rate, **self.kwds)

    def mean(self):
        """Mean of the frozen Poisson distribution."""
        return self.dist.mean(self.rate, **self.kwds)

    def var(self):
        """Variance of the frozen Poisson distribution."""
        return self.dist.var(self.rate, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Poisson distribution."""
        return self.dist.entropy(self.rate, **self.kwds)
