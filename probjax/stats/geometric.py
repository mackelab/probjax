"""
Geometric Distribution (:mod:`probjax.stats.geometric`)
====================================================

This module implements the Geometric distribution.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import unit_interval

__all__ = ["geometric", "geometric_frozen"]


class geometric(rv_discrete):
    """A Geometric discrete random variable."""

    parameters = {
        "p": unit_interval,  # Probability of success
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, p, **kwds):
        """Parse arguments for the Geometric distribution."""
        return (p,), kwds

    @classmethod
    def _get_support(cls, p, **kwds):
        """Get the support of the Geometric distribution."""
        return (1, jnp.inf)

    @classmethod
    def support(cls, p, **kwds):
        """Get the support of the Geometric distribution."""
        return cls._get_support(p, **kwds)

    @classmethod
    def _get_batch_shape(cls, p, **kwds):
        """Get the batch shape of the Geometric distribution."""
        return jnp.shape(p)

    @classmethod
    def _get_event_shape(cls, p, **kwds):
        """Get the event shape of the Geometric distribution."""
        return ()

    @classmethod
    def pmf(cls, x: ArrayLike, p, **kwds):
        """Probability mass function of the Geometric distribution."""
        x = jnp.asarray(x)
        return p * (1 - p) ** (x - 1)

    @classmethod
    def logpmf(cls, x: ArrayLike, p, **kwds):
        """Log probability mass function of the Geometric distribution."""
        x = jnp.asarray(x)
        return jnp.log(p) + (x - 1) * jnp.log(1 - p)

    @classmethod
    def cdf(cls, x: ArrayLike, p, **kwds):
        """Cumulative distribution function of the Geometric distribution."""
        x = jnp.asarray(x)
        return 1 - (1 - p) ** x

    @classmethod
    def ppf(cls, q: ArrayLike, p, **kwds):
        """Percent point function of the Geometric distribution."""
        q = jnp.asarray(q)
        return jnp.ceil(jnp.log(1 - q) / jnp.log(1 - p))

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        p=None,
        **kwds,
    ):
        """Random variates of the Geometric distribution."""
        return random.geometric(rng, p, shape)

    @classmethod
    def mean(cls, p, **kwds):
        """Mean of the Geometric distribution."""
        return 1.0 / p

    @classmethod
    def var(cls, p, **kwds):
        """Variance of the Geometric distribution."""
        return (1 - p) / (p**2)

    @classmethod
    def entropy(cls, p, **kwds):
        """Entropy of the Geometric distribution."""
        return -(1 - p) * jnp.log(1 - p) / p - jnp.log(p)

    @classmethod
    def mode(cls, p, **kwds):
        """Mode of the Geometric distribution."""
        return 1

    def freeze(self, p, **kwds):
        """Freeze the Geometric distribution with the given parameters."""
        return geometric_frozen(self, p=p, **kwds)


class geometric_frozen(rv_discrete_frozen):
    """Frozen Geometric distribution."""

    def __init__(self, dist, p, **kwds):
        super().__init__(dist, p=p, **kwds)
        self.p = p

    def pmf(self, x: ArrayLike):
        """Probability mass function of the frozen Geometric distribution."""
        return self.dist.pmf(x, self.p, **self.kwds)

    def logpmf(self, x: ArrayLike):
        """Log probability mass function of the frozen Geometric distribution."""
        return self.dist.logpmf(x, self.p, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen Geometric distribution."""
        return self.dist.cdf(x, self.p, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Geometric distribution."""
        return self.dist.ppf(q, self.p, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Geometric distribution."""
        return self.dist.rvs(rng, shape, self.p, **self.kwds)

    def mean(self):
        """Mean of the frozen Geometric distribution."""
        return self.dist.mean(self.p, **self.kwds)

    def var(self):
        """Variance of the frozen Geometric distribution."""
        return self.dist.var(self.p, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Geometric distribution."""
        return self.dist.entropy(self.p, **self.kwds)

    def mode(self):
        """Mode of the frozen Geometric distribution."""
        return self.dist.mode(self.p, **self.kwds)

    def support(self):
        """Get the support of the frozen Geometric distribution."""
        return self.dist.support(self.p, **self.kwds)
