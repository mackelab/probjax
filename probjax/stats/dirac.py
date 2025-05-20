"""
Dirac Distribution (:mod:`probjax.stats.dirac`)
=============================================

This module implements the Dirac delta distribution, which puts all probability mass on a single point.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import real

__all__ = ["dirac", "dirac_frozen"]


class dirac(rv_discrete):
    """A Dirac delta discrete random variable."""

    parameters = {
        "value": real,  # Location of the delta
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, value, **kwds):
        """Parse arguments for the Dirac distribution."""
        return (value,), kwds

    @classmethod
    def _get_support(cls, value, **kwds):
        """Get the support of the Dirac distribution."""
        return (value, value)

    @classmethod
    def support(cls, value, **kwds):
        """Get the support of the Dirac distribution."""
        return cls._get_support(value, **kwds)

    @classmethod
    def _get_batch_shape(cls, value, **kwds):
        """Get the batch shape of the Dirac distribution."""
        return jnp.shape(value)

    @classmethod
    def _get_event_shape(cls, value, **kwds):
        """Get the event shape of the Dirac distribution."""
        return ()

    @classmethod
    def pmf(cls, x: ArrayLike, value, **kwds):
        """Probability mass function of the Dirac distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == value, 1.0, 0.0)

    @classmethod
    def logpmf(cls, x: ArrayLike, value, **kwds):
        """Log probability mass function of the Dirac distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == value, 0.0, -jnp.inf)

    @classmethod
    def cdf(cls, x: ArrayLike, value, **kwds):
        """Cumulative distribution function of the Dirac distribution."""
        x = jnp.asarray(x)
        return jnp.where(x >= value, 1.0, 0.0)

    @classmethod
    def ppf(cls, q: ArrayLike, value, **kwds):
        """Percent point function of the Dirac distribution."""
        q = jnp.asarray(q)
        return jnp.where(q > 0, value, -jnp.inf)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        value=None,
        **kwds,
    ):
        """Random variates of the Dirac distribution."""
        return jnp.broadcast_to(value, shape + jnp.shape(value))

    @classmethod
    def mean(cls, value, **kwds):
        """Mean of the Dirac distribution."""
        return value

    @classmethod
    def var(cls, value, **kwds):
        """Variance of the Dirac distribution."""
        return jnp.zeros_like(value)

    @classmethod
    def entropy(cls, value, **kwds):
        """Entropy of the Dirac distribution."""
        return jnp.zeros_like(value)

    @classmethod
    def mode(cls, value, **kwds):
        """Mode of the Dirac distribution."""
        return value

    def freeze(self, value, **kwds):
        """Freeze the Dirac distribution with the given parameters."""
        return dirac_frozen(self, value=value, **kwds)


class dirac_frozen(rv_discrete_frozen):
    """Frozen Dirac distribution."""

    def __init__(self, dist, value, **kwds):
        super().__init__(dist, value=value, **kwds)
        self.value = value

    def pmf(self, x: ArrayLike):
        """Probability mass function of the frozen Dirac distribution."""
        return self.dist.pmf(x, self.value, **self.kwds)

    def logpmf(self, x: ArrayLike):
        """Log probability mass function of the frozen Dirac distribution."""
        return self.dist.logpmf(x, self.value, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen Dirac distribution."""
        return self.dist.cdf(x, self.value, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Dirac distribution."""
        return self.dist.ppf(q, self.value, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Dirac distribution."""
        return self.dist.rvs(rng, shape, self.value, **self.kwds)

    def mean(self):
        """Mean of the frozen Dirac distribution."""
        return self.dist.mean(self.value, **self.kwds)

    def var(self):
        """Variance of the frozen Dirac distribution."""
        return self.dist.var(self.value, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Dirac distribution."""
        return self.dist.entropy(self.value, **self.kwds)

    def mode(self):
        """Mode of the frozen Dirac distribution."""
        return self.dist.mode(self.value, **self.kwds)

    def support(self):
        """Get the support of the frozen Dirac distribution."""
        return self.dist.support(self.value, **self.kwds)
