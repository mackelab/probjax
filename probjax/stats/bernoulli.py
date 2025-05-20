"""
Bernoulli Distribution (:mod:`probjax.stats.bernoulli`)
====================================================

This module implements the Bernoulli distribution.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import unit_interval

__all__ = ["bernoulli", "bernoulli_frozen"]


class bernoulli(rv_discrete):
    """A Bernoulli distribution."""

    parameters = {
        "p": unit_interval,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, p, **kwds):
        """Parse arguments for the Bernoulli distribution."""
        return (p,), kwds

    @classmethod
    def _get_support(cls, p, **kwds):
        """Get the support of the Bernoulli distribution."""
        return (0, 1)

    @classmethod
    def _get_batch_shape(cls, p, **kwds):
        """Get the batch shape of the Bernoulli distribution."""
        return jnp.shape(p)

    @classmethod
    def _get_event_shape(cls, p, **kwds):
        """Get the event shape of the Bernoulli distribution."""
        return ()

    @classmethod
    def pmf(cls, x: ArrayLike, p, **kwds):
        """Probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, p, 1 - p)

    @classmethod
    def logpmf(cls, x: ArrayLike, p, **kwds):
        """Log probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, jnp.log(p), jnp.log(1 - p))

    @classmethod
    def cdf(cls, x: ArrayLike, p, **kwds):
        """Cumulative distribution function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x < 0, 0, jnp.where(x < 1, 1 - p, 1))

    @classmethod
    def ppf(cls, q: ArrayLike, p, **kwds):
        """Percent point function of the Bernoulli distribution."""
        q = jnp.asarray(q)
        return jnp.where(q < 1 - p, 0, 1)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        p=None,
        **kwds,
    ):
        """Random variates of the Bernoulli distribution."""
        return random.bernoulli(rng, p, shape)

    @classmethod
    def mean(cls, p, **kwds):
        """Mean of the Bernoulli distribution."""
        return p

    @classmethod
    def var(cls, p, **kwds):
        """Variance of the Bernoulli distribution."""
        return p * (1 - p)

    @classmethod
    def entropy(cls, p, **kwds):
        """Entropy of the Bernoulli distribution."""
        return -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)

    def freeze(self, p, **kwds):
        """Freeze the Bernoulli distribution with the given parameters."""
        return bernoulli_frozen(self, p, **kwds)


class bernoulli_frozen(rv_discrete_frozen):
    """Frozen Bernoulli distribution."""

    def __init__(self, dist, p, **kwds):
        super().__init__(dist, p=p, **kwds)
        self.p = p

    def pmf(self, x: ArrayLike):
        """Probability mass function of the frozen Bernoulli distribution."""
        return self.dist.pmf(x, self.p, **self.kwds)

    def logpmf(self, x: ArrayLike):
        """Log probability mass function of the frozen Bernoulli distribution."""
        return self.dist.logpmf(x, self.p, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen Bernoulli distribution."""
        return self.dist.cdf(x, self.p, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Bernoulli distribution."""
        return self.dist.ppf(q, self.p, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Bernoulli distribution."""
        return self.dist.rvs(rng, shape, self.p, **self.kwds)

    def mean(self):
        """Mean of the frozen Bernoulli distribution."""
        return self.dist.mean(self.p, **self.kwds)

    def var(self):
        """Variance of the frozen Bernoulli distribution."""
        return self.dist.var(self.p, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Bernoulli distribution."""
        return self.dist.entropy(self.p, **self.kwds)
