"""
Binomial Distribution (:mod:`probjax.stats.binomial`)
==================================================

This module implements the Binomial distribution.
"""

from typing import Any, Dict, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import binom
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import unit_interval, strict_positive_integer

__all__ = ["binomial", "binomial_frozen"]


class binomial(rv_discrete):
    """A Binomial distribution."""

    parameters = {
        "n": strict_positive_integer,
        "probs": unit_interval,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, n, probs, **kwds):
        """Parse arguments for the Binomial distribution."""
        n, probs = jnp.broadcast_arrays(n, probs)
        n = n.astype(jnp.int32)
        return (n, probs), kwds

    @classmethod
    def _get_support(cls, n, probs, **kwds):
        """Get the support of the Binomial distribution."""
        return (0, n)

    @classmethod
    def _get_batch_shape(cls, n, probs, **kwds):
        """Get the batch shape of the Binomial distribution."""
        return probs.shape

    @classmethod
    def _get_event_shape(cls, n, probs, **kwds):
        """Get the event shape of the Binomial distribution."""
        return ()

    @classmethod
    def pmf(cls, k: ArrayLike, n, probs, **kwds):
        """Probability mass function of the Binomial distribution."""
        return jnp.exp(cls.logpmf(k, n, probs, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, n, probs, **kwds):
        """Log probability mass function of the Binomial distribution."""
        return binom.logpmf(k, n, probs)

    @classmethod
    def cdf(cls, k: ArrayLike, n, probs, **kwds):
        """Cumulative distribution function of the Binomial distribution."""
        return binom.cdf(k, n, probs)

    @classmethod
    def ppf(cls, q: ArrayLike, n, probs, **kwds):
        """Percent point function of the Binomial distribution."""
        return binom.ppf(q, n, probs)

    @classmethod
    def rvs(
        cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), n=None, probs=None, **kwds
    ):
        """Random variates of the Binomial distribution."""
        max_n = jnp.max(n)
        rvs_shape = shape + (max_n,) + probs.shape + ()
        trials = random.bernoulli(rng, probs, shape=rvs_shape)
        ax = -len(probs.shape) - 1
        sumed_trials = jnp.cumsum(trials, axis=ax)
        ns = jnp.expand_dims(n, axis=tuple(range(len(rvs_shape) - 1)))
        _take = jax.vmap(lambda x, y: jnp.take(x, y, axis=-1), in_axes=(-1, -1))
        final = _take(sumed_trials, ns - 1)
        final = jnp.transpose(final).reshape(shape + probs.shape + ())
        return final

    @classmethod
    def mean(cls, n, probs, **kwds):
        """Mean of the Binomial distribution."""
        return n * probs

    @classmethod
    def median(cls, n, probs, **kwds):
        """Median of the Binomial distribution."""
        return jnp.floor(n * probs)

    @classmethod
    def mode(cls, n, probs, **kwds):
        """Mode of the Binomial distribution."""
        return jnp.floor((n + 1) * probs)

    @classmethod
    def var(cls, n, probs, **kwds):
        """Variance of the Binomial distribution."""
        return n * probs * (1 - probs)

    @classmethod
    def entropy(cls, n, probs, **kwds):
        """Entropy of the Binomial distribution."""
        return jnp.log(2) - probs * jnp.log(probs) - (1 - probs) * jnp.log(1 - probs)

    def freeze(self, n, probs, **kwds):
        """Freeze the Binomial distribution with the given parameters."""
        return binomial_frozen(self, n, probs, **kwds)


class binomial_frozen(rv_discrete_frozen):
    """Frozen Binomial distribution."""

    def __init__(self, dist, n, probs, **kwds):
        super().__init__(dist, n=n, probs=probs, **kwds)
        self.n = n
        self.probs = probs

    def pmf(self, k: ArrayLike):
        """Probability mass function of the frozen Binomial distribution."""
        return self.dist.pmf(k, self.n, self.probs, **self.kwds)

    def logpmf(self, k: ArrayLike):
        """Log probability mass function of the frozen Binomial distribution."""
        return self.dist.logpmf(k, self.n, self.probs, **self.kwds)

    def cdf(self, k: ArrayLike):
        """Cumulative distribution function of the frozen Binomial distribution."""
        return self.dist.cdf(k, self.n, self.probs, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Binomial distribution."""
        return self.dist.ppf(q, self.n, self.probs, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Binomial distribution."""
        return self.dist.rvs(rng, shape, self.n, self.probs, **self.kwds)

    def mean(self):
        """Mean of the frozen Binomial distribution."""
        return self.dist.mean(self.n, self.probs, **self.kwds)

    def median(self):
        """Median of the frozen Binomial distribution."""
        return self.dist.median(self.n, self.probs, **self.kwds)

    def mode(self):
        """Mode of the frozen Binomial distribution."""
        return self.dist.mode(self.n, self.probs, **self.kwds)

    def var(self):
        """Variance of the frozen Binomial distribution."""
        return self.dist.var(self.n, self.probs, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Binomial distribution."""
        return self.dist.entropy(self.n, self.probs, **self.kwds)
