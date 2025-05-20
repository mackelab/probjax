"""
Categorical Distribution (:mod:`probjax.stats.categorical`)
========================================================

This module implements the Categorical distribution.
"""

from typing import Any, Dict, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import simplex

__all__ = ["categorical", "categorical_frozen"]


class categorical(rv_discrete):
    """A Categorical distribution."""

    parameters = {
        "probs": simplex,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, probs, **kwds):
        """Parse arguments for the Categorical distribution."""
        probs = jax.nn.softmax(probs)
        return (probs,), kwds

    @classmethod
    def _get_support(cls, probs, **kwds):
        """Get the support of the Categorical distribution."""
        return (0, probs.shape[-1] - 1)

    @classmethod
    def support(cls, probs, **kwds):
        """Get the support of the Categorical distribution."""
        return cls._get_support(probs, **kwds)

    @classmethod
    def _get_batch_shape(cls, probs, **kwds):
        """Get the batch shape of the Categorical distribution."""
        if len(probs.shape) > 1:
            return probs.shape[:-1]
        return ()

    @classmethod
    def _get_event_shape(cls, probs, **kwds):
        """Get the event shape of the Categorical distribution."""
        return ()

    @classmethod
    def pmf(cls, k: ArrayLike, probs, **kwds):
        """Probability mass function of the Categorical distribution."""
        return jnp.exp(cls.logpmf(k, probs, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, probs, **kwds):
        """Log probability mass function of the Categorical distribution."""
        k = jnp.asarray(k).astype(jnp.int32)
        k = jax.nn.one_hot(k, probs.shape[-1])
        log_probs = jax.scipy.special.xlogy(k, probs).sum(axis=-1)
        return log_probs

    @classmethod
    def rvs(cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), probs=None, **kwds):
        """Random variates of the Categorical distribution."""
        return random.categorical(rng, probs, shape=shape, axis=-1)

    @classmethod
    def mean(cls, probs, **kwds):
        """Mean of the Categorical distribution."""
        return jnp.sum(probs * jnp.arange(probs.shape[-1]), axis=-1)

    @classmethod
    def var(cls, probs, **kwds):
        """Variance of the Categorical distribution."""
        mean = cls.mean(probs, **kwds)
        return jnp.sum(probs * (jnp.arange(probs.shape[-1]) - mean) ** 2, axis=-1)

    @classmethod
    def entropy(cls, probs, **kwds):
        """Entropy of the Categorical distribution."""
        return -jnp.sum(probs * jnp.log(probs), axis=-1)

    def freeze(self, probs, **kwds):
        """Freeze the Categorical distribution with the given parameters."""
        return categorical_frozen(self, probs, **kwds)


class categorical_frozen(rv_discrete_frozen):
    """Frozen Categorical distribution."""

    def __init__(self, dist, probs, **kwds):
        super().__init__(dist, probs=probs, **kwds)
        self.probs = probs

    def pmf(self, k: ArrayLike):
        """Probability mass function of the frozen Categorical distribution."""
        return self.dist.pmf(k, self.probs, **self.kwds)

    def logpmf(self, k: ArrayLike):
        """Log probability mass function of the frozen Categorical distribution."""
        return self.dist.logpmf(k, self.probs, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Categorical distribution."""
        return self.dist.rvs(rng, shape, self.probs, **self.kwds)

    def mean(self):
        """Mean of the frozen Categorical distribution."""
        return self.dist.mean(self.probs, **self.kwds)

    def var(self):
        """Variance of the frozen Categorical distribution."""
        return self.dist.var(self.probs, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Categorical distribution."""
        return self.dist.entropy(self.probs, **self.kwds)
