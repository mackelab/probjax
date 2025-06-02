"""
Empirical Distribution (:mod:`probjax.stats.empirical`)
===================================================

This module implements the Empirical distribution, which puts probability mass on observed data points.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jaxtyping import ArrayLike, PRNGKeyArray

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import simplex

__all__ = ["empirical", "empirical_frozen"]


class empirical(rv_discrete):
    """An Empirical distribution that puts probability mass on observed data points."""

    parameters = {
        "values": None,  # No constraints on values
        "weights": simplex,  # Optional weights must sum to 1
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, values, weights=None, **kwds):
        """Parse arguments for the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return (values, weights), kwds

    @classmethod
    def _get_support(cls, values, weights=None, **kwds):
        """Get the support of the Empirical distribution."""
        return (jnp.min(values), jnp.max(values))

    @classmethod
    def support(cls, values, weights=None, **kwds):
        """Get the support of the Empirical distribution."""
        return cls._get_support(values, weights, **kwds)

    @classmethod
    def _get_batch_shape(cls, values, weights=None, **kwds):
        """Get the batch shape of the Empirical distribution."""
        return values.shape[1:]

    @classmethod
    def _get_event_shape(cls, values, weights=None, **kwds):
        """Get the event shape of the Empirical distribution."""
        return ()

    @classmethod
    def pmf(cls, x: ArrayLike, values, weights=None, **kwds):
        """Probability mass function of the Empirical distribution."""
        x = jnp.asarray(x)
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return jnp.sum(weights * (x == values), axis=0)

    @classmethod
    def logpmf(cls, x: ArrayLike, values, weights=None, **kwds):
        """Log probability mass function of the Empirical distribution."""
        x = jnp.asarray(x)
        return jnp.log(cls.pmf(x, values, weights, **kwds))

    @classmethod
    def cdf(cls, x: ArrayLike, values, weights=None, **kwds):
        """Cumulative distribution function of the Empirical distribution."""
        x = jnp.asarray(x)
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return jnp.sum(weights * (values <= x), axis=0)

    @classmethod
    def ppf(cls, q: ArrayLike, values, weights=None, **kwds):
        """Percent point function of the Empirical distribution."""
        q = jnp.asarray(q)
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        sorted_idx = jnp.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cdf = jnp.cumsum(sorted_weights)
        return jnp.interp(q, cdf, sorted_values)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        values=None,
        weights=None,
        **kwds,
    ):
        """Random variates of the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        idx = random.categorical(rng, weights, shape)
        return values[idx]

    @classmethod
    def mean(cls, values, weights=None, **kwds):
        """Mean of the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return jnp.sum(weights * values, axis=0)

    @classmethod
    def var(cls, values, weights=None, **kwds):
        """Variance of the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        mean = cls.mean(values, weights, **kwds)
        return jnp.sum(weights * (values - mean) ** 2, axis=0)

    @classmethod
    def entropy(cls, values, weights=None, **kwds):
        """Entropy of the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return -jnp.sum(weights * jnp.log(weights))

    @classmethod
    def mode(cls, values, weights=None, **kwds):
        """Mode of the Empirical distribution."""
        if weights is None:
            weights = jnp.ones(values.shape[0]) / values.shape[0]
        return values[jnp.argmax(weights)]

    def freeze(self, values, weights=None, **kwds):
        """Freeze the Empirical distribution with the given parameters."""
        return empirical_frozen(self, values=values, weights=weights, **kwds)


class empirical_frozen(rv_discrete_frozen):
    """Frozen Empirical distribution."""

    def __init__(self, dist, values, weights=None, **kwds):
        super().__init__(dist, values=values, weights=weights, **kwds)
        self.values = values
        self.weights = weights

    def pmf(self, x: ArrayLike):
        """Probability mass function of the frozen Empirical distribution."""
        return self.dist.pmf(x, self.values, self.weights, **self.kwds)

    def logpmf(self, x: ArrayLike):
        """Log probability mass function of the frozen Empirical distribution."""
        return self.dist.logpmf(x, self.values, self.weights, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen Empirical distribution."""
        return self.dist.cdf(x, self.values, self.weights, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen Empirical distribution."""
        return self.dist.ppf(q, self.values, self.weights, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen Empirical distribution."""
        return self.dist.rvs(rng, shape, self.values, self.weights, **self.kwds)

    def mean(self):
        """Mean of the frozen Empirical distribution."""
        return self.dist.mean(self.values, self.weights, **self.kwds)

    def var(self):
        """Variance of the frozen Empirical distribution."""
        return self.dist.var(self.values, self.weights, **self.kwds)

    def entropy(self):
        """Entropy of the frozen Empirical distribution."""
        return self.dist.entropy(self.values, self.weights, **self.kwds)

    def mode(self):
        """Mode of the frozen Empirical distribution."""
        return self.dist.mode(self.values, self.weights, **self.kwds)

    def support(self):
        """Get the support of the frozen Empirical distribution."""
        return self.dist.support(self.values, self.weights, **self.kwds)
