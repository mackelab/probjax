"""
Empirical Distribution (:mod:`probjax.stats.empirical`)
===================================================

This module implements the Empirical distribution, which puts probability
mass on observed data points.
"""

from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_discrete_frozen
from probjax.stats.constraints import simplex
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["empirical", "empirical_frozen"]


def _default_weights(values, weights=None):
    """Uniform weights over rows when none are provided."""
    if weights is None:
        weights = jnp.ones(values.shape[0]) / values.shape[0]
    return weights


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
            weights = _default_weights(values, weights)
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
    def pmf(
        cls,
        x: ArrayLike,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        **kwds: Any,
    ) -> ArrayLike:
        """Probability mass function of the Empirical distribution."""
        x = jnp.asarray(x)
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        return jnp.sum(weights * (x == values), axis=0)

    @classmethod
    def logpmf(
        cls,
        x: ArrayLike,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        **kwds: Any,
    ) -> ArrayLike:
        """Log probability mass function of the Empirical distribution."""
        x = jnp.asarray(x)
        return jnp.log(cls.pmf(x, values, weights, **kwds))

    @classmethod
    def cdf(
        cls,
        x: ArrayLike,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        **kwds: Any,
    ) -> ArrayLike:
        """Cumulative distribution function of the Empirical distribution."""
        x = jnp.asarray(x)
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        return jnp.sum(weights * (values <= x), axis=0)

    @classmethod
    def ppf(
        cls,
        q: ArrayLike,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        **kwds: Any,
    ) -> ArrayLike:
        """Percent point function of the Empirical distribution."""
        q = jnp.asarray(q)
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        weights = jnp.asarray(weights)
        sorted_idx = jnp.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cdf = jnp.cumsum(sorted_weights)
        return jnp.interp(q, cdf, sorted_values)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        values=None,
        weights=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Empirical distribution."""
        if weights is None:
            weights = _default_weights(values, weights)
        weights = jnp.asarray(weights)
        values = jnp.asarray(values)
        idx = random.categorical(rng, weights, shape=shape)
        return values[idx]

    @classmethod
    def mean(
        cls, values: ArrayLike, weights: Optional[ArrayLike] = None, **kwds: Any
    ) -> ArrayLike:
        """Mean of the Empirical distribution."""
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        return jnp.sum(weights * values, axis=0)

    @classmethod
    def var(
        cls, values: ArrayLike, weights: Optional[ArrayLike] = None, **kwds: Any
    ) -> ArrayLike:
        """Variance of the Empirical distribution."""
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        mean = cls.mean(values, weights, **kwds)
        return jnp.sum(weights * (values - mean) ** 2, axis=0)

    @classmethod
    def entropy(
        cls,
        values: ArrayLike,
        weights: Optional[ArrayLike] = None,
        **kwds: Any,
    ) -> ArrayLike:
        """Entropy of the Empirical distribution."""
        if weights is None:
            weights = _default_weights(values, weights)
        weights = jnp.asarray(weights)
        return -jnp.sum(weights * jnp.log(weights))

    @classmethod
    def mode(
        cls, values: ArrayLike, weights: Optional[ArrayLike] = None, **kwds: Any
    ) -> ArrayLike:
        """Mode of the Empirical distribution."""
        values = jnp.asarray(values)
        if weights is None:
            weights = _default_weights(values, weights)
        return values[jnp.argmax(weights)]

    def freeze(self, values, weights=None, **kwds):
        """Freeze the Empirical distribution with the given parameters."""
        return empirical_frozen(self, values=values, weights=weights, **kwds)


class empirical_frozen(rv_discrete_frozen):
    """Frozen Empirical distribution.

    The methods come from :class:`~probjax.stats.base.rv_discrete_frozen`,
    which calls back into :class:`empirical` with the frozen ``values`` and
    ``weights``.
    """
