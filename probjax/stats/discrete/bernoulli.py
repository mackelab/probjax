"""
Bernoulli Distribution (:mod:`probjax.stats.bernoulli`)
====================================================

This module implements the Bernoulli distribution.
"""

from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import unit_interval
from probjax.stats.utils import weighted_mean
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["bernoulli"]


class bernoulli_gen(rv_discrete, rv_exponential_family):
    """A Bernoulli distribution."""

    parameters = {
        "p": unit_interval,
    }

    @classmethod
    def support(cls, p: ArrayLike, **kwds: Any) -> Tuple[int, int]:
        """Support of the Bernoulli distribution."""
        return (0, 1)

    @classmethod
    def pmf(cls, x: ArrayLike, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, p, 1 - p)

    @classmethod
    def logpmf(cls, x: ArrayLike, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Log probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, jnp.log(p), jnp.log(1 - p))

    @classmethod
    def cdf(cls, x: ArrayLike, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Cumulative distribution function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x < 0, 0, jnp.where(x < 1, 1 - p, 1))

    @classmethod
    def ppf(cls, q: ArrayLike, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Percent point function of the Bernoulli distribution."""
        q = jnp.asarray(q)
        p = jnp.asarray(p)
        # For q < 1-p, return 0
        # For q >= 1-p, return 1
        # Handle edge cases where q is exactly 1-p
        return jnp.where(q <= 1 - p, 0, 1)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        p=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Bernoulli distribution."""
        p = jnp.asarray(p)
        event_shape = p.shape
        return random.bernoulli(rng, p, shape=shape + event_shape)

    @classmethod
    def mean(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mean of the Bernoulli distribution."""
        return p

    @classmethod
    def var(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Variance of the Bernoulli distribution."""
        return p * (1 - p)

    @classmethod
    def entropy(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Entropy of the Bernoulli distribution."""
        return -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)

    @classmethod
    def mode(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mode of the Bernoulli distribution."""
        return jnp.where(p > 0.5, 1, 0)

    @classmethod
    def natural_parameters(cls, p: ArrayLike, **kwds: Any) -> Array:
        """Natural parameters of the Bernoulli distribution."""
        return jnp.array([jnp.log(p / (1 - p))])

    @classmethod
    def sufficient_statistics(cls, x: ArrayLike, **kwds: Any) -> Array:
        """Sufficient statistics of the Bernoulli distribution."""
        return jnp.array([x])

    @classmethod
    def log_partition(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Log partition function of the Bernoulli distribution."""
        return -jnp.log(1 - p)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Bernoulli distribution parameter.

        The MLE for the Bernoulli distribution is simply the sample mean:
        p = mean(data)
        """
        p = weighted_mean(data, weights)
        return (p,)


bernoulli = bernoulli_gen(name="bernoulli")
