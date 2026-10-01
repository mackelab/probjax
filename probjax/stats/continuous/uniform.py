"""
Uniform Distribution (:mod:`probjax.stats.uniform`)
==================================================

This module contains the Uniform distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import uniform as _uniform

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import interval, real
from probjax.stats.utils import _require_unweighted_fit
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["uniform"]


def _loc_scale(low, high):
    """Convert bounds to loc/scale: JAX uniform is [loc, loc + scale]."""
    return low, high - low


class uniform_gen(rv_continuous):
    """Uniform continuous random variable.

    The uniform distribution with lower bound `low` and upper bound `high`.

    Parameters
    ----------
    low : float, optional
        Lower bound of the distribution. Default is 0.
    high : float, optional
        Upper bound of the distribution. Default is 1.
    """

    # Define parameter constraints
    parameters = {'low': real, 'high': real}

    @classmethod
    def support(cls, low=0.0, high=1.0, **kwargs):
        """Support of the uniform distribution."""
        return interval(low, high)

    @classmethod
    def logpdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Log of the probability density function of the uniform distribution."""
        # Scale to [0, 1] for the JAX implementation
        loc, scale = _loc_scale(low, high)
        return _uniform.logpdf(x, loc, scale)

    @classmethod
    def cdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Cumulative distribution function of the uniform distribution."""
        loc, scale = _loc_scale(low, high)
        return _uniform.cdf(x, loc, scale)

    @classmethod
    def logcdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Log of the cumulative distribution function of the uniform distribution."""
        loc, scale = _loc_scale(low, high)
        return jnp.log(_uniform.cdf(x, loc, scale))

    @classmethod
    def ppf(cls, q, low=0.0, high=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the uniform distribution."""
        loc, scale = _loc_scale(low, high)
        return _uniform.ppf(q, loc, scale)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        low=0.0,
        high=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the uniform distribution."""
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        event_shape = jnp.broadcast_shapes(low.shape, high.shape)
        return random.uniform(rng, shape=shape + event_shape, minval=low, maxval=high)

    @classmethod
    def sf(cls, x, low=0.0, high=1.0, **kwargs):
        """Survival function (1 - cdf) of the uniform distribution."""
        loc, scale = _loc_scale(low, high)
        return _uniform.sf(x, loc, scale)

    @classmethod
    def isf(cls, q, low=0.0, high=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the uniform distribution."""
        loc, scale = _loc_scale(low, high)
        return _uniform.isf(q, loc, scale)

    @classmethod
    def mean(cls, low=0.0, high=1.0, **kwargs):
        """Mean of the uniform distribution."""
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        return (low + high) / 2.0

    @classmethod
    def mode(cls, low=0.0, high=1.0, **kwargs):
        """Mode of the uniform distribution."""
        return (
            low + high
        ) / 2.0  # Note: This is arbitrary, any value in the range is a mode

    @classmethod
    def median(cls, low=0.0, high=1.0, **kwargs):
        """Median of the uniform distribution."""
        return (low + high) / 2.0

    @classmethod
    def var(cls, low=0.0, high=1.0, **kwargs):
        """Variance of the uniform distribution."""
        return (high - low) ** 2 / 12.0

    @classmethod
    def entropy(cls, low=0.0, high=1.0, **kwargs):
        """Entropy of the uniform distribution."""
        return jnp.log(high - low)

    @classmethod
    def moment(cls, n, low=0.0, high=1.0, **kwargs):
        """n-th non-central moment of the uniform distribution."""
        n = jnp.asarray(n)
        return (high ** (n + 1) - low ** (n + 1)) / ((n + 1) * (high - low))

    @classmethod
    def skew(cls, low=0.0, high=1.0, **kwargs):
        """Skewness of the uniform distribution."""
        return jnp.zeros_like(
            jnp.asarray(low)
        )  # Skewness is always 0 (symmetric distribution)

    @classmethod
    def kurtosis(cls, low=0.0, high=1.0, **kwargs):
        """Excess kurtosis of the uniform distribution."""
        return -1.2 * jnp.ones_like(low)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of uniform distribution parameters.

        The MLE for the uniform distribution has a closed-form solution:
        - low = min(data)
        - high = max(data)
        """
        _require_unweighted_fit(weights, "uniform")
        data = jnp.asarray(data)
        low = jnp.min(data)
        high = jnp.max(data)
        return (low, high)


uniform = uniform_gen(name="uniform")
