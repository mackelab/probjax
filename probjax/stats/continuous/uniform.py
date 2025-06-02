"""
Uniform Distribution (:mod:`probjax.stats.uniform`)
==================================================

This module contains the Uniform distribution.
"""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray, ArrayLike
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, interval

import jax
from jax.scipy.stats import uniform as _uniform

__all__ = ["uniform"]


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
    def pdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Probability density function of the uniform distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, low, high, **kwargs))

    @classmethod
    def logpdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Log of the probability density function of the uniform distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        # Scale to [0, 1] for the JAX implementation
        loc = low
        scale = high - low
        # JAX uniform is [loc, loc+scale]
        return _uniform.logpdf(x, loc, scale)

    @classmethod
    def cdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Cumulative distribution function of the uniform distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        loc = low
        scale = high - low
        return _uniform.cdf(x, loc, scale)

    @classmethod
    def logcdf(cls, x, low=0.0, high=1.0, **kwargs):
        """Log of the cumulative distribution function of the uniform distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        loc = low
        scale = high - low
        return jnp.log(_uniform.cdf(x, loc, scale))

    @classmethod
    def ppf(cls, q, low=0.0, high=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the uniform distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        loc = low
        scale = high - low
        return _uniform.ppf(q, loc, scale)

    @classmethod
    def rvs(
        cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), low=0.0, high=1.0, **kwargs
    ) -> Float[Array, "..."]:
        """Random variates of the uniform distribution.

        Parameters
        ----------
        rng : PRNGKeyArray
            JAX PRNG key for random number generation
        shape : tuple of ints, optional
            Output shape. Default is (), meaning a single value.
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        event_shape = jnp.broadcast_shapes(low.shape, high.shape)
        return random.uniform(rng, shape=shape + event_shape, minval=low, maxval=high)

    @classmethod
    def sf(cls, x, low=0.0, high=1.0, **kwargs):
        """Survival function (1 - cdf) of the uniform distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        loc = low
        scale = high - low
        return _uniform.sf(x, loc, scale)

    @classmethod
    def isf(cls, q, low=0.0, high=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the uniform distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        loc = low
        scale = high - low
        return _uniform.isf(q, loc, scale)

    @classmethod
    def mean(cls, low=0.0, high=1.0, **kwargs):
        """Mean of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        return (low + high) / 2.0

    @classmethod
    def mode(cls, low=0.0, high=1.0, **kwargs):
        """Mode of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution (technically any value in the range, we return the middle)
        """
        return (
            low + high
        ) / 2.0  # Note: This is arbitrary, any value in the range is a mode

    @classmethod
    def median(cls, low=0.0, high=1.0, **kwargs):
        """Median of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        median : float
            Median of the distribution
        """
        return (low + high) / 2.0

    @classmethod
    def var(cls, low=0.0, high=1.0, **kwargs):
        """Variance of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        return (high - low) ** 2 / 12.0

    @classmethod
    def entropy(cls, low=0.0, high=1.0, **kwargs):
        """Entropy of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return jnp.log(high - low)

    @classmethod
    def moment(cls, n, low=0.0, high=1.0, **kwargs):
        """n-th non-central moment of the uniform distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        n = jnp.asarray(n)
        return (high ** (n + 1) - low ** (n + 1)) / ((n + 1) * (high - low))

    @classmethod
    def skew(cls, low=0.0, high=1.0, **kwargs):
        """Skewness of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return jnp.zeros_like(
            jnp.asarray(low)
        )  # Skewness is always 0 (symmetric distribution)

    @classmethod
    def kurtosis(cls, low=0.0, high=1.0, **kwargs):
        """Excess kurtosis of the uniform distribution.

        Parameters
        ----------
        low : float, optional
            Lower bound of the distribution. Default is 0.
        high : float, optional
            Upper bound of the distribution. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        return -1.2 * jnp.ones_like(low)

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of uniform distribution parameters.

        The MLE for the uniform distribution has a closed-form solution:
        - low = min(data)
        - high = max(data)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (low, high)
        """
        data = jnp.asarray(data)
        low = jnp.min(data)
        high = jnp.max(data)
        return (low, high)


uniform = uniform_gen(name="uniform")
