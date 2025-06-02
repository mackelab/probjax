"""
Cauchy Distribution (:mod:`probjax.stats.cauchy`)
==================================================

This module contains the Cauchy distribution.
"""

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, strict_positive

import jax.scipy.stats.cauchy as _cauchy

__all__ = ["cauchy"]


class cauchy_gen(rv_continuous):
    """Cauchy continuous random variable.

    The Cauchy distribution with location `loc` and scale `scale`.

    Parameters
    ----------
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, **kwargs):
        """Support of the Cauchy distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the Cauchy distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return _cauchy.pdf(x, loc, scale)

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the Cauchy distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        return _cauchy.logpdf(x, loc, scale)

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the Cauchy distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        return _cauchy.cdf(x, loc, scale)

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the Cauchy distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        return _cauchy.ppf(q, loc, scale)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        scale=1.0,
        **kwargs,
    ) -> Float[Array, "..."]:
        """Random variates of the Cauchy distribution.

        Parameters
        ----------
        rng : PRNGKeyArray
            JAX PRNG key for random number generation
        shape : tuple of ints, optional
            Output shape. Default is (), in which case a single value is returned.
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        return random.cauchy(rng, shape=shape + event_shape) * scale + loc

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the Cauchy distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        return _cauchy.sf(x, loc, scale)

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the Cauchy distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        return _cauchy.isf(q, loc, scale)

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the Cauchy distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        return _cauchy.logcdf(x, loc, scale)

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, **kwargs):
        """Mean of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution (undefined, returns NaN)
        """
        return jnp.full_like(loc, jnp.nan)

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, **kwargs):
        """Mode of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
        return jnp.asarray(loc)

    @classmethod
    def median(cls, loc=0.0, scale=1.0, **kwargs):
        """Median of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        median : float
            Median of the distribution
        """
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, **kwargs):
        """Variance of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution (undefined, returns inf)
        """
        return jnp.full_like(loc, jnp.inf)

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return jnp.log(4 * jnp.pi * scale)

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the Cauchy distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment (undefined for n >= 1, returns inf)
        """
        if n == 0:
            return jnp.ones_like(loc)
        return jnp.full_like(loc, jnp.inf)

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution (undefined, returns NaN)
        """
        return jnp.full_like(loc, jnp.nan)

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the Cauchy distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter of the distribution. Default is 0.
        scale : float, optional
            Scale parameter of the distribution. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution (undefined, returns NaN)
        """
        return jnp.full_like(loc, jnp.nan)


cauchy = cauchy_gen(name="cauchy")
