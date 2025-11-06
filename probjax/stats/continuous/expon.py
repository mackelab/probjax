"""
Exponential Distribution (:mod:`probjax.stats.expon`)
==================================================

This module contains the Exponential distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import expon as _expon

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import positive, strict_positive
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["expon"]


class expon_gen(rv_continuous, rv_exponential_family):
    """Exponential continuous random variable.

    The exponential distribution with rate parameter `rate`.

    Parameters
    ----------
    rate : float, optional
        Rate parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'rate': strict_positive}

    @classmethod
    def support(cls, rate=1.0, **kwargs):
        """Support of the exponential distribution."""
        return positive

    @classmethod
    def pdf(cls, x, rate=1.0, **kwargs):
        """Probability density function of the exponential distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, rate, **kwargs))

    @classmethod
    def logpdf(cls, x, rate=1.0, **kwargs):
        """Log of the probability density function of the exponential distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        # JAX implementation uses scale (1/rate) parameter
        scale = 1.0 / rate
        return _expon.logpdf(x, loc=0.0, scale=scale)

    @classmethod
    def cdf(cls, x, rate=1.0, **kwargs):
        """Cumulative distribution function of the exponential distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        scale = 1.0 / rate
        return _expon.cdf(x, loc=0.0, scale=scale)

    @classmethod
    def logcdf(cls, x, rate=1.0, **kwargs):
        """Log of the cumulative distribution function of the exponential distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        scale = 1.0 / rate
        return _expon.logcdf(x, loc=0.0, scale=scale)

    @classmethod
    def ppf(cls, q, rate=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the exponential distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        scale = 1.0 / rate
        return _expon.ppf(q, loc=0.0, scale=scale)

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        rate=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the exponential distribution."""
        rate = jnp.asarray(rate)
        event_shape = rate.shape
        return random.exponential(rng, shape=shape + event_shape) / rate

    @classmethod
    def sf(cls, x, rate=1.0, **kwargs):
        """Survival function (1 - cdf) of the exponential distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        scale = 1.0 / rate
        return _expon.sf(x, loc=0.0, scale=scale)

    @classmethod
    def isf(cls, q, rate=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the exponential distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        q = jnp.asarray(q)
        rate_arr = jnp.asarray(rate)
        return -jnp.log(jnp.clip(q, a_min=jnp.finfo(q.dtype).tiny, a_max=1.0)) / rate_arr

    @classmethod
    def mean(cls, rate=1.0, **kwargs):
        """Mean of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        return jnp.asarray(1.0) / jnp.asarray(rate)

    @classmethod
    def mode(cls, rate=1.0, **kwargs):
        """Mode of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
        return jnp.zeros_like(rate)

    @classmethod
    def var(cls, rate=1.0, **kwargs):
        """Variance of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        rate_arr = jnp.asarray(rate)
        return jnp.asarray(1.0) / (rate_arr**2)

    @classmethod
    def entropy(cls, rate=1.0, **kwargs):
        """Entropy of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return 1.0 - jnp.log(rate)

    @classmethod
    def moment(cls, n, rate=1.0, **kwargs):
        """n-th non-central moment of the exponential distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        n = jnp.asarray(n)
        # Factorial moment: E[X^n] = n! / rate^n
        return jnp.exp(jnp.log(jnp.prod(jnp.arange(1, n + 1))) - n * jnp.log(rate))

    @classmethod
    def skew(cls, rate=1.0, **kwargs):
        """Skewness of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return 2.0 * jnp.ones_like(rate)  # Skewness is always 2

    @classmethod
    def kurtosis(cls, rate=1.0, **kwargs):
        """Excess kurtosis of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        return 6.0 * jnp.ones_like(rate)  # Excess kurtosis is always 6

    @classmethod
    def natural_parameters(cls, rate=1.0, **kwargs):
        """Natural parameters of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        natural_parameters : tuple
            Natural parameters of the distribution
        """
        return jnp.array([-rate])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the exponential distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : tuple
            Sufficient statistics of the distribution
        """
        return jnp.array([x])

    @classmethod
    def log_partition(cls, rate=1.0, **kwargs):
        """Log partition function of the exponential distribution.

        Parameters
        ----------
        rate : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        log_partition : float
            Log partition function of the distribution
        """
        return -jnp.log(rate)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of exponential distribution parameters.

        The MLE for the exponential distribution has a closed-form solution:
        - rate = 1 / mean(data)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (rate,)
        """
        data = jnp.asarray(data)
        data = jnp.reshape(data, (-1,))
        dtype = data.dtype

        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((-1,))
            if weights.shape[0] != data.shape[0]:
                raise ValueError("weights must have the same length as data")
            weights = jnp.clip(weights, 0)
            total = jnp.sum(weights)
            total = jnp.where(total > 0, total, jnp.asarray(data.shape[0], dtype=dtype))
            weights = weights / total
            mean = jnp.sum(weights * data)
        else:
            mean = jnp.mean(data)
        rate = 1.0 / jnp.maximum(mean, jnp.asarray(1e-12, dtype=dtype))
        return (rate,)


expon = expon_gen(name="expon")
