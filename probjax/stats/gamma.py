"""
Gamma Distribution (:mod:`probjax.stats.gamma`)
==================================================

This module contains the Gamma distribution.
"""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

import jax
from jax.scipy.stats import gamma as _gamma
from jax.scipy.special import digamma, gammaln

__all__ = ["gamma"]


class gamma_gen(rv_continuous, rv_exponential_family):
    """Gamma continuous random variable.

    The gamma distribution with shape parameter `alpha` and rate parameter `beta`.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter. Default is 1.
    beta : float, optional
        Rate parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'alpha': strict_positive, 'beta': strict_positive}

    @classmethod
    def support(cls, alpha=1.0, beta=1.0, **kwargs):
        """Support of the gamma distribution."""
        return strict_positive

    @classmethod
    def pdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Probability density function of the gamma distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, alpha, beta, **kwargs))

    @classmethod
    def logpdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the probability density function of the gamma distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        # JAX implementation has scale (1/rate) parameter
        scale = 1.0 / beta
        return _gamma.logpdf(x, alpha, scale=scale)

    @classmethod
    def cdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Cumulative distribution function of the gamma distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        scale = 1.0 / beta
        return _gamma.cdf(x, alpha, scale=scale)

    @classmethod
    def logcdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the cumulative distribution function of the gamma distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        scale = 1.0 / beta
        return _gamma.logcdf(x, alpha, scale=scale)

    @classmethod
    def ppf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the gamma distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        scale = 1.0 / beta
        return _gamma.ppf(q, alpha, scale=scale)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        alpha=1.0,
        beta=1.0,
        **kwargs,
    ) -> Float[Array, "..."]:
        """Random variates of the gamma distribution.

        Parameters
        ----------
        rng : PRNGKeyArray
            JAX PRNG key for random number generation
        shape : tuple of ints, optional
            Output shape. Default is (), meaning a single value.
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        event_shape = jnp.broadcast_shapes(alpha.shape, beta.shape)
        return random.gamma(rng, alpha, shape=shape + event_shape) / beta

    @classmethod
    def sf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Survival function (1 - cdf) of the gamma distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        scale = 1.0 / beta
        return _gamma.sf(x, alpha, scale=scale)

    @classmethod
    def isf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the gamma distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        scale = 1.0 / beta
        return _gamma.isf(q, alpha, scale=scale)

    @classmethod
    def mean(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mean of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        return alpha / beta

    @classmethod
    def mode(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mode of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
        valid = alpha > 1
        return jnp.where(valid, (alpha - 1) / beta, jnp.zeros_like(alpha))

    @classmethod
    def var(cls, alpha=1.0, beta=1.0, **kwargs):
        """Variance of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        return alpha / beta**2

    @classmethod
    def entropy(cls, alpha=1.0, beta=1.0, **kwargs):
        """Entropy of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return alpha - jnp.log(beta) + gammaln(alpha) + (1 - alpha) * digamma(alpha)

    @classmethod
    def moment(cls, n, alpha=1.0, beta=1.0, **kwargs):
        """n-th non-central moment of the gamma distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        # n-th moment: E[X^n] = Γ(α+n)/Γ(α) * β^(-n)
        n = jnp.asarray(n)
        return jnp.exp(gammaln(alpha + n) - gammaln(alpha)) / beta**n

    @classmethod
    def skew(cls, alpha=1.0, beta=1.0, **kwargs):
        """Skewness of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return 2.0 / jnp.sqrt(alpha)

    @classmethod
    def kurtosis(cls, alpha=1.0, beta=1.0, **kwargs):
        """Excess kurtosis of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        return 6.0 / alpha

    @classmethod
    def natural_parameters(cls, alpha=1.0, beta=1.0, **kwargs):
        """Natural parameters of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        natural_parameters : tuple
            Natural parameters of the distribution
        """
        return jnp.array([alpha - 1, -beta])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the gamma distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : tuple
            Sufficient statistics of the distribution
        """
        return jnp.array([jnp.log(x), x])

    @classmethod
    def log_partition(cls, alpha=1.0, beta=1.0, **kwargs):
        """Log partition function of the gamma distribution.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter. Default is 1.
        beta : float, optional
            Rate parameter. Default is 1.

        Returns
        -------
        log_partition : float
            Log partition function of the distribution
        """
        return gammaln(alpha) - alpha * jnp.log(beta)


gamma = gamma_gen(name="gamma")
