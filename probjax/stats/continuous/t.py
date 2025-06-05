"""
Student's t-Distribution (:mod:`probjax.stats.t`)
==================================================

This module contains the Student's t-distribution.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, PRNGKeyArray

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive, strict_positive_integer

__all__ = ["t"]


class t_gen(rv_continuous, rv_exponential_family):
    """Student's t-Distribution parameterized by `df`, `loc`, and `scale`.

    The Student's t-distribution with degrees of freedom `df`, location `loc`, and scale `scale`.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    """

    name = "t"
    parameters = {
        "df": strict_positive_integer,
        "loc": real,
        "scale": strict_positive,
    }

    @classmethod
    def support(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Support of the distribution."""
        return real

    @classmethod
    def pdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jax.scipy.stats.t.pdf(x, df, loc, scale)

    @classmethod
    def logpdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        return jax.scipy.stats.t.logpdf(x, df, loc, scale)

    @classmethod
    def cdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        raise NotImplementedError("CDF not implemented for t distribution")

    @classmethod
    def ppf(cls, q, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the Student's t-distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        raise NotImplementedError("PPF not implemented for t distribution")

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        df=1.0,
        loc=0.0,
        scale=1.0,
        **kwargs,
    ) -> Float[Array, "..."]:
        """Random variates of the Student's t-distribution.

        Parameters
        ----------
        rng : PRNGKeyArray
            JAX PRNG key for random number generation.
        shape : tuple of ints, optional
            Output shape. Default is ().
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        df = jnp.asarray(df)
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(df.shape, loc.shape, scale.shape)
        return random.t(rng, df, shape=shape + event_shape) * scale + loc

    @classmethod
    def sf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        return jax.scipy.stats.t.sf(x, df, loc, scale)

    @classmethod
    def isf(cls, q, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the Student's t-distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        return jax.scipy.stats.t.isf(q, df, loc, scale)

    @classmethod
    def logcdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        return jax.scipy.stats.t.logcdf(x, df, loc, scale)

    @classmethod
    def mean(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Mean of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        return jnp.where(df > 1, loc, jnp.nan)

    @classmethod
    def mode(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Mode of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
        # The mode of a t-distribution is always at its location parameter
        # This is because the PDF is symmetric around loc and has a single maximum
        return jnp.asarray(loc)

    @classmethod
    def var(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Variance of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        return jnp.where(
            df > 2,
            scale**2 * df / (df - 2),
            jnp.where(df > 1, jnp.inf, jnp.nan),
        )

    @classmethod
    def entropy(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return (
            jnp.log(scale)
            + 0.5 * (1 + jnp.log(df))
            + gammaln(0.5 * (df + 1))
            - gammaln(0.5 * df)
        )

    @classmethod
    def moment(cls, n, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the Student's t-distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return cls.mean(df=df, loc=loc)
        elif n == 2:
            return cls.var(df=df, loc=loc, scale=scale) + loc**2
        else:
            # For higher moments, we need to use numerical integration
            # This is a placeholder - in practice, you might want to implement
            # a more efficient method or use numerical integration
            return jnp.nan

    @classmethod
    def skew(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return jnp.where(df > 3, 0.0, jnp.nan)

    @classmethod
    def kurtosis(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        return jnp.where(df > 4, 6.0 / (df - 4), jnp.nan)

    @classmethod
    def natural_parameters(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        natural_parameters : tuple
            Natural parameters of the distribution
        """
        # The t-distribution is not a member of the exponential family
        # This is a placeholder that returns None
        return None

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the Student's t-distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : tuple
            Sufficient statistics of the distribution
        """
        # The t-distribution is not a member of the exponential family
        # This is a placeholder that returns None
        return None

    @classmethod
    def log_partition(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the Student's t-distribution.

        Parameters
        ----------
        df : float
            Degrees of freedom.
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        log_partition : float
            Log partition function of the distribution
        """
        # The t-distribution is not a member of the exponential family
        # This is a placeholder that returns None
        return None


t = t_gen(name="t")
