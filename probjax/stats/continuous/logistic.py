"""
Logistic Distribution (:mod:`probjax.stats.logistic`)
====================================================

This module contains the Logistic distribution.
"""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

import jax
from jax.scipy.stats import logistic as _logistic

__all__ = ["logistic"]


class logistic_gen(rv_continuous, rv_exponential_family):
    """Logistic continuous random variable.

    The logistic distribution is a continuous probability distribution whose
    cumulative distribution function is the logistic function. The probability
    density function is:

    .. math::
        f(x; \mu, s) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}

    Parameters
    ----------
    loc : float, optional
        Location parameter (mean). Default is 0.
    scale : float, optional
        Scale parameter (standard deviation * pi/sqrt(3)). Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, **kwargs):
        """Support of the logistic distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the logistic distribution."""
        return _logistic.pdf(x, loc=loc, scale=scale)

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the logistic distribution."""
        return _logistic.logpdf(x, loc=loc, scale=scale)

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the logistic distribution."""
        return _logistic.cdf(x, loc=loc, scale=scale)

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the logistic distribution."""
        return _logistic.ppf(q, loc=loc, scale=scale)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        scale=1.0,
        **kwargs,
    ):
        """Random variates of the logistic distribution."""
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        return _logistic.rvs(loc=loc, scale=scale, size=shape + event_shape, key=rng)

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the logistic distribution."""
        return _logistic.sf(x, loc=loc, scale=scale)

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the logistic distribution."""
        return _logistic.isf(q, loc=loc, scale=scale)

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the logistic distribution."""
        return _logistic.logcdf(x, loc=loc, scale=scale)

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, **kwargs):
        """Mean of the logistic distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, **kwargs):
        """Mode of the logistic distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, **kwargs):
        """Variance of the logistic distribution."""
        return (jnp.pi * scale) ** 2 / 3

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the logistic distribution."""
        return jnp.log(scale) + 2

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the logistic distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return jnp.asarray(loc)
        elif n == 2:
            return loc**2 + (jnp.pi * scale) ** 2 / 3
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the logistic distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the logistic distribution."""
        return jnp.ones_like(loc) * 1.2

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the logistic distribution."""
        return jnp.array([loc / scale, -1.0 / scale])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the logistic distribution."""
        return jnp.array([x, jnp.abs(x)])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the logistic distribution."""
        return jnp.log(scale) + loc / scale


logistic = logistic_gen(name="logistic")
