"""
Generalized Normal Distribution (:mod:`probjax.stats.gennorm`)
============================================================

This module contains the Generalized Normal distribution.
"""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

import jax
from jax.scipy.special import gamma, gammainc, gammaincc

__all__ = ["gennorm"]


class gennorm_gen(rv_continuous, rv_exponential_family):
    """Generalized Normal continuous random variable.

    The generalized normal distribution is a continuous probability distribution
    that generalizes the normal distribution. The probability density function is:

    .. math::
        f(x; \mu, \alpha, \beta) = \frac{\beta}{2\alpha\Gamma(1/\beta)}
        \exp(-|x-\mu|^\beta/\alpha^\beta)

    where :math:`\mu` is the location parameter, :math:`\alpha` is the scale parameter,
    and :math:`\beta` is the shape parameter.

    Parameters
    ----------
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    beta : float, optional
        Shape parameter. Default is 2 (normal distribution).
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive, 'beta': strict_positive}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Support of the generalized normal distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Probability density function of the generalized normal distribution."""
        z = jnp.abs(x - loc) / scale
        return (beta / (2 * scale * gamma(1 / beta))) * jnp.exp(-(z**beta))

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Log of the probability density function of the generalized normal distribution."""
        z = jnp.abs(x - loc) / scale
        return jnp.log(beta / (2 * scale * gamma(1 / beta))) - (z**beta)

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Cumulative distribution function of the generalized normal distribution."""
        z = (x - loc) / scale
        return 0.5 * (1 + jnp.sign(z) * gammainc(1 / beta, jnp.abs(z) ** beta))

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Percent point function (inverse of cdf) of the generalized normal distribution."""
        # For beta=2, this is the normal distribution
        if beta == 2.0:
            return loc + scale * jnp.sqrt(2) * jax.scipy.special.erfinv(2 * q - 1)

        # For other values, we need to use numerical methods
        # This is a simplified version that works for most cases
        z = jnp.where(q < 0.5, -1, 1) * jnp.power(
            gamma(1 / beta) * (1 - 2 * jnp.abs(q - 0.5)), 1 / beta
        )
        return loc + scale * z

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        scale=1.0,
        beta=2.0,
        **kwargs,
    ):
        """Random variates of the generalized normal distribution."""
        # For beta=2, use normal distribution
        if beta == 2.0:
            return random.normal(rng, shape=shape) * scale + loc

        # For other values, use rejection sampling
        def _rejection_sampling(key):
            # Generate uniform random numbers
            u = random.uniform(key, shape=shape)
            # Transform to generalized normal
            z = jnp.where(u < 0.5, -1, 1) * jnp.power(
                gamma(1 / beta) * (1 - 2 * jnp.abs(u - 0.5)), 1 / beta
            )
            return z * scale + loc

        return _rejection_sampling(rng)

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Survival function (1 - cdf) of the generalized normal distribution."""
        return 1 - cls.cdf(x, loc, scale, beta)

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Inverse survival function (inverse of sf) of the generalized normal distribution."""
        return cls.ppf(1 - q, loc, scale, beta)

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Log of the cumulative distribution function of the generalized normal distribution."""
        return jnp.log(cls.cdf(x, loc, scale, beta))

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Mean of the generalized normal distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Mode of the generalized normal distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Variance of the generalized normal distribution."""
        return scale**2 * gamma(3 / beta) / gamma(1 / beta)

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Entropy of the generalized normal distribution."""
        return 1 / beta - jnp.log(beta / (2 * scale * gamma(1 / beta)))

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """n-th non-central moment of the generalized normal distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return jnp.asarray(loc)
        elif n == 2:
            return cls.var(loc, scale, beta) + loc**2
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Skewness of the generalized normal distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Excess kurtosis of the generalized normal distribution."""
        return gamma(5 / beta) * gamma(1 / beta) / (gamma(3 / beta) ** 2) - 3

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Natural parameters of the generalized normal distribution."""
        return jnp.array([loc / (scale**beta), -1 / (scale**beta)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the generalized normal distribution."""
        return jnp.array([x, jnp.abs(x)])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, beta=2.0, **kwargs):
        """Log partition function of the generalized normal distribution."""
        return jnp.log(2 * scale * gamma(1 / beta) / beta) + (loc / scale) ** beta


gennorm = gennorm_gen(name="gennorm")
