"""
Von Mises Distribution (:mod:`probjax.stats.vonmises`)
====================================================

This module contains the Von Mises distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.special import i0, i1
from jaxtyping import PRNGKeyArray

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

__all__ = ["vonmises"]


class vonmises_gen(rv_continuous, rv_exponential_family):
    """Von Mises continuous random variable.

    The von Mises distribution is a continuous probability distribution on the circle.
    It is a close approximation to the wrapped normal distribution, which is the
    circular analogue of the normal distribution. The probability density function is:

    .. math::
        f(x; \\mu, \\kappa) = \frac{e^{\\kappa\\cos(x-\\mu)}}{2\\pi I_0(\\kappa)}

    where :math:`\\mu` is the location parameter and :math:`\\kappa` is the concentration
    parameter, and :math:`I_0` is the modified Bessel function of order 0.

    Parameters
    ----------
    loc : float, optional
        Location parameter (mean direction). Default is 0.
    kappa : float, optional
        Concentration parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'kappa': strict_positive}

    @classmethod
    def support(cls, loc=0.0, kappa=1.0, **kwargs):
        """Support of the von Mises distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, kappa=1.0, **kwargs):
        """Probability density function of the von Mises distribution."""
        return jnp.exp(kappa * jnp.cos(x - loc)) / (2 * jnp.pi * i0(kappa))

    @classmethod
    def logpdf(cls, x, loc=0.0, kappa=1.0, **kwargs):
        """Log of the probability density function of the von Mises distribution."""
        return kappa * jnp.cos(x - loc) - jnp.log(2 * jnp.pi * i0(kappa))

    @classmethod
    def cdf(cls, x, loc=0.0, kappa=1.0, **kwargs):
        """Cumulative distribution function of the von Mises distribution."""
        # The CDF is not available in closed form
        # We use a numerical approximation
        z = (x - loc) % (2 * jnp.pi)
        return z / (2 * jnp.pi) + jnp.sin(z) * i1(kappa) / (2 * jnp.pi * i0(kappa))

    @classmethod
    def ppf(cls, q, loc=0.0, kappa=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the von Mises distribution."""
        # The PPF is not available in closed form
        # We use a numerical approximation
        z = 2 * jnp.pi * q
        return (z + jnp.arcsin(2 * jnp.pi * q * i0(kappa) / i1(kappa))) % (
            2 * jnp.pi
        ) + loc

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        kappa=1.0,
        **kwargs,
    ):
        """Random variates of the von Mises distribution."""

        # Use rejection sampling
        def _rejection_sampling(key):
            # Generate uniform random numbers
            u = random.uniform(key, shape=shape)
            v = random.uniform(key, shape=shape)
            # Transform to von Mises
            z = jnp.cos(jnp.pi * u)
            f = (1 + kappa * z) / (kappa + z)
            c = kappa * jnp.sqrt((1 - f) / (1 + f))
            return jnp.where(v < c, jnp.arccos(f), jnp.pi - jnp.arccos(f)) + loc

        return _rejection_sampling(rng)

    @classmethod
    def sf(cls, x, loc=0.0, kappa=1.0, **kwargs):
        """Survival function (1 - cdf) of the von Mises distribution."""
        return 1 - cls.cdf(x, loc, kappa)

    @classmethod
    def isf(cls, q, loc=0.0, kappa=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the von Mises distribution."""
        return cls.ppf(1 - q, loc, kappa)

    @classmethod
    def logcdf(cls, x, loc=0.0, kappa=1.0, **kwargs):
        """Log of the cumulative distribution function of the von Mises distribution."""
        return jnp.log(cls.cdf(x, loc, kappa))

    @classmethod
    def mean(cls, loc=0.0, kappa=1.0, **kwargs):
        """Mean of the von Mises distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, kappa=1.0, **kwargs):
        """Mode of the von Mises distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, kappa=1.0, **kwargs):
        """Variance of the von Mises distribution."""
        return 1 - i1(kappa) / i0(kappa)

    @classmethod
    def entropy(cls, loc=0.0, kappa=1.0, **kwargs):
        """Entropy of the von Mises distribution."""
        return -kappa * i1(kappa) / i0(kappa) + jnp.log(2 * jnp.pi * i0(kappa))

    @classmethod
    def moment(cls, n, loc=0.0, kappa=1.0, **kwargs):
        """n-th non-central moment of the von Mises distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return jnp.asarray(loc)
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, loc=0.0, kappa=1.0, **kwargs):
        """Skewness of the von Mises distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def kurtosis(cls, loc=0.0, kappa=1.0, **kwargs):
        """Excess kurtosis of the von Mises distribution."""
        return -2 * (i1(kappa) / i0(kappa)) ** 2

    @classmethod
    def natural_parameters(cls, loc=0.0, kappa=1.0, **kwargs):
        """Natural parameters of the von Mises distribution."""
        return jnp.array([kappa * jnp.cos(loc), kappa * jnp.sin(loc)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the von Mises distribution."""
        return jnp.array([jnp.cos(x), jnp.sin(x)])

    @classmethod
    def log_partition(cls, loc=0.0, kappa=1.0, **kwargs):
        """Log partition function of the von Mises distribution."""
        return jnp.log(2 * jnp.pi * i0(kappa))


vonmises = vonmises_gen(name="vonmises")
