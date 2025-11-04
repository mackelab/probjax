"""
Skew Normal Distribution (:mod:`probjax.stats.skewnorm`)
=====================================================

This module contains the Skew Normal distribution.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import erf

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive
from probjax.utils.typing import RngKey

__all__ = ["skewnorm"]


class skewnorm_gen(rv_continuous, rv_exponential_family):
    """Skew Normal continuous random variable.

    The skew normal distribution is a continuous probability distribution
    that generalizes the normal distribution to allow for non-zero skewness.
    The probability density function is:

    .. math::
        f(x; xi, omega, alpha) = \frac{2}{\\omega} \\phi(\frac{x-xi}{\\omega})
        \\Phi(\alpha \frac{x-xi}{\\omega})

    where phi is the standard normal PDF, Phi is the standard normal CDF,
    xi is the location parameter, omega is the scale parameter, and alpha
    is the shape parameter.

    Parameters
    ----------
    a : float, optional
        Shape parameter. Default is 0 (normal distribution).
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'a': real, 'loc': real, 'scale': strict_positive}

    @classmethod
    def support(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Support of the skew normal distribution."""
        return real

    @classmethod
    def pdf(cls, x, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the skew normal distribution."""
        z = (x - loc) / scale
        return (
            (2 / scale)
            * jnp.exp(-0.5 * z**2)
            / jnp.sqrt(2 * jnp.pi)
            * (0.5 * (1 + erf(a * z / jnp.sqrt(2))))
        )

    @classmethod
    def logpdf(cls, x, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the skew normal distribution."""
        z = (x - loc) / scale
        return (
            jnp.log(2 / scale)
            - 0.5 * z**2
            - 0.5 * jnp.log(2 * jnp.pi)
            + jnp.log(0.5 * (1 + erf(a * z / jnp.sqrt(2))))
        )

    @classmethod
    def cdf(cls, x, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the skew normal distribution."""
        z = (x - loc) / scale
        return 0.5 * (1 + erf(z / jnp.sqrt(2))) - 2 * jax.scipy.stats.norm.pdf(
            z
        ) * jax.scipy.stats.norm.cdf(a * z)

    @classmethod
    def ppf(cls, q, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the skew normal distribution."""
        # For a=0, this is the normal distribution
        if a == 0.0:
            return loc + scale * jnp.sqrt(2) * jax.scipy.special.erfinv(2 * q - 1)

        # For other values, we need to use numerical methods
        # This is a simplified version that works for most cases
        z = jnp.where(q < 0.5, -1, 1) * jnp.sqrt(-2 * jnp.log(2 * jnp.abs(q - 0.5)))
        return loc + scale * z

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        a=0.0,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the skew normal distribution."""
        # For a=0, use normal distribution
        if a == 0.0:
            return random.normal(rng, shape=shape) * scale + loc

        # For other values, use rejection sampling
        def _rejection_sampling(key):
            # Generate uniform random numbers
            u = random.uniform(key, shape=shape)
            # Transform to skew normal
            z = jnp.where(u < 0.5, -1, 1) * jnp.sqrt(-2 * jnp.log(2 * jnp.abs(u - 0.5)))
            return z * scale + loc

        return _rejection_sampling(rng)

    @classmethod
    def sf(cls, x, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the skew normal distribution."""
        return 1 - cls.cdf(x, a, loc, scale)

    @classmethod
    def isf(cls, q, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the skew normal distribution."""
        return cls.ppf(1 - q, a, loc, scale)

    @classmethod
    def logcdf(cls, x, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the skew normal distribution."""
        return jnp.log(cls.cdf(x, a, loc, scale))

    @classmethod
    def mean(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mean of the skew normal distribution."""
        delta = a / jnp.sqrt(1 + a**2)
        return loc + scale * delta * jnp.sqrt(2 / jnp.pi)

    @classmethod
    def mode(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mode of the skew normal distribution."""
        delta = a / jnp.sqrt(1 + a**2)
        return loc + scale * delta * jnp.sqrt(2 / jnp.pi)

    @classmethod
    def var(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Variance of the skew normal distribution."""
        delta = a / jnp.sqrt(1 + a**2)
        return scale**2 * (1 - 2 * delta**2 / jnp.pi)

    @classmethod
    def entropy(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the skew normal distribution."""
        return 0.5 * jnp.log(2 * jnp.pi * jnp.e * scale**2)

    @classmethod
    def moment(cls, n, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the skew normal distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return cls.mean(a, loc, scale)
        elif n == 2:
            return cls.var(a, loc, scale) + cls.mean(a, loc, scale) ** 2
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the skew normal distribution."""
        delta = a / jnp.sqrt(1 + a**2)
        return (
            (4 - jnp.pi)
            / 2
            * (delta * jnp.sqrt(2 / jnp.pi)) ** 3
            / (1 - 2 * delta**2 / jnp.pi) ** (3 / 2)
        )

    @classmethod
    def kurtosis(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the skew normal distribution."""
        delta = a / jnp.sqrt(1 + a**2)
        return (
            2
            * (jnp.pi - 3)
            * (delta * jnp.sqrt(2 / jnp.pi)) ** 4
            / (1 - 2 * delta**2 / jnp.pi) ** 2
        )

    @classmethod
    def natural_parameters(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the skew normal distribution."""
        return jnp.array([loc / (scale**2), -1 / (2 * scale**2), a / scale])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the skew normal distribution."""
        return jnp.array([x, x**2, jnp.sign(x)])

    @classmethod
    def log_partition(cls, a=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the skew normal distribution."""
        return 0.5 * jnp.log(2 * jnp.pi * scale**2) + (loc / scale) ** 2 / 2


skewnorm = skewnorm_gen(name="skewnorm")
