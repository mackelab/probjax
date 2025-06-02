"""
Wrapped Cauchy Distribution (:mod:`probjax.stats.wrapcauchy`)
===========================================================

This module contains the Wrapped Cauchy distribution.
"""

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict, Optional

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

import jax

__all__ = ["wrapcauchy"]


class wrapcauchy_gen(rv_continuous, rv_exponential_family):
    """Wrapped Cauchy continuous random variable.

    The wrapped Cauchy distribution is a continuous probability distribution on the circle.
    It is the circular analogue of the Cauchy distribution. The probability density function is:

    .. math::
        f(x; \mu, \gamma) = \frac{1}{2\pi} \frac{1-\gamma^2}{1+\gamma^2-2\gamma\cos(x-\mu)}

    where :math:`\mu` is the location parameter and :math:`\gamma` is the concentration parameter.

    Parameters
    ----------
    loc : float, optional
        Location parameter (mean direction). Default is 0.
    gamma : float, optional
        Concentration parameter. Default is 0.5.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'gamma': strict_positive}

    @classmethod
    def support(cls, loc=0.0, gamma=0.5, **kwargs):
        """Support of the wrapped Cauchy distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, gamma=0.5, **kwargs):
        """Probability density function of the wrapped Cauchy distribution."""
        return (1 - gamma**2) / (
            2 * jnp.pi * (1 + gamma**2 - 2 * gamma * jnp.cos(x - loc))
        )

    @classmethod
    def logpdf(cls, x, loc=0.0, gamma=0.5, **kwargs):
        """Log of the probability density function of the wrapped Cauchy distribution."""
        return (
            jnp.log(1 - gamma**2)
            - jnp.log(2 * jnp.pi)
            - jnp.log(1 + gamma**2 - 2 * gamma * jnp.cos(x - loc))
        )

    @classmethod
    def cdf(cls, x, loc=0.0, gamma=0.5, **kwargs):
        """Cumulative distribution function of the wrapped Cauchy distribution."""
        z = (x - loc) % (2 * jnp.pi)
        return (
            jnp.arctan2((1 + gamma) * jnp.sin(z), (1 + gamma) * jnp.cos(z) - 2 * gamma)
            + jnp.pi
        ) / (2 * jnp.pi)

    @classmethod
    def ppf(cls, q, loc=0.0, gamma=0.5, **kwargs):
        """Percent point function (inverse of cdf) of the wrapped Cauchy distribution."""
        z = 2 * jnp.pi * q
        return (
            jnp.arctan2(
                (1 - gamma**2) * jnp.sin(z), (1 + gamma**2) * jnp.cos(z) - 2 * gamma
            )
            + loc
        ) % (2 * jnp.pi)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        gamma=0.5,
        **kwargs,
    ):
        """Random variates of the wrapped Cauchy distribution."""
        # Generate Cauchy random variates and wrap them
        u = random.uniform(rng, shape=shape)
        return (2 * jnp.arctan(gamma * jnp.tan(jnp.pi * (u - 0.5))) + loc) % (
            2 * jnp.pi
        )

    @classmethod
    def sf(cls, x, loc=0.0, gamma=0.5, **kwargs):
        """Survival function (1 - cdf) of the wrapped Cauchy distribution."""
        return 1 - cls.cdf(x, loc, gamma)

    @classmethod
    def isf(cls, q, loc=0.0, gamma=0.5, **kwargs):
        """Inverse survival function (inverse of sf) of the wrapped Cauchy distribution."""
        return cls.ppf(1 - q, loc, gamma)

    @classmethod
    def logcdf(cls, x, loc=0.0, gamma=0.5, **kwargs):
        """Log of the cumulative distribution function of the wrapped Cauchy distribution."""
        return jnp.log(cls.cdf(x, loc, gamma))

    @classmethod
    def mean(cls, loc=0.0, gamma=0.5, **kwargs):
        """Mean of the wrapped Cauchy distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, gamma=0.5, **kwargs):
        """Mode of the wrapped Cauchy distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, gamma=0.5, **kwargs):
        """Variance of the wrapped Cauchy distribution."""
        return 2 * jnp.pi**2 * gamma / (1 - gamma**2)

    @classmethod
    def entropy(cls, loc=0.0, gamma=0.5, **kwargs):
        """Entropy of the wrapped Cauchy distribution."""
        return jnp.log(2 * jnp.pi * (1 - gamma**2))

    @classmethod
    def moment(cls, n, loc=0.0, gamma=0.5, **kwargs):
        """n-th non-central moment of the wrapped Cauchy distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return jnp.asarray(loc)
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, loc=0.0, gamma=0.5, **kwargs):
        """Skewness of the wrapped Cauchy distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def kurtosis(cls, loc=0.0, gamma=0.5, **kwargs):
        """Excess kurtosis of the wrapped Cauchy distribution."""
        return jnp.ones_like(loc) * 2

    @classmethod
    def natural_parameters(cls, loc=0.0, gamma=0.5, **kwargs):
        """Natural parameters of the wrapped Cauchy distribution."""
        return jnp.array([gamma * jnp.cos(loc), gamma * jnp.sin(loc)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the wrapped Cauchy distribution."""
        return jnp.array([jnp.cos(x), jnp.sin(x)])

    @classmethod
    def log_partition(cls, loc=0.0, gamma=0.5, **kwargs):
        """Log partition function of the wrapped Cauchy distribution."""
        return jnp.log(2 * jnp.pi * (1 - gamma**2))


wrapcauchy = wrapcauchy_gen(name="wrapcauchy")
