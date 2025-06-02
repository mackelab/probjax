"""
Truncated Normal Distribution (:mod:`probjax.stats.truncnorm`)
=============================================================

This module contains the Truncated Normal distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax.scipy.stats import truncnorm as _truncnorm
from jaxtyping import PRNGKeyArray

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

__all__ = ["truncnorm"]


class truncnorm_gen(rv_continuous, rv_exponential_family):
    """Truncated Normal continuous random variable.

    The truncated normal distribution is a normal distribution that is bounded
    on both sides. The probability density function is:

    .. math::
        f(x; \\mu, \\sigma, a, b) = \frac{\\phi(\frac{x-\\mu}{\\sigma})}
        {\\sigma(\\Phi(\frac{b-\\mu}{\\sigma}) - \\Phi(\frac{a-\\mu}{\\sigma}))}

    where :math:`\\phi` is the standard normal PDF and :math:`\\Phi` is the standard
    normal CDF.

    Parameters
    ----------
    loc : float, optional
        Mean of the distribution. Default is 0.
    scale : float, optional
        Standard deviation of the distribution. Default is 1.
    a : float, optional
        Lower bound of the truncation. Default is -inf.
    b : float, optional
        Upper bound of the truncation. Default is inf.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive, 'a': real, 'b': real}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Support of the truncated normal distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Probability density function of the truncated normal distribution."""
        return _truncnorm.pdf(
            x, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log of the probability density function of the truncated normal distribution."""
        return _truncnorm.logpdf(
            x, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Cumulative distribution function of the truncated normal distribution."""
        return _truncnorm.cdf(
            x, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Percent point function (inverse of cdf) of the truncated normal distribution."""
        return _truncnorm.ppf(
            q, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        loc=0.0,
        scale=1.0,
        a=-jnp.inf,
        b=jnp.inf,
        **kwargs,
    ):
        """Random variates of the truncated normal distribution."""
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        event_shape = jnp.broadcast_shapes(loc.shape, scale.shape, a.shape, b.shape)
        return _truncnorm.rvs(
            a=(a - loc) / scale,
            b=(b - loc) / scale,
            loc=loc,
            scale=scale,
            size=shape + event_shape,
            key=rng,
        )

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Survival function (1 - cdf) of the truncated normal distribution."""
        return _truncnorm.sf(
            x, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Inverse survival function (inverse of sf) of the truncated normal distribution."""
        return _truncnorm.isf(
            q, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log of the cumulative distribution function of the truncated normal distribution."""
        return _truncnorm.logcdf(
            x, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Mean of the truncated normal distribution."""
        return _truncnorm.mean(
            a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Mode of the truncated normal distribution."""
        return jnp.clip(loc, a, b)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Variance of the truncated normal distribution."""
        return _truncnorm.var(
            a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Entropy of the truncated normal distribution."""
        return _truncnorm.entropy(
            a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """n-th non-central moment of the truncated normal distribution."""
        return _truncnorm.moment(
            n, a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Skewness of the truncated normal distribution."""
        return _truncnorm.skew(
            a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Excess kurtosis of the truncated normal distribution."""
        return _truncnorm.kurtosis(
            a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale
        )

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Natural parameters of the truncated normal distribution."""
        var = scale**2
        return jnp.array([loc / var, -1.0 / (2.0 * var)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the truncated normal distribution."""
        return jnp.array([x, x**2])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log partition function of the truncated normal distribution."""
        var = scale**2
        return 0.5 * jnp.log(2 * jnp.pi * var) + (loc**2) / (2 * var)


truncnorm = truncnorm_gen(name="truncnorm")
