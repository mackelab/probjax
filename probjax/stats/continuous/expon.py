"""
Exponential Distribution (:mod:`probjax.stats.expon`)
==================================================

This module contains the Exponential distribution.
"""

from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import expon as _expon

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import positive, strict_positive
from probjax.stats.utils import weighted_mean
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["expon"]


def _scale(rate):
    """Convert rate to scale: the JAX implementation uses scale (1/rate)."""
    return 1.0 / rate


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
    def logpdf(cls, x: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Log of the probability density function of the exponential distribution."""
        scale = _scale(rate)
        return _expon.logpdf(x, loc=0.0, scale=scale)

    @classmethod
    def cdf(cls, x: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Cumulative distribution function of the exponential distribution."""
        scale = _scale(rate)
        return _expon.cdf(x, loc=0.0, scale=scale)

    @classmethod
    def logcdf(cls, x: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Log of the cumulative distribution function of the exponential
        distribution."""
        scale = _scale(rate)
        return _expon.logcdf(x, loc=0.0, scale=scale)

    @classmethod
    def ppf(cls, q: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Percent point function (inverse of cdf) of the exponential distribution."""
        scale = _scale(rate)
        return _expon.ppf(q, loc=0.0, scale=scale)

    @classmethod
    def _rvs_impl(
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
    def sf(cls, x: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Survival function (1 - cdf) of the exponential distribution."""
        scale = _scale(rate)
        return _expon.sf(x, loc=0.0, scale=scale)

    @classmethod
    def isf(cls, q: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Inverse survival function (inverse of sf) of the exponential distribution."""
        q = jnp.asarray(q)
        rate_arr = jnp.asarray(rate)
        return (
            -jnp.log(jnp.clip(q, a_min=jnp.finfo(q.dtype).tiny, a_max=1.0)) / rate_arr
        )

    @classmethod
    def mean(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Mean of the exponential distribution."""
        return jnp.asarray(1.0) / jnp.asarray(rate)

    @classmethod
    def mode(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Mode of the exponential distribution."""
        return jnp.zeros_like(rate)

    @classmethod
    def var(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Variance of the exponential distribution."""
        rate_arr = jnp.asarray(rate)
        return jnp.asarray(1.0) / (rate_arr**2)

    @classmethod
    def entropy(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Entropy of the exponential distribution."""
        return 1.0 - jnp.log(rate)

    @classmethod
    def moment(cls, n: ArrayLike, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """n-th non-central moment of the exponential distribution."""
        n = jnp.asarray(n)
        # Factorial moment: E[X^n] = n! / rate^n
        return jnp.exp(jnp.log(jnp.prod(jnp.arange(1, n + 1))) - n * jnp.log(rate))

    @classmethod
    def skew(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Skewness of the exponential distribution."""
        return 2.0 * jnp.ones_like(rate)  # Skewness is always 2

    @classmethod
    def kurtosis(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Excess kurtosis of the exponential distribution."""
        return 6.0 * jnp.ones_like(rate)  # Excess kurtosis is always 6

    @classmethod
    def natural_parameters(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Natural parameters of the exponential distribution."""
        rate = jnp.asarray(rate)
        return jnp.array([-rate])

    @classmethod
    def sufficient_statistics(cls, x: ArrayLike, **kwargs: Any) -> ArrayLike:
        """Sufficient statistics of the exponential distribution."""
        return jnp.array([x])

    @classmethod
    def log_partition(cls, rate: ArrayLike = 1.0, **kwargs: Any) -> ArrayLike:
        """Log partition function of the exponential distribution."""
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
        """
        data = jnp.asarray(data)
        mean = weighted_mean(data, weights)
        rate = 1.0 / jnp.maximum(mean, jnp.asarray(1e-12, dtype=data.dtype))
        return (rate,)


expon = expon_gen(name="expon")
