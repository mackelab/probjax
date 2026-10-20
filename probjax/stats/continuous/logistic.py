"""
Logistic Distribution (:mod:`probjax.stats.logistic`)
====================================================

This module contains the Logistic distribution.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive
from probjax.stats.utils import (
    clip_prob,
    flatten_samples,
    mean_and_var_1d,
    normalize_sample_weights,
)
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["logistic"]


class logistic_gen(rv_continuous, rv_exponential_family):
    """Logistic continuous random variable.

    The logistic distribution is a continuous probability distribution whose
    cumulative distribution function is the logistic function. The probability
    density function is:

    .. math::
        f(x; \\mu, s) = \frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2}

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
    def pdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Probability density function of the logistic distribution."""
        x_arr = jnp.asarray(x)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        z = (x_arr - loc_arr) / scale_arr
        exp_neg_z = jnp.exp(-z)
        denom = scale_arr * (1.0 + exp_neg_z) ** 2
        return exp_neg_z / denom

    @classmethod
    def logpdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Log of the probability density function of the logistic distribution."""
        x_arr = jnp.asarray(x)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        z = (x_arr - loc_arr) / scale_arr
        return -z - jnp.log(scale_arr) - 2.0 * jax.nn.softplus(-z)

    @classmethod
    def cdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Cumulative distribution function of the logistic distribution."""
        x_arr = jnp.asarray(x)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        z = (x_arr - loc_arr) / scale_arr
        return jax.nn.sigmoid(z)

    @classmethod
    def ppf(
        cls,
        q: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Percent point function (inverse of cdf) of the logistic distribution."""
        q_arr = jnp.asarray(q)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        q_clipped = clip_prob(q_arr)
        return loc_arr + scale_arr * jnp.log(q_clipped / (1.0 - q_clipped))

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the logistic distribution."""
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(loc_arr.shape, scale_arr.shape)
        u = random.uniform(rng, shape=shape + event_shape, minval=0.0, maxval=1.0)
        u = clip_prob(u)
        return loc_arr + scale_arr * jnp.log(u / (1.0 - u))

    @classmethod
    def sf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Survival function (1 - cdf) of the logistic distribution."""
        return 1.0 - cls.cdf(x, loc=loc, scale=scale, **kwargs)

    @classmethod
    def isf(
        cls,
        q: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Inverse survival function (inverse of sf) of the logistic distribution."""
        return cls.ppf(1.0 - jnp.asarray(q), loc=loc, scale=scale, **kwargs)

    @classmethod
    def logcdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Log of the cumulative distribution function of the logistic distribution."""
        x_arr = jnp.asarray(x)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        z = (x_arr - loc_arr) / scale_arr
        return -jax.nn.softplus(-z)

    @classmethod
    def mean(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Mean of the logistic distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Mode of the logistic distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(
        cls, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0, **kwargs: Any
    ) -> ArrayLike:
        """Variance of the logistic distribution."""
        scale_arr = jnp.asarray(scale)
        return (jnp.pi * scale_arr) ** 2 / 3.0

    @classmethod
    def entropy(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Entropy of the logistic distribution."""
        scale_arr = jnp.asarray(scale)
        return jnp.log(scale_arr) + 2.0

    @classmethod
    def moment(
        cls,
        n: int,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """n-th non-central moment of the logistic distribution."""
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        n_int = int(n)
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        if n_int == 0:
            return jnp.ones_like(loc_arr)
        if n_int == 1:
            return loc_arr
        if n_int == 2:
            return loc_arr**2 + (jnp.pi * scale_arr) ** 2 / 3.0
        raise NotImplementedError(f"Moment of order {n_int} not implemented")

    @classmethod
    def skew(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Skewness of the logistic distribution."""
        return jnp.zeros_like(jnp.asarray(loc))

    @classmethod
    def kurtosis(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Excess kurtosis of the logistic distribution."""
        loc_arr = jnp.asarray(loc)
        return jnp.ones_like(loc_arr) * 1.2

    @classmethod
    def natural_parameters(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Natural parameters of the logistic distribution."""
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        return jnp.stack((loc_arr / scale_arr, -1.0 / scale_arr))

    @classmethod
    def sufficient_statistics(cls, x: ArrayLike, **kwargs: Any) -> ArrayLike:
        """Sufficient statistics of the logistic distribution."""
        x_arr = jnp.asarray(x)
        return jnp.stack((x_arr, jnp.abs(x_arr)))

    @classmethod
    def log_partition(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Log partition function of the logistic distribution."""
        loc_arr = jnp.asarray(loc)
        scale_arr = jnp.asarray(scale)
        return jnp.log(scale_arr) + loc_arr / scale_arr

    @classmethod
    def fit(
        cls,
        data,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Closed-form fit using (optionally weighted) mean and variance."""
        data = flatten_samples(data)
        dtype = data.dtype

        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=dtype,
        )
        loc, var = mean_and_var_1d(data, weights_arr)

        scale = (
            jnp.sqrt(jnp.maximum(var, jnp.asarray(1e-9, dtype=dtype)) * 3.0) / jnp.pi
        )
        scale = jnp.maximum(scale, jnp.asarray(1e-6, dtype=dtype))
        return loc, scale


logistic = logistic_gen(name="logistic")
