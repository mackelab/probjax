"""
Normal Distribution (:mod:`probjax.stats.norm`)
==================================================

This module contains the Normal (Gaussian) distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm as _norm

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive
from probjax.stats.utils import (
    flatten_samples,
    loc_scale_sample,
    mean_and_var_1d,
    normalize_sample_weights,
)
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["norm"]


class norm_gen(rv_continuous, rv_exponential_family):
    """Normal continuous random variable.

    The normal distribution with mean `loc` and standard deviation `scale`.

    Parameters
    ----------
    loc : float, optional
        Mean of the distribution. Default is 0.
    scale : float, optional
        Standard deviation of the distribution. Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive}

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the normal distribution."""
        return _norm.pdf(x, loc, scale)

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the normal distribution."""
        return _norm.logpdf(x, loc, scale)

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the normal distribution."""
        return _norm.cdf(x, loc, scale)

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the normal distribution."""
        return _norm.ppf(q, loc, scale)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the normal distribution."""
        return loc_scale_sample(rng, random.normal, shape=shape, loc=loc, scale=scale)

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the normal distribution."""
        return _norm.sf(x, loc, scale)

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the normal distribution."""
        return _norm.isf(q, loc, scale)

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the normal distribution."""
        return _norm.logcdf(x, loc, scale)

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, **kwargs):
        """Mean of the normal distribution."""
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, **kwargs):
        """Mode of the normal distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, **kwargs):
        """Variance of the normal distribution."""
        return jnp.asarray(scale) ** 2

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the normal distribution."""
        return 0.5 * jnp.log(2 * jnp.pi * jnp.e * scale**2)

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the normal distribution."""
        # Use Hermite polynomials to compute moments efficiently
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return jnp.asarray(loc)
        elif n == 2:
            return loc**2 + scale**2
        else:
            # Recursive formula for higher moments
            def hermite_polynomial(n, x):
                if n == 0:
                    return jnp.ones_like(x)
                elif n == 1:
                    return x
                else:
                    return x * hermite_polynomial(n - 1, x) - (
                        n - 1
                    ) * hermite_polynomial(n - 2, x)

            # Expected value of Hermite polynomial with normal distribution
            # For standard normal (0, 1), E[H_n(X)] = 0 if n is odd, and n! if n is even
            # For non-standard normal, transform using E[(X-μ)/σ]
            standardized_moment = jnp.where(
                n % 2 == 0, jnp.prod(jnp.arange(1, n + 1, 2)), 0.0
            )

            return jnp.sum([
                jnp.array(
                    jnp.math.comb(n, k)
                    * loc ** (n - k)
                    * scale**k
                    * standardized_moment
                )
                for k in range(n + 1)
            ])

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the normal distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the normal distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the normal distribution."""
        var = scale**2
        return jnp.array([loc / var, -1.0 / (2.0 * var)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the normal distribution."""
        return jnp.array([x, x**2])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the normal distribution."""
        return 0.5 * (loc**2 / scale**2 + jnp.log(2 * jnp.pi * scale**2))

    @classmethod
    def fit(cls, data: ArrayLike, weights: Optional[ArrayLike] = None, **kwds):
        """Maximum likelihood estimation of normal distribution parameters.

        The MLE for the normal distribution has a closed-form solution:
        - loc = mean(data)
        - scale = std(data)
        """
        data = flatten_samples(data)
        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=data.dtype,
        )
        loc, var = mean_and_var_1d(data, weights_arr)

        scale = jnp.sqrt(jnp.maximum(var, jnp.asarray(1e-6, dtype=data.dtype)))
        return loc, scale


norm = norm_gen(name="norm")
