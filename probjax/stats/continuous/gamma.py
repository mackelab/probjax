"""
Gamma Distribution (:mod:`probjax.stats.gamma`)
==================================================

This module contains the Gamma distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import digamma, gammaln
from jax.scipy.stats import gamma as _gamma

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import strict_positive
from probjax.stats.utils import flatten_samples, weighted_mean
from probjax.utils.special import gammaincinv, gammainccinv
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["gamma"]


def _scale(beta):
    """Convert rate to scale: the JAX implementation uses scale (1/rate)."""
    return 1.0 / beta


class gamma_gen(rv_continuous, rv_exponential_family):
    """Gamma continuous random variable.

    The gamma distribution with shape parameter `alpha` and rate parameter `beta`.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter. Default is 1.
    beta : float, optional
        Rate parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'alpha': strict_positive, 'beta': strict_positive}
    parameter_aliases = {'concentration': 'alpha', 'rate': 'beta'}

    @classmethod
    def support(cls, alpha=1.0, beta=1.0, **kwargs):
        """Support of the gamma distribution."""
        return strict_positive

    @classmethod
    def logpdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the probability density function of the gamma distribution."""
        scale = _scale(beta)
        return _gamma.logpdf(x, alpha, scale=scale)

    @classmethod
    def cdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Cumulative distribution function of the gamma distribution."""
        scale = _scale(beta)
        return _gamma.cdf(x, alpha, scale=scale)

    @classmethod
    def logcdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the cumulative distribution function of the gamma distribution."""
        scale = _scale(beta)
        return _gamma.logcdf(x, alpha, scale=scale)

    @classmethod
    def ppf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the gamma distribution."""
        scale = _scale(beta)
        return gammaincinv(alpha, q) * scale

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        alpha=1.0,
        beta=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the gamma distribution."""
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        event_shape = jnp.broadcast_shapes(alpha.shape, beta.shape)
        return random.gamma(rng, alpha, shape=shape + event_shape) / beta

    @classmethod
    def sf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Survival function (1 - cdf) of the gamma distribution."""
        scale = _scale(beta)
        return _gamma.sf(x, alpha, scale=scale)

    @classmethod
    def isf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the gamma distribution."""
        return gammainccinv(alpha, q) / jnp.asarray(beta)

    @classmethod
    def mean(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mean of the gamma distribution."""
        return jnp.asarray(alpha) / jnp.asarray(beta)

    @classmethod
    def mode(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mode of the gamma distribution."""
        valid = alpha >= 1
        return jnp.where(valid, (alpha - 1) / beta, jnp.zeros_like(alpha))

    @classmethod
    def var(cls, alpha=1.0, beta=1.0, **kwargs):
        """Variance of the gamma distribution."""
        alpha_arr = jnp.asarray(alpha)
        beta_arr = jnp.asarray(beta)
        return alpha_arr / (beta_arr**2)

    @classmethod
    def entropy(cls, alpha=1.0, beta=1.0, **kwargs):
        """Entropy of the gamma distribution."""
        alpha_arr = jnp.asarray(alpha)
        beta_arr = jnp.asarray(beta)
        return (
            alpha_arr
            - jnp.log(beta_arr)
            + gammaln(alpha_arr)
            + (1 - alpha_arr) * digamma(alpha_arr)
        )

    @classmethod
    def moment(cls, n, alpha=1.0, beta=1.0, **kwargs):
        """n-th non-central moment of the gamma distribution."""
        # n-th moment: E[X^n] = Γ(α+n)/Γ(α) * β^(-n)
        n = jnp.asarray(n)
        alpha_arr = jnp.asarray(alpha)
        beta_arr = jnp.asarray(beta)
        return jnp.exp(gammaln(alpha_arr + n) - gammaln(alpha_arr)) / (beta_arr**n)

    @classmethod
    def skew(cls, alpha=1.0, beta=1.0, **kwargs):
        """Skewness of the gamma distribution."""
        return 2.0 / jnp.sqrt(alpha)

    @classmethod
    def kurtosis(cls, alpha=1.0, beta=1.0, **kwargs):
        """Excess kurtosis of the gamma distribution."""
        return 6.0 / alpha

    @classmethod
    def natural_parameters(cls, alpha=1.0, beta=1.0, **kwargs):
        """Natural parameters of the gamma distribution."""
        return jnp.array([alpha - 1, -beta])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the gamma distribution."""
        return jnp.array([jnp.log(x), x])

    @classmethod
    def log_partition(cls, alpha=1.0, beta=1.0, **kwargs):
        """Log partition function of the gamma distribution."""
        return gammaln(alpha) - alpha * jnp.log(beta)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of gamma distribution parameters.

        The MLE for the gamma distribution has a closed-form solution for beta:
        - beta = alpha / mean(data)

        For alpha, we need to solve the equation:
        log(alpha) - digamma(alpha) = log(mean(data)) - mean(log(data))
        """
        data = flatten_samples(data)
        dtype = data.dtype
        log_data = jnp.log(data)

        mean_data = weighted_mean(data, weights)
        mean_log_data = weighted_mean(log_data, weights)

        # Initial guess for alpha
        alpha = 0.5 / (jnp.log(mean_data) - mean_log_data)

        # Newton-Raphson iteration to find alpha
        def newton_step(alpha):
            f = jnp.log(alpha) - digamma(alpha) - (jnp.log(mean_data) - mean_log_data)
            f_prime = 1.0 / alpha - jax.grad(digamma)(alpha)
            return alpha - f / f_prime

        # Run a few iterations
        for _ in range(10):
            alpha = newton_step(alpha)

        # Compute beta
        beta = alpha / jnp.maximum(mean_data, jnp.asarray(1e-12, dtype=dtype))

        return alpha, beta


gamma = gamma_gen(name="gamma")
