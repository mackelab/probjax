"""
Beta Distribution (:mod:`probjax.stats.beta`)
==================================================

This module contains the Beta distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import digamma, gammaln
from jax.scipy.stats import beta as _beta

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import strict_positive, unit_interval
from probjax.stats.utils import clip_prob, flatten_samples, weighted_mean
from probjax.utils.special import betaincinv, betainccinv
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["beta"]


class beta_gen(rv_continuous, rv_exponential_family):
    """Beta continuous random variable.

    The beta distribution with concentration parameters `alpha` and `beta`.

    Parameters
    ----------
    alpha : float, optional
        Concentration parameter alpha. Default is 1.
    beta : float, optional
        Concentration parameter beta. Default is 1.
    """

    # Define parameter constraints
    parameters = {'alpha': strict_positive, 'beta': strict_positive}

    @classmethod
    def support(cls, alpha=1.0, beta=1.0, **kwargs):
        """Support of the beta distribution."""
        return unit_interval

    @classmethod
    def logpdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the probability density function of the beta distribution."""
        # Numerical stability clip values to avoid log(0)
        x = clip_prob(x)
        return _beta.logpdf(x, alpha, beta)

    @classmethod
    def cdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Cumulative distribution function of the beta distribution."""
        return _beta.cdf(x, alpha, beta)

    @classmethod
    def logcdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the cumulative distribution function of the beta distribution."""
        return jnp.log(_beta.cdf(x, alpha, beta))

    @classmethod
    def ppf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the beta distribution."""
        return betaincinv(alpha, beta, q)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        alpha=1.0,
        beta=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the beta distribution."""
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        event_shape = jnp.broadcast_shapes(alpha.shape, beta.shape)
        return random.beta(rng, alpha, beta, shape=shape + event_shape)

    @classmethod
    def sf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Survival function (1 - cdf) of the beta distribution."""
        return _beta.sf(x, alpha, beta)

    @classmethod
    def isf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the beta distribution."""
        q = jnp.asarray(q)
        return betainccinv(alpha, beta, q)

    @classmethod
    def mean(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mean of the beta distribution."""
        return alpha / (alpha + beta)

    @classmethod
    def mode(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mode of the beta distribution."""
        # Mode is (alpha-1)/(alpha+beta-2) for alpha,beta > 1
        # For alpha,beta <= 1, the mode is at the boundary
        valid = (alpha > 1) & (beta > 1)
        mode_interior = (alpha - 1) / (alpha + beta - 2)

        # When alpha < 1, beta > 1, mode is at 0
        mode_at_0 = (alpha < 1) & (beta >= 1)

        # When alpha > 1, beta < 1, mode is at 1
        mode_at_1 = (alpha >= 1) & (beta < 1)

        # When alpha < 1, beta < 1, the mode is at both 0 and 1
        # conventionally, we return the average
        mode_bimodal = (alpha < 1) & (beta < 1)

        return jnp.where(
            valid,
            mode_interior,
            jnp.where(
                mode_at_0,
                0.0,
                jnp.where(mode_at_1, 1.0, jnp.where(mode_bimodal, 0.5, 0.5)),
            ),  # 0.5 for uniform case
        )

    @classmethod
    def var(cls, alpha=1.0, beta=1.0, **kwargs):
        """Variance of the beta distribution."""
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    @classmethod
    def entropy(cls, alpha=1.0, beta=1.0, **kwargs):
        """Entropy of the beta distribution."""
        return (
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1) * digamma(alpha)
            + (beta - 1) * digamma(beta)
            - (alpha + beta - 2) * digamma(alpha + beta)
        )

    @classmethod
    def moment(cls, n, alpha=1.0, beta=1.0, **kwargs):
        """n-th non-central moment of the beta distribution."""
        n = jnp.asarray(n)
        return jnp.exp(
            gammaln(alpha + n)
            + gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(alpha + beta + n)
        )

    @classmethod
    def skew(cls, alpha=1.0, beta=1.0, **kwargs):
        """Skewness of the beta distribution."""
        return (
            2
            * (beta - alpha)
            * jnp.sqrt(alpha + beta + 1)
            / ((alpha + beta + 2) * jnp.sqrt(alpha * beta))
        )

    @classmethod
    def kurtosis(cls, alpha=1.0, beta=1.0, **kwargs):
        """Excess kurtosis of the beta distribution."""
        ab = alpha + beta
        num = 6 * ((alpha - beta) ** 2 * (ab + 1) - alpha * beta * (ab + 2))
        den = alpha * beta * (ab + 2) * (ab + 3)
        return num / den

    @classmethod
    def natural_parameters(cls, alpha=1.0, beta=1.0, **kwargs):
        """Natural parameters of the beta distribution."""
        return jnp.array([alpha - 1, beta - 1])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the beta distribution."""
        return jnp.array([jnp.log(x), jnp.log(1 - x)])

    @classmethod
    def log_partition(cls, alpha=1.0, beta=1.0, **kwargs):
        """Log partition function of the beta distribution."""
        return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of beta distribution parameters.

        The MLE for the beta distribution requires solving a system of equations:
        - digamma(alpha) - digamma(alpha + beta) = mean(log(x))
        - digamma(beta) - digamma(alpha + beta) = mean(log(1-x))

        We use Newton-Raphson iteration to solve this system.
        """
        data = flatten_samples(data)
        dtype = data.dtype
        log_data = jnp.log(data)
        log_1_minus_data = jnp.log(1 - data)

        mean_data = weighted_mean(data, weights)
        mean_log_data = weighted_mean(log_data, weights)
        mean_log_1_minus_data = weighted_mean(log_1_minus_data, weights)
        var_data = weighted_mean((data - mean_data) ** 2, weights)
        alpha = mean_data * (mean_data * (1 - mean_data) / var_data - 1)
        beta = (1 - mean_data) * (mean_data * (1 - mean_data) / var_data - 1)

        # Newton-Raphson iteration to find alpha and beta
        def newton_step(params):
            alpha, beta = params
            # Compute the functions
            f1 = digamma(alpha) - digamma(alpha + beta) - mean_log_data
            f2 = digamma(beta) - digamma(alpha + beta) - mean_log_1_minus_data

            # Compute the Jacobian
            trigamma_alpha = jax.grad(digamma)(alpha)
            trigamma_beta = jax.grad(digamma)(beta)
            trigamma_sum = jax.grad(digamma)(alpha + beta)

            J11 = trigamma_alpha - trigamma_sum
            J12 = -trigamma_sum
            J21 = -trigamma_sum
            J22 = trigamma_beta - trigamma_sum

            # Compute the inverse of the Jacobian
            det = J11 * J22 - J12 * J21
            J_inv = jnp.array([[J22, -J12], [-J21, J11]]) / det

            # Update parameters
            delta = J_inv @ jnp.array([f1, f2])
            return jnp.array([alpha - delta[0], beta - delta[1]])

        # Run a few iterations
        params = jnp.array([alpha, beta], dtype=dtype)
        for _ in range(10):
            params = newton_step(params)

        return tuple(params)


beta = beta_gen(name="beta")
