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
from probjax.utils.special import betaincinv
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
    def pdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Probability density function of the beta distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, alpha, beta, **kwargs))

    @classmethod
    def logpdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the probability density function of the beta distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        # Numerical stability clip values to avoid log(0)
        x = jnp.clip(x, jnp.finfo(jnp.float32).eps, 1.0 - jnp.finfo(jnp.float32).eps)
        return _beta.logpdf(x, alpha, beta)

    @classmethod
    def cdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Cumulative distribution function of the beta distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        return _beta.cdf(x, alpha, beta)

    @classmethod
    def logcdf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Log of the cumulative distribution function of the beta distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        return jnp.log(_beta.cdf(x, alpha, beta))

    @classmethod
    def ppf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the beta distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        return betaincinv(alpha, beta, q)

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        alpha=1.0,
        beta=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the beta distribution.

        Parameters
        ----------
        rng : RngKey
            JAX PRNG key for random number generation
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.
        shape : tuple of ints, optional
            Output shape. Default is (), meaning a single value.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        event_shape = jnp.broadcast_shapes(alpha.shape, beta.shape)
        return random.beta(rng, alpha, beta, shape=shape + event_shape)

    @classmethod
    def sf(cls, x, alpha=1.0, beta=1.0, **kwargs):
        """Survival function (1 - cdf) of the beta distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        return _beta.sf(x, alpha, beta)

    @classmethod
    def isf(cls, q, alpha=1.0, beta=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the beta distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        return _beta.isf(q, alpha, beta)

    @classmethod
    def mean(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mean of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        return alpha / (alpha + beta)

    @classmethod
    def mode(cls, alpha=1.0, beta=1.0, **kwargs):
        """Mode of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
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

        # When alpha = beta = 1, the beta is uniform, mode is arbitrary
        mode_uniform = (alpha == 1) & (beta == 1)

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
        """Variance of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    @classmethod
    def entropy(cls, alpha=1.0, beta=1.0, **kwargs):
        """Entropy of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
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
        """n-th non-central moment of the beta distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        n = jnp.asarray(n)
        return jnp.exp(
            gammaln(alpha + n)
            + gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(alpha + beta + n)
        )

    @classmethod
    def skew(cls, alpha=1.0, beta=1.0, **kwargs):
        """Skewness of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return (
            2
            * (beta - alpha)
            * jnp.sqrt(alpha + beta + 1)
            / ((alpha + beta + 2) * jnp.sqrt(alpha * beta))
        )

    @classmethod
    def kurtosis(cls, alpha=1.0, beta=1.0, **kwargs):
        """Excess kurtosis of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        ab = alpha + beta
        num = 6 * ((alpha - beta) ** 2 * (ab + 1) - alpha * beta * (ab + 2))
        den = alpha * beta * (ab + 2) * (ab + 3)
        return num / den

    @classmethod
    def natural_parameters(cls, alpha=1.0, beta=1.0, **kwargs):
        """Natural parameters of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        natural_parameters : tuple
            Natural parameters of the distribution
        """
        return jnp.array([alpha - 1, beta - 1])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the beta distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : tuple
            Sufficient statistics of the distribution
        """
        return jnp.array([jnp.log(x), jnp.log(1 - x)])

    @classmethod
    def log_partition(cls, alpha=1.0, beta=1.0, **kwargs):
        """Log partition function of the beta distribution.

        Parameters
        ----------
        alpha : float, optional
            Concentration parameter alpha. Default is 1.
        beta : float, optional
            Concentration parameter beta. Default is 1.

        Returns
        -------
        log_partition : float
            Log partition function of the distribution
        """
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

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (alpha, beta)
        """
        data = jnp.asarray(data)
        data = jnp.reshape(data, (-1,))
        dtype = data.dtype
        log_data = jnp.log(data)
        log_1_minus_data = jnp.log(1 - data)

        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((-1,))
            if weights.shape[0] != data.shape[0]:
                raise ValueError("weights must have the same length as data")
            weights = jnp.clip(weights, 0)
            total = jnp.sum(weights)
            total = jnp.where(total > 0, total, jnp.asarray(data.shape[0], dtype=dtype))
            weights = weights / total
            mean_data = jnp.sum(weights * data)
            mean_log_data = jnp.sum(weights * log_data)
            mean_log_1_minus_data = jnp.sum(weights * log_1_minus_data)
            var_data = jnp.sum(weights * (data - mean_data) ** 2)
        else:
            mean_log_data = jnp.mean(log_data)
            mean_log_1_minus_data = jnp.mean(log_1_minus_data)
            mean_data = jnp.mean(data)
            var_data = jnp.var(data)
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
