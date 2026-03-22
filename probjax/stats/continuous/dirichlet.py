"""
Dirichlet Distribution (:mod:`probjax.stats.dirichlet`)
==================================================

This module contains the Dirichlet distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import digamma, gammaln

from probjax.stats.base import rv_exponential_family, rv_multivariate
from probjax.stats.constraints import (
    positive,
    simplex,
    symmetric_positive_definite_matrix,
)
from probjax.stats.utils import normalize_sample_weights, row_mean_and_var
from probjax.utils.stats import mle_dirichlet
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["dirichlet"]


class dirichlet_gen(rv_multivariate, rv_exponential_family):
    """Dirichlet continuous random variable.

    The Dirichlet distribution parameterized by concentration parameters `alpha`.

    Parameters
    ----------
    alpha : array_like
        Concentration parameters. Must be positive.
    """

    name = "dirichlet"
    parameters = {"alpha": positive}
    multivariate = True

    @classmethod
    def support(cls, alpha=None, **kwargs):
        """Support of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like, optional
            Concentration parameters. Default is None.

        Returns
        -------
        support : constraint
            Support of the distribution
        """
        return positive

    @classmethod
    def pdf(cls, x: Array, alpha: Array, **kwargs):
        """Probability density function of the Dirichlet distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function
        alpha : array_like
            Concentration parameters

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, alpha, **kwargs))

    @classmethod
    def _multivariate_batch_event_shape(cls, alpha: Array, **kwargs):
        alpha_arr = jnp.asarray(alpha)
        if alpha_arr.ndim < 1:
            raise ValueError("alpha must be at least one-dimensional.")
        batch_shape = tuple(int(dim) for dim in alpha_arr.shape[:-1])
        event_shape = (int(alpha_arr.shape[-1]),)
        return batch_shape, event_shape

    @classmethod
    def logpdf(cls, x: Array, alpha: Array, **kwargs):
        """Log of the probability density function of the Dirichlet distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log probability density function
        alpha : array_like
            Concentration parameters

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        alpha_arr, x_arr = jnp.broadcast_arrays(alpha, x)
        x_arr = jnp.asarray(x_arr)
        if x_arr.ndim == 1:
            return jax.scipy.stats.dirichlet.logpdf(x_arr, alpha_arr)
        log_fn = jax.scipy.stats.dirichlet.logpdf
        for _ in range(x_arr.ndim - 1):
            log_fn = jax.vmap(log_fn)
        return log_fn(x_arr, alpha_arr)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        alpha: Array,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Dirichlet distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        alpha : array_like
            Concentration parameters
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().

        Returns
        -------
        rvs : ndarray
            Random variates of given shape
        """
        assert jnp.ndim(alpha) >= 1, "alpha must be at least one-dimensional."
        assert jnp.all(alpha > 0), "alpha must be positive."

        if alpha.ndim > 1:
            batch_shape = jnp.shape(alpha)[:-1]
            event_shape = jnp.shape(alpha)[-1:]
        else:
            batch_shape = ()
            event_shape = jnp.shape(alpha)

        shape = shape + batch_shape
        return random.dirichlet(rng, alpha, shape)

    @classmethod
    def mean(cls, alpha: Array, **kwargs):
        """Mean of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        mean : ndarray
            Mean of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return alpha / alpha_sum

    @classmethod
    def mode(cls, alpha: Array, **kwargs):
        """Mode of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        mode : ndarray
            Mode of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        valid = alpha > 1
        mode = jnp.where(valid, (alpha - 1) / (alpha_sum - alpha.shape[-1]), 1e-20)
        # Mode must sum to 1
        return mode / jnp.sum(mode, axis=-1, keepdims=True)

    @classmethod
    def var(cls, alpha: Array, **kwargs):
        """Variance of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        var : ndarray
            Variance of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return alpha * (alpha_sum - alpha) / (alpha_sum**2 * (alpha_sum + 1))

    @classmethod
    def entropy(cls, alpha: Array, **kwargs):
        """Entropy of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        entropy : ndarray
            Entropy of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return (
            gammaln(alpha_sum)
            - gammaln(alpha)
            + (alpha - 1) * digamma(alpha)
            - (alpha_sum - alpha.shape[-1]) * digamma(alpha_sum)
        )

    @classmethod
    def skew(cls, alpha: Array, **kwargs):
        """Skewness of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        skew : ndarray
            Skewness of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return (
            2
            * (alpha_sum - alpha)
            * jnp.sqrt(alpha_sum + 1)
            / ((alpha_sum + 2) * jnp.sqrt(alpha * (alpha_sum - alpha)))
        )

    @classmethod
    def kurtosis(cls, alpha: Array, **kwargs):
        """Excess kurtosis of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        kurtosis : ndarray
            Excess kurtosis of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return 6 * (
            (alpha_sum**2 * (alpha_sum + 1))
            / (alpha * (alpha_sum - alpha) * (alpha_sum + 2) * (alpha_sum + 3))
            - 1
        )

    @classmethod
    def natural_parameters(cls, alpha: Array, **kwargs):
        """Natural parameters of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        natural_parameters : ndarray
            Natural parameters of the distribution
        """
        return alpha - 1

    @classmethod
    def sufficient_statistics(cls, x: Array, **kwargs):
        """Sufficient statistics of the Dirichlet distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : ndarray
            Sufficient statistics of the distribution
        """
        return jnp.log(x)

    @classmethod
    def log_partition(cls, alpha: Array, **kwargs):
        """Log partition function of the Dirichlet distribution.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters

        Returns
        -------
        log_partition : ndarray
            Log partition function of the distribution
        """
        alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
        return gammaln(alpha_sum) - jnp.sum(gammaln(alpha), axis=-1)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Maximum likelihood estimation of Dirichlet concentration parameters.

        Uses method of moments for the initial estimate, then refines via
        fixed-point MLE iteration (Minka 2000).
        """
        data = jnp.asarray(data)
        if data.ndim == 1:
            raise ValueError("Dirichlet fitting expects observations arranged by rows.")
        dtype = data.dtype
        eps = jnp.asarray(1e-6, dtype=dtype)

        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=dtype,
            mismatch_message="weights must have the same number of rows as data",
            column=True,
        )
        mean, var = row_mean_and_var(data, weights_arr)

        mean = jnp.clip(mean, eps, 1 - eps)
        var = jnp.maximum(var, eps)
        alpha0 = jnp.mean(mean * (1 - mean) / var - 1.0)
        alpha0 = jnp.maximum(alpha0, eps)
        alpha_init = jnp.clip(mean * alpha0, eps, None)

        alpha = mle_dirichlet(data, alpha0=alpha_init)
        alpha = jnp.maximum(alpha, eps)
        return (alpha,)


dirichlet = dirichlet_gen(name="dirichlet")
