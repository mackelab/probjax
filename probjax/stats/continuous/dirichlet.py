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
from jaxtyping import Array, PRNGKeyArray

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import positive

__all__ = ["dirichlet"]


class dirichlet_gen(rv_continuous, rv_exponential_family):
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
        alpha, x = jnp.broadcast_arrays(alpha, x)
        log_prob_fn = jax.scipy.stats.dirichlet.logpdf
        for _ in range(x.ndim - 1):
            log_prob_fn = jax.vmap(log_prob_fn)
        return log_prob_fn(x, alpha)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        alpha: Array = None,
        **kwargs,
    ):
        """Random variates of the Dirichlet distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().
        alpha : array_like
            Concentration parameters

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
        return jnp.where(valid, (alpha - 1) / (alpha_sum - alpha.shape[-1]), 0)

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

    def freeze(self, alpha: Array, **kwargs):
        """Freeze the distribution by fixing the parameters.

        Parameters
        ----------
        alpha : array_like
            Concentration parameters
        """
        rv = super().freeze(alpha=alpha, **kwargs)
        rv._batch_shape = alpha.shape[:-1]
        rv._event_shape = alpha.shape[-1:]
        return rv

dirichlet = dirichlet_gen(name="dirichlet")
