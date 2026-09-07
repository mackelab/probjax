"""
Binomial Distribution (:mod:`probjax.stats.binomial`)
==================================================

This module implements the Binomial distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import binom

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import strict_positive_integer, unit_interval
from probjax.stats.discrete._ppf_search import ppf_by_cdf_search
from probjax.stats.utils import weighted_mean
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["binomial"]


class binomial_gen(rv_discrete, rv_exponential_family):
    """A Binomial distribution."""

    parameters = {
        "n": strict_positive_integer,
        "probs": unit_interval,
    }

    @classmethod
    def support(cls, n, probs, **kwds):
        """Support of the Binomial distribution."""
        return (0, n)

    @classmethod
    def logpmf(cls, k: ArrayLike, n, probs, **kwds):
        """Log probability mass function of the Binomial distribution."""
        return binom.logpmf(k, n, probs)

    @classmethod
    def cdf(cls, k: ArrayLike, n, probs, **kwds):
        """Cumulative distribution function of the Binomial distribution.

        The CDF is computed using the regularized incomplete beta function:
        F(k; n, p) = I_{1-p}(n-k, k+1)

        Parameters
        ----------
        k : array_like
            Quantiles
        n : int or array_like
            Number of trials
        probs : float or array_like
            Probability of success

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at k
        """
        k = jnp.asarray(k)
        n = jnp.asarray(n)
        probs = jnp.asarray(probs)

        # Handle edge cases
        cdf = jnp.where(
            k < 0,
            0,
            jnp.where(
                k >= n,
                1,
                # For 0 <= k < n, use the regularized incomplete beta function
                jax.scipy.special.betainc(n - k, k + 1, 1 - probs),
            ),
        )
        return cdf

    @classmethod
    def ppf(cls, q: ArrayLike, n, probs, **kwds):
        """Percent point function of the Binomial distribution.

        Parameters
        ----------
        q : array_like
            Quantiles
        n : int or array_like
            Number of trials
        probs : float or array_like
            Probability of success

        Returns
        -------
        ppf : ndarray
            Percent point function evaluated at q
        """
        q = jnp.asarray(q)
        n = jnp.asarray(n, dtype=jnp.int32)  # Ensure n is an integer
        probs = jnp.asarray(probs)

        return ppf_by_cdf_search(q, cls.cdf, n, probs, hi=n)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        n=None,
        probs=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Binomial distribution."""
        n = jnp.asarray(n)
        probs = jnp.asarray(probs)
        event_shape = jnp.broadcast_shapes(n.shape, probs.shape)
        return random.binomial(rng, n, probs, shape=shape + event_shape)

    @classmethod
    def mean(cls, n, probs, **kwds):
        """Mean of the Binomial distribution."""
        return n * probs

    @classmethod
    def median(cls, n, probs, **kwds):
        """Median of the Binomial distribution."""
        return jnp.floor(n * probs)

    @classmethod
    def mode(cls, n, probs, **kwds):
        """Mode of the Binomial distribution."""
        return jnp.floor((n + 1) * probs)

    @classmethod
    def var(cls, n, probs, **kwds):
        """Variance of the Binomial distribution."""
        return n * probs * (1 - probs)

    @classmethod
    def entropy(cls, n, probs, **kwds):
        """Entropy of the Binomial distribution."""
        return jnp.log(2) - probs * jnp.log(probs) - (1 - probs) * jnp.log(1 - probs)

    @classmethod
    def natural_parameters(cls, n, probs, **kwds):
        """Natural parameters of the Binomial distribution."""
        return jnp.array([jnp.log(probs / (1 - probs))])

    @classmethod
    def sufficient_statistics(cls, x, **kwds):
        """Sufficient statistics of the Binomial distribution."""
        return jnp.array([x])

    @classmethod
    def log_partition(cls, n, probs, **kwds):
        """Log partition function of the Binomial distribution."""
        return n * jnp.log(1 + jnp.exp(jnp.log(probs / (1 - probs))))

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        n: int,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Binomial distribution parameter.

        The MLE for the Binomial distribution is:
        p = mean(data) / n

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        n : int
            Number of trials (must be provided)
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (n, p)
        """
        mean_successes = weighted_mean(data, weights)
        p = mean_successes / n
        return (n, p)


binomial = binomial_gen(name="binomial")
