"""
Binomial Distribution (:mod:`probjax.stats.binomial`)
==================================================

This module implements the Binomial distribution.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import binom
from jaxtyping import ArrayLike, PRNGKeyArray

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import strict_positive_integer, unit_interval

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
    def pmf(cls, k: ArrayLike, n, probs, **kwds):
        """Probability mass function of the Binomial distribution."""
        return jnp.exp(cls.logpmf(k, n, probs, **kwds))

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

        # Ensure q is between 0 and 1
        q = jnp.clip(q, 0, 1)

        # Define the single-element version of the function
        def ppf_single(q_single, n_single, probs_single):
            def body_fun(state):
                k, found = state
                cdf_val = cls.cdf(k, n_single, probs_single)
                found = found | (cdf_val >= q_single)
                return (k + 1, found)

            def cond_fun(state):
                k, found = state
                # Also check k <= n_single to prevent infinite loop if q is close to 1
                # and cdf never quite reaches q due to floating point inaccuracies.
                return ~found & (k <= n_single)

            # Initialize with k=0 and found=False
            init_state = (0, False)

            # Use jax.lax.while_loop to find the smallest k
            final_k, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)

            # final_k could be n_single + 1 if q is 1 or very close to 1.
            # If final_k is n_single + 1, it means the loop terminated because k > n_single,
            # and the actual ppf should be n_single.
            # Otherwise, it's final_k - 1 because we incremented one too many times.
            return jnp.where(final_k > n_single, n_single, final_k - 1)

        # Vectorize the function over the inputs
        # Ensure all inputs are broadcastable
        q_b, n_b, probs_b = jnp.broadcast_arrays(q, n, probs)
        for _ in range(q_b.ndim):
            ppf_single = jax.vmap(ppf_single)

        return ppf_single(q_b, n_b, probs_b)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
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
    def fit(cls, data: ArrayLike, n: int, **kwds):
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
        data = jnp.asarray(data)
        p = jnp.mean(data) / n
        return (n, p)


binomial = binomial_gen(name="binomial")
