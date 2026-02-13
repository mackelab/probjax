"""
Poisson Distribution (:mod:`probjax.stats.poisson`)
================================================

This module implements the Poisson distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import poisson as jax_poisson

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import positive_integer
from probjax.stats.utils import flatten_samples, normalize_sample_weights
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["poisson"]


class poisson_gen(rv_discrete, rv_exponential_family):
    """A Poisson distribution."""

    parameters = {
        "rate": positive_integer,
    }

    @classmethod
    def support(cls, rate, **kwds):
        """Support of the Poisson distribution."""
        return (0, jnp.inf)

    @classmethod
    def pmf(cls, k: ArrayLike, rate, **kwds):
        """Probability mass function of the Poisson distribution."""
        return jnp.exp(cls.logpmf(k, rate, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, rate, **kwds):
        """Log probability mass function of the Poisson distribution."""
        return jax_poisson.logpmf(k, rate)

    @classmethod
    def cdf(cls, k: ArrayLike, rate, **kwds):
        """Cumulative distribution function of the Poisson distribution."""
        return jax_poisson.cdf(k, rate)

    @classmethod
    def ppf(cls, q: ArrayLike, rate, **kwds):
        """Percent point function of the Poisson distribution.

        Parameters
        ----------
        q : array_like
            Quantile, must be between 0 and 1
        rate : float or array_like
            Rate parameter of the Poisson distribution

        Returns
        -------
        k : ndarray
            The smallest integer k such that CDF(k) ≥ q
        """
        q = jnp.asarray(q)
        rate = jnp.asarray(rate)

        # Ensure q is between 0 and 1
        q = jnp.clip(q, 0, 1)

        # Define the single-element version of the function
        def ppf_single(q_single, rate_single):
            def body_fun(state):
                k, found = state
                cdf_val = cls.cdf(k, rate_single)
                found = found | (cdf_val >= q_single)
                return (k + 1, found)

            def cond_fun(state):
                k, found = state
                return ~found

            # Initialize with k=0 and found=False
            init_state = (0, False)

            # Use jax.lax.while_loop to find the smallest k
            final_k, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)

            return final_k - 1  # Subtract 1 because we incremented one too many times

        # Vectorize the function over the inputs
        q, rate = jnp.broadcast_arrays(q, rate)
        for i in range(q.ndim):
            ppf_single = jax.vmap(ppf_single)
        return ppf_single(q, rate)

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        rate=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Poisson distribution."""
        rate = jnp.asarray(rate)
        event_shape = rate.shape
        return random.poisson(rng, rate, shape=shape + event_shape)

    @classmethod
    def mean(cls, rate, **kwds):
        """Mean of the Poisson distribution."""
        return rate

    @classmethod
    def var(cls, rate, **kwds):
        """Variance of the Poisson distribution."""
        return rate

    @classmethod
    def mode(cls, rate, **kwds):
        """Mode of the Poisson distribution.

        For a Poisson distribution with rate λ:
        - If λ is an integer, both λ and λ-1 are modes
        - If λ is not an integer, the mode is floor(λ)

        Parameters
        ----------
        rate : float or array_like
            Rate parameter of the Poisson distribution

        Returns
        -------
        mode : ndarray
            Mode of the Poisson distribution
        """
        rate = jnp.asarray(rate)
        # For non-integer rates, mode is floor(rate)
        # For integer rates, both rate and rate-1 are modes, but we return rate
        # as it's the larger of the two values
        return jnp.floor(rate)

    @classmethod
    def entropy(cls, rate, **kwds):
        """Entropy of the Poisson distribution."""
        return rate * (1 - jnp.log(rate))

    @classmethod
    def natural_parameters(cls, rate, **kwds):
        """Natural parameters of the Poisson distribution."""
        return jnp.array([jnp.log(rate)])

    @classmethod
    def sufficient_statistics(cls, x, **kwds):
        """Sufficient statistics of the Poisson distribution."""
        return jnp.array([x])

    @classmethod
    def log_partition(cls, rate, **kwds):
        """Log partition function of the Poisson distribution."""
        return rate

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Poisson distribution parameter.

        The MLE for the Poisson distribution is simply the sample mean:
        rate = mean(data)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameter (rate,)
        """
        data = flatten_samples(data)
        dtype = data.dtype
        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=dtype,
        )
        if weights_arr is not None:
            rate = jnp.sum(weights_arr * data)
        else:
            rate = jnp.mean(data)
        return (rate,)


poisson = poisson_gen(name="poisson")
