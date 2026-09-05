"""
Bernoulli Distribution (:mod:`probjax.stats.bernoulli`)
====================================================

This module implements the Bernoulli distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import unit_interval
from probjax.stats.utils import flatten_samples, normalize_sample_weights
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["bernoulli"]


class bernoulli_gen(rv_discrete, rv_exponential_family):
    """A Bernoulli distribution."""

    parameters = {
        "p": unit_interval,
    }

    @classmethod
    def support(cls, p, **kwds):
        """Support of the Bernoulli distribution."""
        return (0, 1)

    @classmethod
    def pmf(cls, x: ArrayLike, p, **kwds):
        """Probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, p, 1 - p)

    @classmethod
    def logpmf(cls, x: ArrayLike, p, **kwds):
        """Log probability mass function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x == 1, jnp.log(p), jnp.log(1 - p))

    @classmethod
    def cdf(cls, x: ArrayLike, p, **kwds):
        """Cumulative distribution function of the Bernoulli distribution."""
        x = jnp.asarray(x)
        return jnp.where(x < 0, 0, jnp.where(x < 1, 1 - p, 1))

    @classmethod
    def ppf(cls, q: ArrayLike, p, **kwds):
        """Percent point function of the Bernoulli distribution.

        Parameters
        ----------
        q : array_like
            Lower tail probability
        p : float
            Probability of success

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        q = jnp.asarray(q)
        p = jnp.asarray(p)
        # For q < 1-p, return 0
        # For q >= 1-p, return 1
        # Handle edge cases where q is exactly 1-p
        return jnp.where(q <= 1 - p, 0, 1)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        p=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Bernoulli distribution."""
        p = jnp.asarray(p)
        event_shape = p.shape
        return random.bernoulli(rng, p, shape=shape + event_shape)

    @classmethod
    def mean(cls, p, **kwds):
        """Mean of the Bernoulli distribution."""
        return p

    @classmethod
    def var(cls, p, **kwds):
        """Variance of the Bernoulli distribution."""
        return p * (1 - p)

    @classmethod
    def entropy(cls, p, **kwds):
        """Entropy of the Bernoulli distribution."""
        return -p * jnp.log(p) - (1 - p) * jnp.log(1 - p)

    @classmethod
    def mode(cls, p, **kwds):
        """Mode of the Bernoulli distribution."""
        return jnp.where(p > 0.5, 1, 0)

    @classmethod
    def natural_parameters(cls, p, **kwds):
        """Natural parameters of the Bernoulli distribution."""
        return jnp.array([jnp.log(p / (1 - p))])

    @classmethod
    def sufficient_statistics(cls, x, **kwds):
        """Sufficient statistics of the Bernoulli distribution."""
        return jnp.array([x])

    @classmethod
    def log_partition(cls, p, **kwds):
        """Log partition function of the Bernoulli distribution."""
        return -jnp.log(1 - p)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Bernoulli distribution parameter.

        The MLE for the Bernoulli distribution is simply the sample mean:
        p = mean(data)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameter (p,)
        """
        data = flatten_samples(data)
        dtype = data.dtype
        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=dtype,
        )
        if weights_arr is not None:
            p = jnp.sum(weights_arr * data)
        else:
            p = jnp.mean(data)
        return (p,)


bernoulli = bernoulli_gen(name="bernoulli")
