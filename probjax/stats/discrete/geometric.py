"""
Geometric Distribution (:mod:`probjax.stats.geometric`)
====================================================

This module implements the Geometric distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import random
from jaxtyping import ArrayLike, PRNGKeyArray

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import unit_interval

__all__ = ["geometric"]


class geometric_gen(rv_discrete, rv_exponential_family):
    """A Geometric distribution."""

    parameters = {
        "p": unit_interval,
    }

    @classmethod
    def support(cls, p, **kwds):
        """Support of the Geometric distribution."""
        return (0, jnp.inf)

    @classmethod
    def pmf(cls, k: ArrayLike, p, **kwds):
        """Probability mass function of the Geometric distribution."""
        return jnp.exp(cls.logpmf(k, p, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, p, **kwds):
        """Log probability mass function of the Geometric distribution."""
        k = jnp.asarray(k)
        return k * jnp.log(1 - p) + jnp.log(p)

    @classmethod
    def rvs(cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), p=None, **kwds):
        """Random variates of the Geometric distribution."""
        p = jnp.asarray(p)
        event_shape = p.shape
        return random.geometric(rng, p, shape=shape + event_shape)

    @classmethod
    def mean(cls, p, **kwds):
        """Mean of the Geometric distribution."""
        return (1 - p) / p

    @classmethod
    def var(cls, p, **kwds):
        """Variance of the Geometric distribution."""
        return (1 - p) / (p * p)

    @classmethod
    def entropy(cls, p, **kwds):
        """Entropy of the Geometric distribution."""
        return -(1 - p) * jnp.log(1 - p) / p - jnp.log(p)

    @classmethod
    def mode(cls, p, **kwds):
        """Mode of the Geometric distribution."""
        return jnp.ones_like(p)

    @classmethod
    def natural_parameters(cls, p, **kwds):
        """Natural parameters of the Geometric distribution."""
        return jnp.log(1 - p)

    @classmethod
    def sufficient_statistics(cls, x, p, **kwds):
        """Sufficient statistics of the Geometric distribution."""
        return x

    @classmethod
    def log_partition(cls, p, **kwds):
        """Log partition function of the Geometric distribution."""
        return -jnp.log(p)

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of Geometric distribution parameters.

        The MLE for the Geometric distribution has a closed-form solution:
        p = 1 / (1 + mean(data))

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameter p
        """
        data = jnp.asarray(data)
        p = 1.0 / (1.0 + jnp.mean(data))
        return (p,)


geometric = geometric_gen(name="geometric")
