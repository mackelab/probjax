"""
Geometric Distribution (:mod:`probjax.stats.geometric`)
====================================================

This module implements the Geometric distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import unit_interval
from probjax.utils.typing import ArrayLike, RngKey

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
        dtype = jnp.result_type(k, p)
        k = k.astype(dtype)
        is_integer = jnp.equal(k, jnp.floor(k))
        log1m_p = jnp.log1p(-p)
        valid = jnp.logical_and(k >= 0, is_integer)
        log_prob = jnp.log(p) + k * log1m_p
        log_prob = jnp.where(k == 0, jnp.log(p), log_prob)
        return jnp.where(valid, log_prob, -jnp.inf)

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        p=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Geometric distribution."""
        p = jnp.asarray(p)
        event_shape = p.shape
        samples = random.geometric(rng, p, shape=shape + event_shape)
        samples = jnp.asarray(samples, dtype=jnp.int32)
        return samples - jnp.int32(1)

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
        return jnp.zeros_like(p)

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
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
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
        data = jnp.reshape(data, (-1,))
        dtype = data.dtype
        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((-1,))
            if weights.shape[0] != data.shape[0]:
                raise ValueError("weights must have the same length as data")
            weights = jnp.clip(weights, 0)
            total = jnp.sum(weights)
            total = jnp.where(total > 0, total, jnp.asarray(data.shape[0], dtype=dtype))
            weights = weights / total
            mean_data = jnp.sum(weights * data)
        else:
            mean_data = jnp.mean(data)
        p = 1.0 / (1.0 + mean_data)
        return (p,)


geometric = geometric_gen(name="geometric")
