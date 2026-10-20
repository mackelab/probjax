"""
Geometric Distribution (:mod:`probjax.stats.geometric`)
====================================================

This module implements the Geometric distribution.
"""

from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import unit_interval
from probjax.stats.utils import weighted_mean
from probjax.utils.jaxutils import where
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
    def logpmf(cls, k: ArrayLike, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Log probability mass function of the Geometric distribution."""
        k = jnp.asarray(k)
        p = jnp.asarray(p)
        dtype = jnp.result_type(k, p)
        k = k.astype(dtype)
        is_integer = jnp.equal(k, jnp.floor(k))
        log1m_p = jnp.log1p(-p)
        valid = jnp.logical_and(k >= 0, is_integer)
        log_prob = jnp.log(p) + k * log1m_p
        log_prob = where(k == 0, jnp.log(p), log_prob)
        return where(valid, log_prob, -jnp.inf)

    @classmethod
    def _rvs_impl(
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
    def mean(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mean of the Geometric distribution."""
        return (1 - p) / p

    @classmethod
    def var(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Variance of the Geometric distribution."""
        return (1 - p) / (p * p)

    @classmethod
    def entropy(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Entropy of the Geometric distribution."""
        return -(1 - p) * jnp.log(1 - p) / p - jnp.log(p)

    @classmethod
    def mode(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mode of the Geometric distribution."""
        return jnp.zeros_like(p)

    @classmethod
    def natural_parameters(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
        """Natural parameters of the Geometric distribution."""
        return jnp.log(1 - p)

    @classmethod
    def sufficient_statistics(
        cls, x: ArrayLike, p: ArrayLike, **kwds: Any
    ) -> ArrayLike:
        """Sufficient statistics of the Geometric distribution."""
        return x

    @classmethod
    def log_partition(cls, p: ArrayLike, **kwds: Any) -> ArrayLike:
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
        """
        mean_data = weighted_mean(data, weights)
        p = 1.0 / (1.0 + mean_data)
        return (p,)


geometric = geometric_gen(name="geometric")
