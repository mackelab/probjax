"""
Dirac Distribution (:mod:`probjax.stats.dirac`)
=============================================

This module implements the Dirac distribution.
"""

from typing import Any, Dict, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import real

__all__ = ["dirac"]


class dirac_gen(rv_discrete, rv_exponential_family):
    """A Dirac distribution."""

    parameters = {
        "loc": real,
    }

    @classmethod
    def support(cls, loc, **kwds):
        """Support of the Dirac distribution."""
        return (loc, loc)

    @classmethod
    def pmf(cls, k: ArrayLike, loc, **kwds):
        """Probability mass function of the Dirac distribution."""
        return jnp.exp(cls.logpmf(k, loc, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, loc, **kwds):
        """Log probability mass function of the Dirac distribution."""
        return jnp.where(k == loc, 0.0, -jnp.inf)

    @classmethod
    def rvs(cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), loc=None, **kwds):
        """Random variates of the Dirac distribution."""
        loc = jnp.asarray(loc)
        event_shape = loc.shape
        return jnp.broadcast_to(loc, shape + event_shape)

    @classmethod
    def mean(cls, loc, **kwds):
        """Mean of the Dirac distribution."""
        return loc

    @classmethod
    def var(cls, loc, **kwds):
        """Variance of the Dirac distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def entropy(cls, loc, **kwds):
        """Entropy of the Dirac distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def mode(cls, loc, **kwds):
        """Mode of the Dirac distribution."""
        return loc

    @classmethod
    def natural_parameters(cls, loc, **kwds):
        """Natural parameters of the Dirac distribution."""
        return loc

    @classmethod
    def sufficient_statistics(cls, x, loc, **kwds):
        """Sufficient statistics of the Dirac distribution."""
        return x

    @classmethod
    def log_partition(cls, loc, **kwds):
        """Log partition function of the Dirac distribution."""
        return 0.0


dirac = dirac_gen(name="dirac")
