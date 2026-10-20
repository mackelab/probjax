"""
Dirac Distribution (:mod:`probjax.stats.dirac`)
=============================================

This module implements the Dirac distribution.
"""

from typing import Any, Tuple

import jax.numpy as jnp

from probjax.stats.base import rv_discrete, rv_exponential_family
from probjax.stats.constraints import real
from probjax.utils.jaxutils import where
from probjax.utils.typing import ArrayLike, RngKey

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
    def logpmf(cls, k: ArrayLike, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Log probability mass function of the Dirac distribution."""
        return where(k == loc, 0.0, -jnp.inf)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Dirac distribution."""
        loc = jnp.asarray(loc)
        event_shape = loc.shape
        return jnp.broadcast_to(loc, shape + event_shape)

    @classmethod
    def mean(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mean of the Dirac distribution."""
        return loc

    @classmethod
    def var(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Variance of the Dirac distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def entropy(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Entropy of the Dirac distribution."""
        return jnp.zeros_like(loc)

    @classmethod
    def mode(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Mode of the Dirac distribution."""
        return loc

    @classmethod
    def natural_parameters(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Natural parameters of the Dirac distribution."""
        return loc

    @classmethod
    def sufficient_statistics(
        cls, x: ArrayLike, loc: ArrayLike, **kwds: Any
    ) -> ArrayLike:
        """Sufficient statistics of the Dirac distribution."""
        return x

    @classmethod
    def log_partition(cls, loc: ArrayLike, **kwds: Any) -> ArrayLike:
        """Log partition function of the Dirac distribution."""
        return 0.0


dirac = dirac_gen(name="dirac")
