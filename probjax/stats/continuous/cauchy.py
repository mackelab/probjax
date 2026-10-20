"""
Cauchy Distribution (:mod:`probjax.stats.cauchy`)
==================================================

This module contains the Cauchy distribution.
"""

from typing import Any, Optional, Tuple

import jax.numpy as jnp
import jax.scipy.stats.cauchy as _cauchy
from jax import random
from jax.scipy.optimize import minimize as jax_minimize

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, strict_positive
from probjax.stats.utils import _require_unweighted_fit, loc_scale_sample
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["cauchy"]


class cauchy_gen(rv_continuous):
    """Cauchy continuous random variable.

    The Cauchy distribution with location `loc` and scale `scale`.

    Parameters
    ----------
    loc : float, optional
        Location parameter of the distribution. Default is 0.
    scale : float, optional
        Scale parameter of the distribution. Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive}

    @classmethod
    def pdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Probability density function of the Cauchy distribution."""
        return _cauchy.pdf(x, loc, scale)

    @classmethod
    def logpdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Log of the probability density function of the Cauchy distribution."""
        return _cauchy.logpdf(x, loc, scale)

    @classmethod
    def cdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Cumulative distribution function of the Cauchy distribution."""
        return _cauchy.cdf(x, loc, scale)

    @classmethod
    def ppf(
        cls,
        q: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Percent point function (inverse of cdf) of the Cauchy distribution."""
        return _cauchy.ppf(q, loc, scale)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the Cauchy distribution."""
        return loc_scale_sample(rng, random.cauchy, shape=shape, loc=loc, scale=scale)

    @classmethod
    def sf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Survival function (1 - cdf) of the Cauchy distribution."""
        return _cauchy.sf(x, loc, scale)

    @classmethod
    def isf(
        cls,
        q: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Inverse survival function (inverse of sf) of the Cauchy distribution."""
        return _cauchy.isf(q, loc, scale)

    @classmethod
    def logcdf(
        cls,
        x: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Log of the cumulative distribution function of the Cauchy distribution."""
        return _cauchy.logcdf(x, loc, scale)

    @classmethod
    def mean(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Mean of the Cauchy distribution."""
        return jnp.full_like(loc, jnp.nan)

    @classmethod
    def mode(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Mode of the Cauchy distribution."""
        return jnp.asarray(loc)

    @classmethod
    def median(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Median of the Cauchy distribution."""
        return jnp.asarray(loc)

    @classmethod
    def var(
        cls, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0, **kwargs: Any
    ) -> ArrayLike:
        """Variance of the Cauchy distribution."""
        return jnp.full_like(loc, jnp.inf)

    @classmethod
    def entropy(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Entropy of the Cauchy distribution."""
        return jnp.log(4 * jnp.pi * scale)

    @classmethod
    def moment(
        cls,
        n: int,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """n-th non-central moment of the Cauchy distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        return jnp.full_like(loc, jnp.inf)

    @classmethod
    def skew(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Skewness of the Cauchy distribution."""
        return jnp.full_like(loc, jnp.nan)

    @classmethod
    def kurtosis(
        cls,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        **kwargs: Any,
    ) -> ArrayLike:
        """Excess kurtosis of the Cauchy distribution."""
        return jnp.full_like(loc, jnp.nan)

    @classmethod
    def fit(
        cls,
        data,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Maximum likelihood estimation of Cauchy distribution parameters.

        Uses the sample median and MAD for the initial estimate, then refines
        via BFGS optimization of the negative log-likelihood.
        """
        _require_unweighted_fit(weights, "Cauchy")
        data = jnp.asarray(data)
        loc_init = jnp.median(data, axis=0)
        scale_init = jnp.median(jnp.abs(data - loc_init), axis=0)
        scale_init = jnp.maximum(scale_init, jnp.asarray(1e-6, dtype=data.dtype))

        # Optimize log-likelihood via BFGS in unconstrained space (log scale)
        init_flat = jnp.concatenate([
            jnp.atleast_1d(loc_init),
            jnp.log(jnp.atleast_1d(scale_init)),
        ])

        def neg_log_lik(params_flat):
            loc = params_flat[0]
            scale = jnp.exp(params_flat[1])
            return -jnp.sum(cls.logpdf(data, loc=loc, scale=scale))

        result = jax_minimize(neg_log_lik, init_flat, method="BFGS")
        if not result.success or not bool(jnp.all(jnp.isfinite(result.x))):
            return loc_init, scale_init
        fitted = result.x
        loc = fitted[0]
        scale = jnp.exp(fitted[1])
        return loc, scale


cauchy = cauchy_gen(name="cauchy")
