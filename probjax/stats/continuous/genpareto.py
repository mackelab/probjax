"""
Generalized Pareto Distribution (:mod:`probjax.stats.genpareto`)
=============================================================

This module contains the Generalized Pareto distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import (
    interval as interval_constraint,
    real,
    strict_positive,
)
from probjax.utils.typing import RngKey

__all__ = ["genpareto"]


class genpareto_gen(rv_continuous, rv_exponential_family):
    """Generalized Pareto continuous random variable.

    The generalized Pareto distribution is a continuous probability distribution
    that is often used to model extreme values. The probability density function is:

    .. math::
        f(x; c, loc, scale) = \frac{1}{scale} (1 + c \frac{x - loc}{scale})^{-1 - 1/c}

    where :math:`c` is the shape parameter, :math:`loc` is the location parameter,
    and :math:`scale` is the scale parameter.

    Parameters
    ----------
    c : float, optional
        Shape parameter. Default is 0.
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'c': real, 'loc': real, 'scale': strict_positive}

    @classmethod
    def support(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Support of the generalized Pareto distribution."""
        c_val = float(jnp.asarray(c))
        loc_val = float(jnp.asarray(loc))
        scale_val = float(jnp.asarray(scale))
        if c_val >= 0.0:
            return interval_constraint(loc_val, float("inf"), closed_right=False)
        upper = loc_val - scale_val / c_val
        return interval_constraint(loc_val, upper, closed_right=False)

    @classmethod
    def _logsf(cls, x, c, loc, scale):
        z = (jnp.asarray(x) - loc) / scale
        c = jnp.asarray(c)
        u = c * z
        safe_u = jnp.where((u > -1) & (u != 0), u, 1.0)
        ratio = jnp.where(
            jnp.abs(u) < 1e-4, 1 - u / 2 + u * u / 3, jnp.log1p(safe_u) / safe_u
        )
        value = -z * ratio
        return jnp.where(
            z < 0,
            0.0,
            jnp.where(jnp.isposinf(z) | ((c < 0) & (1 + c * z <= 0)), -jnp.inf, value),
        )

    @classmethod
    def pdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        return jnp.exp(cls.logpdf(x, c, loc, scale))

    @classmethod
    def logpdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        z = (jnp.asarray(x) - loc) / scale
        c = jnp.asarray(c)
        logsf = cls._logsf(x, c, loc, scale)
        value = -jnp.log(scale) + (1 + c) * jnp.where(c == -1, 0.0, logsf)
        endpoint = jnp.where(
            c == -1, -jnp.log(scale), jnp.where(c < -1, jnp.inf, -jnp.inf)
        )
        value = jnp.where((c < 0) & (1 + c * z == 0), endpoint, value)
        return jnp.where((z < 0) | ((c < 0) & (1 + c * z < 0)), -jnp.inf, value)

    @classmethod
    def cdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        return -jnp.expm1(cls._logsf(x, c, loc, scale))

    @classmethod
    def _inverse_logsf(cls, logq, c, loc, scale):
        finite_logq = jnp.where(jnp.isfinite(logq), logq, 0.0)
        w = -c * finite_logq
        small = jnp.abs(w) < 1e-4
        safe_w = jnp.where(small, 1.0, w)
        ratio = jnp.where(small, 1 + w / 2 + w * w / 6, jnp.expm1(safe_w) / safe_w)
        value = loc - scale * finite_logq * ratio
        endpoint = jnp.where(c < 0, loc - scale / jnp.where(c < 0, c, -1.0), jnp.inf)
        return jnp.where(jnp.isneginf(logq), endpoint, value)

    @classmethod
    def ppf(cls, q, c=0.0, loc=0.0, scale=1.0, **kwargs):
        q, c = jnp.asarray(q), jnp.asarray(c)
        value = cls._inverse_logsf(jnp.log1p(-q), c, loc, scale)
        return jnp.where((q >= 0) & (q <= 1), value, jnp.nan)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        c=0.0,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the generalized Pareto distribution."""
        c, loc, scale = jnp.broadcast_arrays(c, loc, scale)
        dtype = jnp.result_type(c, loc, scale, float)
        u = random.uniform(rng, shape=tuple(shape) + c.shape, dtype=dtype)
        return cls.ppf(u, c, loc, scale)

    @classmethod
    def sf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        return jnp.exp(cls._logsf(x, c, loc, scale))

    @classmethod
    def isf(cls, q, c=0.0, loc=0.0, scale=1.0, **kwargs):
        q, c = jnp.asarray(q), jnp.asarray(c)
        value = cls._inverse_logsf(jnp.log(q), c, loc, scale)
        return jnp.where((q >= 0) & (q <= 1), value, jnp.nan)

    @classmethod
    def logcdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        return jnp.log(-jnp.expm1(cls._logsf(x, c, loc, scale)))

    @classmethod
    def mean(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mean of the generalized Pareto distribution."""
        return jnp.where(c < 1, loc + scale / (1 - c), jnp.inf)

    @classmethod
    def mode(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mode of the generalized Pareto distribution."""
        c, loc, scale = jnp.broadcast_arrays(c, loc, scale)
        return jnp.where(c < -1, loc - scale / jnp.where(c < -1, c, -1), loc)

    @classmethod
    def var(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Variance of the generalized Pareto distribution."""
        return jnp.where(
            c < 0.5,
            scale**2 / ((1 - c) ** 2 * (1 - 2 * c)),
            jnp.inf,
        )

    @classmethod
    def entropy(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the generalized Pareto distribution."""
        return jnp.log(scale) + 1 + c

    @classmethod
    def moment(cls, n, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the generalized Pareto distribution."""
        if n == 0:
            return jnp.ones_like(loc)
        elif n == 1:
            return cls.mean(c, loc, scale)
        elif n == 2:
            return cls.var(c, loc, scale) + cls.mean(c, loc, scale) ** 2
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the generalized Pareto distribution."""
        return jnp.where(
            c < 1 / 3,
            2 * (1 + c) * jnp.sqrt(1 - 2 * c) / (1 - 3 * c),
            jnp.inf,
        )

    @classmethod
    def kurtosis(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the generalized Pareto distribution."""
        return jnp.where(
            c < 1 / 4,
            3 * (1 - 2 * c) * (2 * c**2 + c + 3) / ((1 - 3 * c) * (1 - 4 * c)) - 3,
            jnp.inf,
        )

    @classmethod
    def natural_parameters(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the generalized Pareto distribution."""
        return jnp.array([-1 / c, -1 / scale])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the generalized Pareto distribution."""
        return jnp.array([jnp.log(x), x])

    @classmethod
    def log_partition(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the generalized Pareto distribution."""
        return jnp.log(scale) + jnp.log(1 + c)


genpareto = genpareto_gen(name="genpareto")
