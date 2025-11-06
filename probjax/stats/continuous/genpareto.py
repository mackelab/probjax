"""
Generalized Pareto Distribution (:mod:`probjax.stats.genpareto`)
=============================================================

This module contains the Generalized Pareto distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import interval as interval_constraint, real, strict_positive
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
    def pdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the generalized Pareto distribution."""
        z = (x - loc) / scale
        return (1 / scale) * (1 + c * z) ** (-1 - 1 / c)

    @classmethod
    def logpdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the generalized Pareto distribution."""
        z = (x - loc) / scale
        return -jnp.log(scale) - (1 + 1 / c) * jnp.log(1 + c * z)

    @classmethod
    def cdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the generalized Pareto distribution."""
        z = (x - loc) / scale
        return 1 - (1 + c * z) ** (-1 / c)

    @classmethod
    def ppf(cls, q, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the generalized Pareto distribution."""
        return loc + scale * ((1 - q) ** (-c) - 1) / c

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        shape: Tuple[int, ...] = (),
        c=0.0,
        loc=0.0,
        scale=1.0,
        **kwargs,
    ):
        """Random variates of the generalized Pareto distribution."""
        u = random.uniform(rng, shape=shape)
        return cls.ppf(u, c, loc, scale)

    @classmethod
    def sf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the generalized Pareto distribution."""
        z = (x - loc) / scale
        return (1 + c * z) ** (-1 / c)

    @classmethod
    def isf(cls, q, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the generalized Pareto distribution."""
        return loc + scale * (q ** (-c) - 1) / c

    @classmethod
    def logcdf(cls, x, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the generalized Pareto distribution."""
        z = (x - loc) / scale
        return jnp.log(1 - (1 + c * z) ** (-1 / c))

    @classmethod
    def mean(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mean of the generalized Pareto distribution."""
        return jnp.where(c < 1, loc + scale / (1 - c), jnp.inf)

    @classmethod
    def mode(cls, c=0.0, loc=0.0, scale=1.0, **kwargs):
        """Mode of the generalized Pareto distribution."""
        return jnp.asarray(loc)

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
