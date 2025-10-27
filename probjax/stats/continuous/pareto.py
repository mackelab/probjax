"""
Pareto Distribution (:mod:`probjax.stats.pareto`)
================================================

This module contains the Pareto distribution.
"""

from typing import Tuple

import jax.numpy as jnp
from jax.scipy.stats import pareto as _pareto
from jaxtyping import ArrayLike, PRNGKeyArray

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive

__all__ = ["pareto"]


class pareto_gen(rv_continuous, rv_exponential_family):
    """Pareto continuous random variable.

    The Pareto distribution is a power law probability distribution that is used
    in describing social, scientific, geophysical, actuarial, and many other types
    of observable phenomena. The probability density function is:

    .. math::
        f(x; b, \alpha) = \frac{\alpha b^\alpha}{x^{\alpha+1}}

    for :math:`x \\geq b` and :math:`\alpha > 0`.

    Parameters
    ----------
    b : float, optional
        Scale parameter (minimum value). Default is 1.
    alpha : float, optional
        Shape parameter (tail index). Default is 1.
    """

    # Define parameter constraints
    parameters = {'b': strict_positive, 'alpha': strict_positive}

    @classmethod
    def support(cls, b=1.0, alpha=1.0, **kwargs):
        """Support of the Pareto distribution."""
        return real

    @classmethod
    def pdf(cls, x, b=1.0, alpha=1.0, **kwargs):
        """Probability density function of the Pareto distribution."""
        return _pareto.pdf(x, b=b, scale=alpha)

    @classmethod
    def logpdf(cls, x, b=1.0, alpha=1.0, **kwargs):
        """Log of the probability density function of the Pareto distribution."""
        return _pareto.logpdf(x, b=b, scale=alpha)

    @classmethod
    def cdf(cls, x, b=1.0, alpha=1.0, **kwargs):
        """Cumulative distribution function of the Pareto distribution."""
        return _pareto.cdf(x, b=b, scale=alpha)

    @classmethod
    def ppf(cls, q, b=1.0, alpha=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the Pareto distribution."""
        return _pareto.ppf(q, b=b, scale=alpha)

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        b=1.0,
        alpha=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Pareto distribution."""
        b = jnp.asarray(b)
        alpha = jnp.asarray(alpha)
        event_shape = jnp.broadcast_shapes(b.shape, alpha.shape)
        return _pareto.rvs(b=b, scale=alpha, size=shape + event_shape, key=rng)

    @classmethod
    def sf(cls, x, b=1.0, alpha=1.0, **kwargs):
        """Survival function (1 - cdf) of the Pareto distribution."""
        return _pareto.sf(x, b=b, scale=alpha)

    @classmethod
    def isf(cls, q, b=1.0, alpha=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the Pareto distribution."""
        return _pareto.isf(q, b=b, scale=alpha)

    @classmethod
    def logcdf(cls, x, b=1.0, alpha=1.0, **kwargs):
        """Log of the cumulative distribution function of the Pareto distribution."""
        return _pareto.logcdf(x, b=b, scale=alpha)

    @classmethod
    def mean(cls, b=1.0, alpha=1.0, **kwargs):
        """Mean of the Pareto distribution."""
        return jnp.where(alpha > 1, alpha * b / (alpha - 1), jnp.inf)

    @classmethod
    def mode(cls, b=1.0, alpha=1.0, **kwargs):
        """Mode of the Pareto distribution."""
        return jnp.asarray(b)

    @classmethod
    def var(cls, b=1.0, alpha=1.0, **kwargs):
        """Variance of the Pareto distribution."""
        return jnp.where(
            alpha > 2, b**2 * alpha / ((alpha - 1) ** 2 * (alpha - 2)), jnp.inf
        )

    @classmethod
    def entropy(cls, b=1.0, alpha=1.0, **kwargs):
        """Entropy of the Pareto distribution."""
        return jnp.log(b / alpha) + 1 + 1 / alpha

    @classmethod
    def moment(cls, n, b=1.0, alpha=1.0, **kwargs):
        """n-th non-central moment of the Pareto distribution."""
        if n == 0:
            return jnp.ones_like(b)
        elif n == 1:
            return cls.mean(b=b, alpha=alpha)
        elif n == 2:
            return cls.var(b=b, alpha=alpha) + cls.mean(b=b, alpha=alpha) ** 2
        else:
            raise NotImplementedError(f"Moment of order {n} not implemented")

    @classmethod
    def skew(cls, b=1.0, alpha=1.0, **kwargs):
        """Skewness of the Pareto distribution."""
        return jnp.where(
            alpha > 3,
            2 * (1 + alpha) / (alpha - 3) * jnp.sqrt((alpha - 2) / alpha),
            jnp.inf,
        )

    @classmethod
    def kurtosis(cls, b=1.0, alpha=1.0, **kwargs):
        """Excess kurtosis of the Pareto distribution."""
        return jnp.where(
            alpha > 4,
            6
            * (alpha**3 + alpha**2 - 6 * alpha - 2)
            / (alpha * (alpha - 3) * (alpha - 4)),
            jnp.inf,
        )

    @classmethod
    def natural_parameters(cls, b=1.0, alpha=1.0, **kwargs):
        """Natural parameters of the Pareto distribution."""
        return jnp.array([-alpha - 1, alpha * jnp.log(b)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the Pareto distribution."""
        return jnp.array([jnp.log(x), 1])

    @classmethod
    def log_partition(cls, b=1.0, alpha=1.0, **kwargs):
        """Log partition function of the Pareto distribution."""
        return jnp.log(alpha * b**alpha)

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of Pareto distribution parameters.

        The MLE for the Pareto distribution has a closed-form solution:
        - b = min(data)
        - alpha = n / sum(log(data/b))

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (b, alpha)
        """
        data = jnp.asarray(data)
        b = jnp.min(data)
        alpha = len(data) / jnp.sum(jnp.log(data / b))
        return (b, alpha)


pareto = pareto_gen(name="pareto")
