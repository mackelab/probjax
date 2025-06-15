from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import chi2 as _chi2
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, strict_positive, strict_positive_integer

__all__ = ["chi2"]


class chi2_gen(rv_continuous):
    """Chi-squared continuous random variable.

    The chi-squared distribution with degrees of freedom `df`, location `loc`, and scale `scale`.

    Parameters
    ----------
    df : float, optional
        Degrees of freedom. Default is 1.
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    """

    parameters = {"df": strict_positive_integer, "loc": real, "scale": strict_positive}

    @classmethod
    def support(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Support of the chi-squared distribution."""
        return real

    @classmethod
    def pdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the chi-squared distribution."""
        return _chi2.pdf(x, df, loc, scale)

    @classmethod
    def logpdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the chi-squared distribution."""
        return _chi2.logpdf(x, df, loc, scale)

    @classmethod
    def cdf(cls, x, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the chi-squared distribution."""
        raise NotImplementedError("CDF not implemented for chi2 distribution")

    @classmethod
    def ppf(cls, q, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the chi-squared distribution."""
        raise NotImplementedError("PPF not implemented for chi2 distribution")

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        df=1.0,
        loc=0.0,
        scale=1.0,
        **kwargs,
    ) -> Float[Array, "..."]:
        """Random variates of the chi-squared distribution."""
        df = jnp.asarray(df)
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(df.shape, loc.shape, scale.shape)
        return random.chisquare(rng, df, shape=shape + event_shape) * scale + loc

    @classmethod
    def mean(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Mean of the chi-squared distribution."""
        return jnp.asarray(df) * jnp.asarray(scale) + jnp.asarray(loc)

    @classmethod
    def var(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Variance of the chi-squared distribution."""
        return 2 * jnp.asarray(df) * jnp.asarray(scale) ** 2

    @classmethod
    def entropy(cls, df=1.0, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the chi-squared distribution."""
        # See scipy.stats.chi2.entropy for formula
        df = jnp.asarray(df)
        scale = jnp.asarray(scale)
        return (
            0.5 * df
            + jnp.log(2 * scale)
            + jax.lax.lgamma(0.5 * df)
            + (1 - 0.5 * df) * jax.lax.digamma(0.5 * df)
        )

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of chi-squared distribution parameters.

        The MLE for the chi-squared distribution has a closed-form solution:
        - loc = 0 (fixed)
        - scale = mean(data) / df
        - df = mean(data) / var(data)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (df, loc, scale)
        """
        data = jnp.asarray(data)
        loc = 0.0  # Fixed at 0
        mean = jnp.mean(data)
        var = jnp.var(data)
        df = mean**2 / var
        scale = mean / df
        return (df, loc, scale)


chi2 = chi2_gen(name="chi2")
