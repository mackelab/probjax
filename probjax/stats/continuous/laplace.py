"""
Laplace Distribution (:mod:`probjax.stats.laplace`)
==================================================

This module contains the Laplace distribution.
"""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import laplace as _laplace

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["laplace"]


class laplace_gen(rv_continuous, rv_exponential_family):
    """Laplace continuous random variable.

    The Laplace distribution with location parameter `loc` and scale parameter `scale`.

    Parameters
    ----------
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Scale parameter. Default is 1.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, **kwargs):
        """Support of the Laplace distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Probability density function of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(cls.logpdf(x, loc, scale, **kwargs))

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the probability density function of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        return _laplace.logpdf(x, loc, scale)

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Cumulative distribution function of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
        """
        return _laplace.cdf(x, loc, scale)

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Log of the cumulative distribution function of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        logcdf : ndarray
            Log of the cumulative distribution function evaluated at x
        """
        return jnp.log(_laplace.cdf(x, loc, scale))

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Percent point function (inverse of cdf) of the Laplace distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        ppf : ndarray
            Quantile corresponding to the lower tail probability q
        """
        # For Laplace distribution, ppf(q) = loc + scale * sign(q-0.5) * ln(1-2|q-0.5|)
        q = jnp.asarray(q)
        # For q <= 0.5: loc + scale * log(2q)
        # For q > 0.5: loc - scale * log(2(1-q))
        return jnp.where(
            q <= 0.5,
            loc + scale * jnp.log(2.0 * q),
            loc - scale * jnp.log(2.0 * (1.0 - q)),
        )

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        loc=0.0,
        scale=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Random variates of the Laplace distribution.

        Parameters
        ----------
        rng : RngKey
            JAX PRNG key for random number generation
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.
        shape : tuple of ints, optional
            Output shape. Default is (), meaning a single value.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        event_shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        return random.laplace(rng, shape=shape + event_shape) * scale + loc

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, **kwargs):
        """Survival function (1 - cdf) of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        sf : ndarray
            Survival function evaluated at x
        """
        return _laplace.sf(x, loc, scale)

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, **kwargs):
        """Inverse survival function (inverse of sf) of the Laplace distribution.

        Parameters
        ----------
        q : array_like
            upper tail probability
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        isf : ndarray
            Quantile corresponding to the upper tail probability q
        """
        return _laplace.isf(q, loc, scale)

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, **kwargs):
        """Mean of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        mean : float
            Mean of the distribution
        """
        return jnp.asarray(loc)

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, **kwargs):
        """Mode of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        mode : float
            Mode of the distribution
        """
        return jnp.asarray(loc)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, **kwargs):
        """Variance of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        var : float
            Variance of the distribution
        """
        return 2.0 * scale**2

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, **kwargs):
        """Entropy of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        entropy : float
            Entropy of the distribution
        """
        return 1.0 + jnp.log(2.0 * scale)

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, **kwargs):
        """n-th non-central moment of the Laplace distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        moment : float
            n-th non-central moment
        """
        n = jnp.asarray(n)
        # For odd n, central moment is 0
        # For even n, central moment is n! * scale^n
        # Convert to non-central moment by adding loc

        # Central moments
        even_central = jnp.exp(
            jnp.log(jnp.prod(jnp.arange(1, n + 1))) + n * jnp.log(scale)
        )
        central_moment = jnp.where(n % 2 == 0, even_central, 0.0)

        # Use binomial expansion to compute non-central moments
        k = jnp.arange(n + 1)
        binomial_coef = jnp.exp(jnp.log(jnp.math.comb(n, k)))
        loc_powers = loc ** (n - k)

        # Get central moments for each k (which are 0 for odd k)
        k_central_moments = jnp.where(
            k % 2 == 0,
            jnp.exp(
                jnp.log(jnp.prod(jnp.where(k >= 1, jnp.arange(1, k + 1), 1)))
                + k * jnp.log(scale)
            ),
            0.0,
        )

        # Combine using binomial expansion
        return jnp.sum(binomial_coef * loc_powers * k_central_moments)

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, **kwargs):
        """Skewness of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        skew : float
            Skewness of the distribution
        """
        return jnp.zeros_like(loc)  # Skewness is always 0 (symmetric distribution)

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, **kwargs):
        """Excess kurtosis of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        kurtosis : float
            Excess kurtosis of the distribution
        """
        return 3.0 * jnp.ones_like(loc)

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, **kwargs):
        """Natural parameters of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        natural_parameters : tuple
            Natural parameters of the distribution
        """
        return jnp.array([loc, -1.0 / scale])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the Laplace distribution.

        Parameters
        ----------
        x : array_like
            Data points

        Returns
        -------
        sufficient_statistics : tuple
            Sufficient statistics of the distribution
        """
        return jnp.array([x, jnp.abs(x)])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, **kwargs):
        """Log partition function of the Laplace distribution.

        Parameters
        ----------
        loc : float, optional
            Location parameter. Default is 0.
        scale : float, optional
            Scale parameter. Default is 1.

        Returns
        -------
        log_partition : float
            Log partition function of the distribution
        """
        return jnp.log(2.0 * scale) + jnp.abs(loc) / scale

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Laplace distribution parameters.

        The MLE for the Laplace distribution has a closed-form solution:
        - loc = median(data)
        - scale = mean(|data - loc|)

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted parameters (loc, scale)
        """
        if weights is not None:
            raise NotImplementedError(
                "Weighted fitting is not implemented for the Laplace distribution."
            )
        data = jnp.asarray(data)
        loc = jnp.median(data)
        scale = jnp.mean(jnp.abs(data - loc))
        return (loc, scale)


laplace = laplace_gen(name="laplace")
