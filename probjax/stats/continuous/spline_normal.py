"""
Spline-warped normal distribution (:mod:`probjax.stats.spline_normal`)
=======================================================================

``X = spline(Z)`` with ``Z ~ N(0, 1)`` and a monotone rational-quadratic
spline: a one-dimensional normalizing flow exposed as an ordinary
distribution.

This is the most flexible univariate family here that still has closed-form
density *and* quantile: the spline's inverse and log-determinant are analytic
(:mod:`probjax.stats.bijective.rational_quadratic`), so

    logpdf(x) = norm.logpdf(z) + log |dz/dx|,     z = spline^-1(x)
    ppf(q)    = spline(norm.ppf(q))
    rvs       = spline(normal sample)

all cost one spline evaluation. Outside the knot range the spline extrapolates
linearly, so the support is the whole real line and the tails are Gaussian
rescaled by the boundary slope.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous
from probjax.stats.bijective.rational_quadratic import (
    inv_rational_quadratic_spline,
    rational_quadratic_spline,
)
from probjax.stats.constraints import real, strict_positive
from probjax.utils.typing import Array, RngKey

__all__ = ["spline_normal"]


def _vectorized(fn, num_knots: int, n_knot_args: int = 3, n_out: int = 1):
    """Lift a scalar-x spline call over arbitrary batch shapes.

    The core spline functions take a scalar ``x`` with 1-D knot arrays; the
    inverse returns ``(x, logdet)``, hence the output count.
    """
    knots = ",".join([f"({num_knots})"] * n_knot_args)
    out = ",".join(["()"] * n_out)
    return jnp.vectorize(fn, signature=f"(),{knots}->{out}")


class spline_normal_gen(rv_continuous):
    """Normal distribution warped by a monotone rational-quadratic spline.

    Parameters
    ----------
    x_pos : array_like
        Increasing knot positions in the *latent* (normal) coordinate, shape
        ``(..., num_bins + 1)``.
    y_pos : array_like
        Increasing knot positions in the *data* coordinate, same shape.
    knot_slopes : array_like
        Strictly positive derivatives at the knots, same shape.

    Notes
    -----
    Zero-ish parameters -- evenly spaced matching knots with unit slopes -- give
    back a standard normal, which makes this a well-behaved default head for a
    conditioner whose last layer is zero-initialised.
    """

    parameters = {
        "x_pos": real,
        "y_pos": real,
        "knot_slopes": strict_positive,
    }

    @classmethod
    def param_sizes(cls, num_bins: int = 8, **kwargs) -> dict:
        """Trailing size of each parameter: one entry per knot."""
        del kwargs
        if num_bins < 1:
            raise ValueError(f"num_bins must be positive; got {num_bins}.")
        return {
            "x_pos": num_bins + 1,
            "y_pos": num_bins + 1,
            "knot_slopes": num_bins + 1,
        }

    @classmethod
    def support(cls, **kwargs):
        """Linear tails, so the support is all of R."""
        return real

    @classmethod
    def _to_latent(cls, x, x_pos, y_pos, knot_slopes):
        """``(z, log|dz/dx|)`` -- the spline runs latent -> data, so invert."""
        num_knots = jnp.asarray(x_pos).shape[-1]
        fn = _vectorized(inv_rational_quadratic_spline, num_knots, n_out=2)
        return fn(jnp.asarray(x), x_pos, y_pos, knot_slopes)

    @classmethod
    def _to_data(cls, z, x_pos, y_pos, knot_slopes):
        num_knots = jnp.asarray(x_pos).shape[-1]
        fn = _vectorized(rational_quadratic_spline, num_knots, n_out=1)
        return fn(jnp.asarray(z), x_pos, y_pos, knot_slopes)

    @classmethod
    def logpdf(cls, x, x_pos=None, y_pos=None, knot_slopes=None, **kwargs):
        """Change of variables through the spline."""
        del kwargs
        z, logdet = cls._to_latent(x, x_pos, y_pos, knot_slopes)
        return jax.scipy.stats.norm.logpdf(z) + logdet

    @classmethod
    def cdf(cls, x, x_pos=None, y_pos=None, knot_slopes=None, **kwargs):
        """The spline is increasing, so the CDF is the normal CDF of the latent."""
        del kwargs
        z, _ = cls._to_latent(x, x_pos, y_pos, knot_slopes)
        return jax.scipy.stats.norm.cdf(z)

    @classmethod
    def ppf(cls, q, x_pos=None, y_pos=None, knot_slopes=None, **kwargs):
        """Push the normal quantile through the spline."""
        del kwargs
        z = jax.scipy.stats.norm.ppf(jnp.asarray(q))
        return cls._to_data(z, x_pos, y_pos, knot_slopes)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        x_pos=None,
        y_pos=None,
        knot_slopes=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Sample the latent normal and warp it."""
        del kwargs
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(x_pos)[:-1], jnp.shape(y_pos)[:-1], jnp.shape(knot_slopes)[:-1]
        )
        z = random.normal(rng, tuple(shape) + batch_shape)
        return cls._to_data(z, x_pos, y_pos, knot_slopes)


spline_normal = spline_normal_gen(name="spline_normal")
