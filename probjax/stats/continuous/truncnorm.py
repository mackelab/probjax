"""
Truncated Normal Distribution (:mod:`probjax.stats.truncnorm`)
=============================================================

This module contains the Truncated Normal distribution.
"""

from typing import Tuple
import math

import jax
import numpy as np

import jax.numpy as jnp
from jax import random
from jax.scipy.special import log_ndtr
from probjax.utils.special.logspace import logdiffexp
from jax.scipy.stats import truncnorm as _truncnorm

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, strict_positive
from probjax.utils.typing import RngKey

__all__ = ["truncnorm"]


def _standardize_bound(bound, loc, scale):
    # Keep infinite endpoints constant under differentiation of loc/scale.
    bound = jnp.asarray(bound)
    finite = jnp.isfinite(bound)
    return jnp.where(finite, (jnp.where(finite, bound, 0) - loc) / scale, bound)


def _log_normal_mass(alpha, beta):
    left = logdiffexp(log_ndtr(beta), log_ndtr(alpha))
    right = logdiffexp(log_ndtr(-alpha), log_ndtr(-beta))
    return jnp.where(alpha > 0, right, left)


@jax.custom_jvp
def _standard_ppf(q, alpha, beta):
    """Bracketed quantile with implicit derivatives of every order."""
    safe_q = jnp.where((q > 0) & (q < 1), q, 0.5)
    lo = jnp.where(
        jnp.isfinite(alpha),
        alpha,
        jnp.minimum(-jnp.sqrt(-2 * jnp.log(safe_q)) - 2, beta - 1),
    )
    hi = jnp.where(
        jnp.isfinite(beta),
        beta,
        jnp.maximum(jnp.sqrt(-2 * jnp.log1p(-safe_q)) + 2, alpha + 1),
    )
    logq = jnp.log(safe_q)

    def step(_, bounds):
        lo, hi = bounds
        mid = lo + 0.5 * (hi - lo)
        below = _truncnorm.logcdf(mid, alpha, beta) < logq
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 64, step, (lo, hi))
    root = lo + 0.5 * (hi - lo)
    return jnp.where(q == 0, alpha, jnp.where(q == 1, beta, root))


@_standard_ppf.defjvp
def _standard_ppf_jvp(primals, tangents):
    q, alpha, beta = primals
    dq, da, db = tangents
    x = _standard_ppf(q, alpha, beta)
    log_phi_x = -0.5 * x * x - 0.5 * jnp.log(2 * jnp.pi)
    inverse_pdf = jnp.exp(_log_normal_mass(alpha, beta) - log_phi_x)

    def boundary_ratio(bound):
        finite = jnp.isfinite(bound)
        safe_bound = jnp.where(finite, bound, 0.0)
        return jnp.where(finite, jnp.exp(0.5 * (x * x - safe_bound * safe_bound)), 0.0)

    dx = (
        inverse_pdf * dq
        + (1 - q) * boundary_ratio(alpha) * da
        + q * boundary_ratio(beta) * db
    )
    return x, dx


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _interval_quadrature(alpha, beta, order):
    """Stable moments for bounded intervals with moderate density variation.

    Centering avoids cancellation in narrow intervals. One-sided tails beyond
    five standard deviations truncate after a relative density drop of at
    least exp(-48); other wide/unbounded intervals use the recurrence.
    """
    right_tail = jnp.isfinite(alpha) & (alpha >= 5) & jnp.isposinf(beta)
    left_tail = jnp.isfinite(beta) & (beta <= -5) & jnp.isneginf(alpha)
    safe_alpha = jnp.where(right_tail, alpha, 5.0)
    safe_beta = jnp.where(left_tail, beta, -5.0)
    effective_alpha = jnp.where(left_tail, safe_beta + 48 / safe_beta, alpha)
    effective_beta = jnp.where(right_tail, safe_alpha + 48 / safe_alpha, beta)
    finite = jnp.isfinite(effective_alpha) & jnp.isfinite(effective_beta)
    a = jnp.where(finite, effective_alpha, 0.0)
    b = jnp.where(finite, effective_beta, 1.0)
    use = (
        right_tail
        | left_tail
        | (
            finite
            & (b > a)
            & ((b - a) * (1 + jnp.maximum(jnp.abs(a), jnp.abs(b))) < 64)
        )
    )
    a, b = jnp.where(use, a, 0.0), jnp.where(use, b, 1.0)
    half, mid = (b - a) / 2, a + (b - a) / 2
    nodes = jnp.asarray(_QUADRATURE_NODES, dtype=alpha.dtype)
    weights = jnp.asarray(_QUADRATURE_WEIGHTS, dtype=alpha.dtype)
    x = mid[..., None] + half[..., None] * nodes
    mode = jnp.clip(jnp.zeros_like(a), a, b)[..., None]
    displacement = (mid[..., None] - mode) + half[..., None] * nodes
    log_density = -0.5 * displacement * (2 * mode + displacement)
    mass = weights * jnp.exp(log_density)
    z = jnp.sum(mass, axis=-1)
    mass = mass / z[..., None]
    mean_node = jnp.sum(mass * nodes, axis=-1)
    centered = half[..., None] * (nodes - mean_node[..., None])
    raw = [jnp.sum(mass * x**k, axis=-1) for k in range(order + 1)]
    central = [jnp.sum(mass * centered**k, axis=-1) for k in range(order + 1)]
    log_z = (
        jnp.log(half * z) - 0.5 * jnp.squeeze(mode, -1) ** 2 - 0.5 * jnp.log(2 * jnp.pi)
    )
    entropy = jnp.log(half * z) - jnp.sum(mass * log_density, axis=-1)
    return use, raw, central, log_z, entropy


def _standard_moments(alpha, beta, order):
    left = logdiffexp(log_ndtr(beta), log_ndtr(alpha))
    right = logdiffexp(log_ndtr(-alpha), log_ndtr(-beta))
    log_z = jnp.where(alpha > 0, right, left)
    pa = jnp.exp(-(alpha**2) / 2 - 0.5 * jnp.log(2 * jnp.pi) - log_z)
    pb = jnp.exp(-(beta**2) / 2 - 0.5 * jnp.log(2 * jnp.pi) - log_z)
    a, b = (
        jnp.where(jnp.isfinite(alpha), alpha, 0),
        jnp.where(jnp.isfinite(beta), beta, 0),
    )
    moments = [jnp.ones_like(log_z)]
    if order:
        moments.append(pa - pb)
    for k in range(2, order + 1):
        moments.append((k - 1) * moments[k - 2] + a ** (k - 1) * pa - b ** (k - 1) * pb)
    use, raw, _, quad_log_z, _ = _interval_quadrature(alpha, beta, order)
    return [jnp.where(use, q, m) for q, m in zip(raw, moments)], jnp.where(
        use, quad_log_z, log_z
    )


class truncnorm_gen(rv_continuous, rv_exponential_family):
    """Truncated Normal continuous random variable.

    The truncated normal distribution is a normal distribution that is bounded
    on both sides. The probability density function is:

    .. math::
        f(x; \\mu, \\sigma, a, b) = \frac{\\phi(\frac{x-\\mu}{\\sigma})}
        {\\sigma(\\Phi(\frac{b-\\mu}{\\sigma}) - \\Phi(\frac{a-\\mu}{\\sigma}))}

    where :math:`\\phi` is the standard normal PDF and :math:`\\Phi` is the standard
    normal CDF.

    Parameters
    ----------
    loc : float, optional
        Mean of the distribution. Default is 0.
    scale : float, optional
        Standard deviation of the distribution. Default is 1.
    a : float, optional
        Lower bound of the truncation. Default is -inf.
    b : float, optional
        Upper bound of the truncation. Default is inf.
    """

    # Define parameter constraints
    parameters = {'loc': real, 'scale': strict_positive, 'a': real, 'b': real}

    @classmethod
    def support(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Support of the truncated normal distribution."""
        return real

    @classmethod
    def pdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Probability density function of the truncated normal distribution."""
        return _truncnorm.pdf(
            x,
            a=_standardize_bound(a, loc, scale),
            b=_standardize_bound(b, loc, scale),
            loc=loc,
            scale=scale,
        )

    @classmethod
    def logpdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log of the probability density function of the truncated normal distribution."""
        return _truncnorm.logpdf(
            x,
            a=_standardize_bound(a, loc, scale),
            b=_standardize_bound(b, loc, scale),
            loc=loc,
            scale=scale,
        )

    @classmethod
    def cdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Cumulative distribution function of the truncated normal distribution."""
        return _truncnorm.cdf(
            x,
            a=_standardize_bound(a, loc, scale),
            b=_standardize_bound(b, loc, scale),
            loc=loc,
            scale=scale,
        )

    @classmethod
    def ppf(cls, q, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Percent point function (inverse of cdf) of the truncated normal distribution."""
        q, loc, scale, a, b = jnp.broadcast_arrays(q, loc, scale, a, b)
        alpha, beta = (
            _standardize_bound(a, loc, scale),
            _standardize_bound(b, loc, scale),
        )
        value = loc + scale * _standard_ppf(q, alpha, beta)
        value = jnp.where(q == 0, a, jnp.where(q == 1, b, value))
        return jnp.where((q >= 0) & (q <= 1) & (a < b) & (scale > 0), value, jnp.nan)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc=0.0,
        scale=1.0,
        a=-jnp.inf,
        b=jnp.inf,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the truncated normal distribution."""
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        event_shape = jnp.broadcast_shapes(loc.shape, scale.shape, a.shape, b.shape)
        loc = jnp.broadcast_to(loc, event_shape)
        scale = jnp.broadcast_to(scale, event_shape)
        a = jnp.broadcast_to(a, event_shape)
        b = jnp.broadcast_to(b, event_shape)
        size = shape + event_shape
        dtype = jnp.result_type(loc, scale, a, b)
        lower = _standardize_bound(a, loc, scale)
        upper = _standardize_bound(b, loc, scale)
        lower = jnp.broadcast_to(lower, event_shape)
        upper = jnp.broadcast_to(upper, event_shape)
        samples = random.truncated_normal(rng, lower, upper, shape=size, dtype=dtype)
        return samples * scale + loc

    @classmethod
    def sf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Survival function (1 - cdf) of the truncated normal distribution."""
        return _truncnorm.sf(
            x,
            a=_standardize_bound(a, loc, scale),
            b=_standardize_bound(b, loc, scale),
            loc=loc,
            scale=scale,
        )

    @classmethod
    def isf(cls, q, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Inverse survival function (inverse of sf) of the truncated normal distribution."""
        return -cls.ppf(
            q, loc=-jnp.asarray(loc), scale=scale, a=-jnp.asarray(b), b=-jnp.asarray(a)
        )

    @classmethod
    def logcdf(cls, x, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log of the cumulative distribution function of the truncated normal distribution."""
        return _truncnorm.logcdf(
            x,
            a=_standardize_bound(a, loc, scale),
            b=_standardize_bound(b, loc, scale),
            loc=loc,
            scale=scale,
        )

    @classmethod
    def mean(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Mean of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        moments, _ = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 1
        )
        return loc + scale * moments[1]

    @classmethod
    def mode(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Mode of the truncated normal distribution."""
        return jnp.clip(loc, a, b)

    @classmethod
    def var(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Variance of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        moments, _ = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 2
        )
        use, _, central, _, _ = _interval_quadrature(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 2
        )
        variance = jnp.where(use, central[2], moments[2] - moments[1] ** 2)
        return scale**2 * jnp.maximum(variance, 0)

    @classmethod
    def entropy(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Entropy of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        moments, log_z = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 2
        )
        use, _, _, _, quad_entropy = _interval_quadrature(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 2
        )
        entropy = log_z + 0.5 * jnp.log(2 * jnp.pi) + 0.5 * moments[2]
        return jnp.log(scale) + jnp.where(use, quad_entropy, entropy)

    @classmethod
    def moment(cls, n, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """n-th non-central moment of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative static integer")
        moments, _ = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), n
        )
        return sum(
            math.comb(n, k) * loc ** (n - k) * scale**k * moments[k]
            for k in range(n + 1)
        )

    @classmethod
    def skew(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Skewness of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        m, _ = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 3
        )
        use, _, central, _, _ = _interval_quadrature(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 3
        )
        numerator = jnp.where(use, central[3], m[3] - 3 * m[1] * m[2] + 2 * m[1] ** 3)
        variance = jnp.where(use, central[2], m[2] - m[1] ** 2)
        return numerator / variance**1.5

    @classmethod
    def kurtosis(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Excess kurtosis of the truncated normal distribution."""
        loc, scale, a, b = jnp.broadcast_arrays(loc, scale, a, b)
        m, _ = _standard_moments(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 4
        )
        use, _, central, _, _ = _interval_quadrature(
            _standardize_bound(a, loc, scale), _standardize_bound(b, loc, scale), 4
        )
        numerator = jnp.where(
            use,
            central[4],
            m[4] - 4 * m[1] * m[3] + 6 * m[1] ** 2 * m[2] - 3 * m[1] ** 4,
        )
        variance = jnp.where(use, central[2], m[2] - m[1] ** 2)
        return numerator / variance**2 - 3

    @classmethod
    def natural_parameters(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Natural parameters of the truncated normal distribution."""
        var = scale**2
        return jnp.array([loc / var, -1.0 / (2.0 * var)])

    @classmethod
    def sufficient_statistics(cls, x, **kwargs):
        """Sufficient statistics of the truncated normal distribution."""
        return jnp.array([x, x**2])

    @classmethod
    def log_partition(cls, loc=0.0, scale=1.0, a=-jnp.inf, b=jnp.inf, **kwargs):
        """Log partition function of the truncated normal distribution."""
        var = scale**2
        return 0.5 * jnp.log(2 * jnp.pi * var) + (loc**2) / (2 * var)


truncnorm = truncnorm_gen(name="truncnorm")
