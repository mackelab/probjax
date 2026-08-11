"""Monotone neural bijectors in natural parameterization.

Each bijector here is an analytic, strictly increasing map ``G(y, *natural)``
used in the *density* direction (data -> base): its value and derivative are
closed-form and differentiable, so ``logpdf`` and training gradients never
touch a numerical solver. The bijector's *forward* (base -> data, used for
sampling) solves ``G(y) = x`` with a bracketed bisection and is registered via
:class:`~probjax.core.custom_inverse`:

    forward:  y = bijector(x, *natural)   (root solve, do not differentiate)
    inverse:  x = G(y, *natural),  logdet = log G'(y, *natural)   (analytic)

Conventions match the rest of :mod:`probjax.stats.bijective`: ``x`` comes first,
the remaining arguments are *already-constrained* natural parameters with a
trailing component axis, and mapping an unconstrained network output onto them
is the caller's job (see :mod:`probjax.nn.generative.nflows.config`). Everything
broadcasts over leading batch dimensions, combining via ``x[..., None]`` and
reducing over the last axis.

Implemented families:

* :func:`deep_sigmoid` — NAF's deep sigmoidal flow (Huang et al., 2018).
  ``(log_weights, slopes, biases)``, each of shape ``(..., K)``.
* :func:`unconstrained_monotone` — UMNN-style monotone network, integrated with
  fixed Gauss-Legendre quadrature (Wehenkel & Louppe, 2019). The weights
  ``(hidden_weights, hidden_biases, out_weights, out_bias, offset)`` are
  genuinely unconstrained; positivity is architectural (see below).
* :func:`sos_polynomial` — sum-of-squares polynomial flow (Jaini et al., 2019),
  acting on ``x / bound`` with linear tails outside.
  ``(coefficients, constant)`` with ``coefficients`` of shape ``(..., K, r + 1)``.
* :func:`bernstein` — monotone Bernstein polynomial on an interval with identity
  tails (Sick et al., 2021). ``theta`` increasing from 0 to 1, shape ``(..., m + 1)``.
* :func:`mixture_cdf` — Gaussianization-flow kernel layer: logistic-mixture CDF
  followed by the logistic quantile (Meng et al., 2020).
  ``(log_weights, locs, scales)``, each of shape ``(..., K)``.

Weights are taken in **log** space (``log_weights``, a point on the log-simplex)
rather than as probabilities: every closed form below consumes them through
``logsumexp``, so keeping the whole chain in log space avoids both a redundant
``log`` and an epsilon floor against softmax underflow at the boundary.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from probjax.core import custom_inverse

__all__ = [
    "bernstein",
    "deep_sigmoid",
    "inv_bernstein",
    "inv_deep_sigmoid",
    "inv_mixture_cdf",
    "inv_sos_polynomial",
    "inv_unconstrained_monotone",
    "mixture_cdf",
    "sos_polynomial",
    "unconstrained_monotone",
]

_EPS = 1e-6


def _solve_increasing(
    g,
    target,
    num_expansions: int = 24,
    num_bisections: int = 60,
    rtol: float = 1e-3,
    atol: float = 1e-4,
):
    """Solve ``g(t) = target`` elementwise for strictly increasing ``g``.

    Doubles a symmetric bracket around 0 until it contains the root, then
    bisects. Not differentiable — callers differentiate through the analytic
    inverse instead.

    Returns NaN where the solve did not converge. Comparisons are written so a
    non-finite ``g`` counts as "did not bracket" rather than falling through:
    with a bare ``g(lo) > target``, a NaN makes both directions read False, so
    expansion stops immediately and bisection walks to the edge of the bracket,
    returning a confidently wrong root. A NaN is recoverable information for the
    caller; a silently wrong inverse is not.

    ``num_expansions`` only has to cover the map's growth. Every family here has
    linear or identity tails, so a reach of ``2**24`` is ample, and further
    doubling would only produce brackets too wide for float32 bisection to
    resolve.
    """
    target = jnp.asarray(target)
    lo = jnp.full_like(target, -1.0)
    hi = jnp.full_like(target, 1.0)

    def expand(i, carry):
        lo, hi = carry
        width = 2.0**i
        g_lo, g_hi = g(lo), g(hi)
        push_lo = jnp.where(jnp.isfinite(g_lo), g_lo > target, False)
        push_hi = jnp.where(jnp.isfinite(g_hi), g_hi < target, False)
        return jnp.where(push_lo, lo - width, lo), jnp.where(push_hi, hi + width, hi)

    lo, hi = jax.lax.fori_loop(0, num_expansions, expand, (lo, hi))

    def bisect(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        # A non-finite midpoint cannot be trusted to pick a side; keep the
        # bracket and let the residual check below flag the failure.
        go_left = jnp.where(jnp.isfinite(g_mid), g_mid > target, False)
        return jnp.where(go_left, lo, mid), jnp.where(go_left, mid, hi)

    lo, hi = jax.lax.fori_loop(0, num_bisections, bisect, (lo, hi))
    root = 0.5 * (lo + hi)

    residual = jnp.abs(g(root) - target)
    converged = residual <= atol + rtol * jnp.abs(target)
    return jnp.where(converged, root, jnp.nan)


# ---------------------------------------------------------------------------
# NAF — deep sigmoidal flow
# ---------------------------------------------------------------------------


def _dsf_value_and_logdet(t, log_weights, slopes, biases):
    t = jnp.asarray(t)
    log_w = jnp.asarray(log_weights)
    a = jnp.asarray(slopes)
    b = jnp.asarray(biases)

    pre = a * t[..., None] + b
    # s = sum_k w_k sigmoid(pre_k), computed in log space for stability
    log_s = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(pre), axis=-1)
    log_1ms = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(-pre), axis=-1)
    value = log_s - log_1ms  # logit(s)

    # d value / dt = (sum_k w_k sigmoid'(pre_k) a_k) / (s * (1 - s))
    log_ds = jax.nn.logsumexp(
        log_w + jax.nn.log_sigmoid(pre) + jax.nn.log_sigmoid(-pre) + jnp.log(a),
        axis=-1,
    )
    log_grad = log_ds - log_s - log_1ms
    return value, log_grad


@partial(custom_inverse, inv_argnum=0)
def deep_sigmoid(
    x: ArrayLike, log_weights: ArrayLike, slopes: ArrayLike, biases: ArrayLike
):
    """NAF deep-sigmoidal transform (sampling direction; root solve)."""
    return _solve_increasing(
        lambda t: _dsf_value_and_logdet(t, log_weights, slopes, biases)[0],
        jnp.asarray(x),
    )


def inv_deep_sigmoid(
    y: ArrayLike, log_weights: ArrayLike, slopes: ArrayLike, biases: ArrayLike
):
    """Analytic data -> base direction; returns ``(x, logdet)``."""
    return _dsf_value_and_logdet(y, log_weights, slopes, biases)


deep_sigmoid.definv_and_logdet(inv_deep_sigmoid)


# ---------------------------------------------------------------------------
# UNAF — unconstrained monotone (neural-integrand) flow
# ---------------------------------------------------------------------------

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_GL_NODES = jnp.asarray(_GL_NODES)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS)
_MIN_INTEGRAND = 1e-2


def _umnn_integrand(
    t, hidden_weights, hidden_biases, out_weights, out_bias, min_integrand
):
    """Positive scalar integrand: a one-hidden-layer net with a softplus head.

    The softplus is architectural rather than a parameter constraint — it is
    what makes the integrand (and hence the map) positive for *unconstrained*
    weights, which is the point of UMNN.
    """
    hidden = jnp.tanh(hidden_weights * t[..., None] + hidden_biases)
    raw = jnp.sum(out_weights * hidden, axis=-1) + out_bias
    return jax.nn.softplus(raw) + min_integrand


def _umnn_value_and_logdet(
    t,
    hidden_weights,
    hidden_biases,
    out_weights,
    out_bias,
    offset,
    min_integrand=_MIN_INTEGRAND,
):
    t = jnp.asarray(t)
    hidden_weights = jnp.asarray(hidden_weights)
    hidden_biases = jnp.asarray(hidden_biases)
    out_weights = jnp.asarray(out_weights)
    out_bias = jnp.asarray(out_bias)

    # int_0^t integrand via Gauss-Legendre on [0, t]
    half_t = 0.5 * t
    nodes = half_t[..., None] * (_GL_NODES + 1.0)  # (..., 32)
    vals = _umnn_integrand(
        nodes,
        hidden_weights[..., None, :],
        hidden_biases[..., None, :],
        out_weights[..., None, :],
        out_bias[..., None],
        min_integrand,
    )
    integral = half_t * jnp.sum(_GL_WEIGHTS * vals, axis=-1)

    value = integral + offset
    log_grad = jnp.log(
        _umnn_integrand(
            t, hidden_weights, hidden_biases, out_weights, out_bias, min_integrand
        )
    )
    return value, log_grad


@partial(custom_inverse, inv_argnum=0)
def unconstrained_monotone(
    x: ArrayLike,
    hidden_weights: ArrayLike,
    hidden_biases: ArrayLike,
    out_weights: ArrayLike,
    out_bias: ArrayLike,
    offset: ArrayLike,
    min_integrand: float = _MIN_INTEGRAND,
):
    """UMNN transform (sampling direction; root solve).

    All weights are unconstrained: monotonicity comes from the softplus head on
    the integrand, floored at ``min_integrand`` to keep the map strictly
    increasing.
    """
    return _solve_increasing(
        lambda t: _umnn_value_and_logdet(
            t,
            hidden_weights,
            hidden_biases,
            out_weights,
            out_bias,
            offset,
            min_integrand,
        )[0],
        jnp.asarray(x),
    )


def inv_unconstrained_monotone(
    y: ArrayLike,
    hidden_weights: ArrayLike,
    hidden_biases: ArrayLike,
    out_weights: ArrayLike,
    out_bias: ArrayLike,
    offset: ArrayLike,
    min_integrand: float = _MIN_INTEGRAND,
):
    """Analytic data -> base direction; returns ``(x, logdet)``."""
    return _umnn_value_and_logdet(
        y,
        hidden_weights,
        hidden_biases,
        out_weights,
        out_bias,
        offset,
        min_integrand,
    )


unconstrained_monotone.definv_and_logdet(inv_unconstrained_monotone)


# ---------------------------------------------------------------------------
# SOS — sum-of-squares polynomial flow
# ---------------------------------------------------------------------------


_SOS_BOUND = 5.0


def _sos_value_and_logdet(t, coefficients, constant, bound=_SOS_BOUND):
    t = jnp.asarray(t)
    coeffs = jnp.asarray(coefficients)  # (..., K, r+1)
    degree = coeffs.shape[-1] - 1

    # The polynomial is evaluated in u = t / bound, clipped to [-1, 1]. Raw t
    # would be raised to the power 2*degree+1 (t**7 by default), which overflows
    # float32 above |t| ~ 340 and makes the derivative -- and hence the log-det
    # -- infinite well before that; in u the powers stay O(1) everywhere.
    u = t / bound
    u_in = jnp.clip(u, -1.0, 1.0)

    def slope(v):
        """G'(t) = eps + sum_k poly_k(v)^2, positive by construction."""
        powers = v[..., None] ** jnp.arange(degree + 1)  # (..., r+1)
        poly_vals = jnp.sum(coeffs * powers[..., None, :], axis=-1)  # (..., K)
        return _EPS + jnp.sum(poly_vals**2, axis=-1)

    def antiderivative(v):
        """int_0^v (eps + sum_k poly_k^2), via anti-diagonal sums of the
        coefficient outer product (the squared polynomial's coefficients)."""
        outer = coeffs[..., :, None] * coeffs[..., None, :]  # (..., K, r+1, r+1)
        degree2 = 2 * degree
        v_pows = v[..., None] ** jnp.arange(1, degree2 + 2)  # v^{j+1}
        out = _EPS * v
        for j in range(degree2 + 1):
            b_j = jnp.zeros(v.shape) if v.ndim else jnp.asarray(0.0)
            for l in range(max(0, j - degree), min(j, degree) + 1):
                b_j = b_j + jnp.sum(outer[..., l, j - l], axis=-1)
            out = out + b_j * v_pows[..., j] / (j + 1)
        return out

    # Inside the bound this is the local slope and the offset term vanishes;
    # outside, both freeze at the boundary, giving a linear tail that is
    # continuous in value *and* slope.
    edge_slope = slope(u_in)
    value = constant + bound * antiderivative(u_in) + (t - bound * u_in) * edge_slope
    return value, jnp.log(edge_slope)


@partial(custom_inverse, inv_argnum=0)
def sos_polynomial(
    x: ArrayLike,
    coefficients: ArrayLike,
    constant: ArrayLike,
    bound: float = _SOS_BOUND,
):
    """Sum-of-squares polynomial transform (sampling direction; root solve).

    ``coefficients`` has shape ``(..., num_polys, degree + 1)``; positivity of
    the derivative is structural, so no constraint is required on its entries.
    The polynomial acts on ``x / bound`` within ``|x| <= bound`` and continues
    linearly outside, so the map is a well-conditioned bijection on all of R.
    """
    return _solve_increasing(
        lambda t: _sos_value_and_logdet(t, coefficients, constant, bound)[0],
        jnp.asarray(x),
    )


def inv_sos_polynomial(
    y: ArrayLike,
    coefficients: ArrayLike,
    constant: ArrayLike,
    bound: float = _SOS_BOUND,
):
    """Analytic data -> base direction; returns ``(x, logdet)``."""
    return _sos_value_and_logdet(y, coefficients, constant, bound)


sos_polynomial.definv_and_logdet(inv_sos_polynomial)


# ---------------------------------------------------------------------------
# BPF — Bernstein polynomial flow
# ---------------------------------------------------------------------------

_BERNSTEIN_BOUND = 5.0


def _binom(n):
    from math import comb

    return jnp.asarray([comb(n, i) for i in range(n + 1)], dtype=jnp.float32)


def _bernstein_value_and_logdet(t, theta, bound=_BERNSTEIN_BOUND):
    t = jnp.asarray(t)
    theta = jnp.asarray(theta)  # (..., m+1), increasing, theta_0 = 0, theta_m = 1
    m = theta.shape[-1] - 1
    weights = jnp.diff(theta, axis=-1)  # (..., m)

    u = jnp.clip((t + bound) / (2.0 * bound), 0.0, 1.0)
    i = jnp.arange(m + 1)
    basis = _binom(m) * u[..., None] ** i * (1.0 - u[..., None]) ** (m - i)
    inside_value = -bound + 2.0 * bound * jnp.sum(theta * basis, axis=-1)

    i1 = jnp.arange(m)
    basis_d = _binom(m - 1) * u[..., None] ** i1 * (1.0 - u[..., None]) ** (m - 1 - i1)
    inside_grad = m * jnp.sum(weights * basis_d, axis=-1)  # d value / dt

    inside = jnp.abs(t) < bound
    value = jnp.where(inside, inside_value, t)
    log_grad = jnp.where(inside, jnp.log(inside_grad + _EPS), 0.0)
    return value, log_grad


@partial(custom_inverse, inv_argnum=0)
def bernstein(x: ArrayLike, theta: ArrayLike, bound: float = _BERNSTEIN_BOUND):
    """Monotone Bernstein-polynomial transform (sampling direction; root solve).

    ``theta`` must be increasing with ``theta[..., 0] == 0`` and
    ``theta[..., -1] == 1``; outside ``|x| < bound`` the map is the identity.
    """
    return _solve_increasing(
        lambda t: _bernstein_value_and_logdet(t, theta, bound)[0], jnp.asarray(x)
    )


def inv_bernstein(y: ArrayLike, theta: ArrayLike, bound: float = _BERNSTEIN_BOUND):
    """Analytic data -> base direction; returns ``(x, logdet)``."""
    return _bernstein_value_and_logdet(y, theta, bound)


bernstein.definv_and_logdet(inv_bernstein)


# ---------------------------------------------------------------------------
# GF — logistic-mixture CDF + logistic quantile (Gaussianization kernel layer)
# ---------------------------------------------------------------------------
#
# The quantile is the logistic (logit) rather than the probit of the original
# paper: logit(mixture-CDF) has exactly linear tails, so the map is a
# numerically stable bijection of the whole real line (the probit version
# saturates in float32 beyond |x| ~ 4.8). The two differ only by the fixed
# smooth reparameterization probit∘logistic.


def _mixture_cdf_value_and_logdet(t, log_weights, locs, scales):
    t = jnp.asarray(t)
    log_w = jnp.asarray(log_weights)
    mu = jnp.asarray(locs)
    s = jnp.asarray(scales)

    z = (t[..., None] - mu) / s
    log_cdf = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(z), axis=-1)
    log_1mcdf = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(-z), axis=-1)
    value = log_cdf - log_1mcdf  # logit(mixture cdf)

    log_pdf_mix = jax.nn.logsumexp(
        log_w + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z) - jnp.log(s), axis=-1
    )
    log_grad = log_pdf_mix - log_cdf - log_1mcdf
    return value, log_grad


@partial(custom_inverse, inv_argnum=0)
def mixture_cdf(
    x: ArrayLike, log_weights: ArrayLike, locs: ArrayLike, scales: ArrayLike
):
    """Gaussianization kernel transform (sampling direction; root solve)."""
    return _solve_increasing(
        lambda t: _mixture_cdf_value_and_logdet(t, log_weights, locs, scales)[0],
        jnp.asarray(x),
    )


def inv_mixture_cdf(
    y: ArrayLike, log_weights: ArrayLike, locs: ArrayLike, scales: ArrayLike
):
    """Analytic data -> base direction; returns ``(x, logdet)``."""
    return _mixture_cdf_value_and_logdet(y, log_weights, locs, scales)


mixture_cdf.definv_and_logdet(inv_mixture_cdf)
