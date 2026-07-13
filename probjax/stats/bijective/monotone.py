"""Monotone neural bijectors (params-first, elementwise).

Each bijector here is an analytic, strictly increasing map ``G(params, t)``
used in the *density* direction (data -> base): its value and derivative are
closed-form and differentiable, so ``logpdf`` and training gradients never
touch a numerical solver. The bijector's *forward* (base -> data, used for
sampling) solves ``G(params, y) = x`` with a bracketed bisection and is
registered via :class:`~probjax.core.custom_inverse`:

    forward:  y = bijector(params, x)   (root solve, do not differentiate)
    inverse:  x = G(params, y),  logdet = log G'(params, y)   (analytic)

Conventions match :mod:`probjax.stats.bijective.custom_inverses`: ``params``
has trailing size ``bijector_dim`` and broadcasts against scalar-per-element
``x`` (blocks combine via ``x[..., None]`` and reduce over the last axis).

Implemented families (all zero-init to a well-conditioned near-identity map):

* :func:`deep_sigmoid_bijector` — NAF's deep sigmoidal flow
  (Huang et al., 2018). ``bijector_dim = 3 * num_components``.
* :func:`unconstrained_monotone_bijector` — UMNN-style monotone network,
  integrated with fixed Gauss-Legendre quadrature (Wehenkel & Louppe, 2019).
  ``bijector_dim = 3 * num_hidden + 2``.
* :func:`sos_polynomial_bijector` — sum-of-squares polynomial flow
  (Jaini et al., 2019). ``bijector_dim = num_polys * (degree + 1) + 1``.
* :func:`bernstein_bijector` — monotone Bernstein polynomial on an interval
  with identity tails (Sick et al., 2021). ``bijector_dim = degree``.
* :func:`mixture_cdf_bijector` — Gaussianization-flow kernel layer:
  logistic-mixture CDF followed by the standard normal quantile
  (Meng et al., 2020). ``bijector_dim = 3 * num_components``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from probjax.core import custom_inverse

__all__ = [
    "deep_sigmoid_bijector",
    "unconstrained_monotone_bijector",
    "sos_polynomial_bijector",
    "bernstein_bijector",
    "mixture_cdf_bijector",
]

_EPS = 1e-6


def _solve_increasing(g, target, num_expansions: int = 40, num_bisections: int = 60):
    """Solve ``g(t) = target`` elementwise for strictly increasing ``g``.

    Doubles a symmetric bracket around 0 until it contains the root
    (covers |t| up to ~2^40), then bisects. Not differentiable — callers
    differentiate through the analytic inverse instead.
    """
    target = jnp.asarray(target)
    lo = jnp.full_like(target, -1.0)
    hi = jnp.full_like(target, 1.0)

    def expand(i, carry):
        lo, hi = carry
        width = 2.0**i
        lo = jnp.where(g(lo) > target, lo - width, lo)
        hi = jnp.where(g(hi) < target, hi + width, hi)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, num_expansions, expand, (lo, hi))

    def bisect(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        go_left = g(mid) > target
        return jnp.where(go_left, lo, mid), jnp.where(go_left, mid, hi)

    lo, hi = jax.lax.fori_loop(0, num_bisections, bisect, (lo, hi))
    return 0.5 * (lo + hi)


def _blocks(params, sizes):
    """Split trailing axis of ``params`` into blocks of the given sizes."""
    params = jnp.asarray(params)
    out = []
    start = 0
    for size in sizes:
        out.append(params[..., start : start + size])
        start += size
    return out


# ---------------------------------------------------------------------------
# NAF — deep sigmoidal flow
# ---------------------------------------------------------------------------


def _dsf_value_and_logdet(params, t):
    k = params.shape[-1] // 3
    w_logits, raw_a, b = _blocks(params, (k, k, k))
    log_w = jax.nn.log_softmax(w_logits, axis=-1)
    a = jax.nn.softplus(raw_a) + _EPS

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


@partial(custom_inverse, inv_argnum=1)
def deep_sigmoid_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    """NAF deep-sigmoidal transform (sampling direction; root solve)."""
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    return _solve_increasing(lambda t: _dsf_value_and_logdet(params, t)[0], x)


deep_sigmoid_bijector.definv_and_logdet(
    lambda params, y, **kwargs: _dsf_value_and_logdet(
        jnp.asarray(params), jnp.asarray(y)
    )
)


# ---------------------------------------------------------------------------
# UNAF — unconstrained monotone (neural-integrand) flow
# ---------------------------------------------------------------------------

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_GL_NODES = jnp.asarray(_GL_NODES)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS)
_MIN_INTEGRAND = 1e-2


def _umnn_integrand(params, t):
    """Positive scalar integrand: one-hidden-layer net emitted by the conditioner."""
    k = (params.shape[-1] - 2) // 3
    w1, b1, w2, b2, _ = _blocks(params, (k, k, k, 1, 1))
    hidden = jnp.tanh(w1 * t[..., None] + b1)
    raw = jnp.sum(w2 * hidden, axis=-1) + b2[..., 0]
    return jax.nn.softplus(raw) + _MIN_INTEGRAND


def _umnn_value_and_logdet(params, t):
    offset = params[..., -1]

    # int_0^t integrand via Gauss-Legendre on [0, t]
    half_t = 0.5 * t
    nodes = half_t[..., None] * (_GL_NODES + 1.0)  # (..., 32)
    vals = _umnn_integrand(params[..., None, :], nodes)
    integral = half_t * jnp.sum(_GL_WEIGHTS * vals, axis=-1)

    value = integral + offset
    log_grad = jnp.log(_umnn_integrand(params, t))
    return value, log_grad


@partial(custom_inverse, inv_argnum=1)
def unconstrained_monotone_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    """UMNN transform (sampling direction; root solve)."""
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    return _solve_increasing(lambda t: _umnn_value_and_logdet(params, t)[0], x)


unconstrained_monotone_bijector.definv_and_logdet(
    lambda params, y, **kwargs: _umnn_value_and_logdet(
        jnp.asarray(params), jnp.asarray(y)
    )
)


# ---------------------------------------------------------------------------
# SOS — sum-of-squares polynomial flow
# ---------------------------------------------------------------------------

_SOS_DEGREE = 3  # per-polynomial degree r; bijector_dim = K*(r+1) + 1


def _sos_value_and_logdet(params, t):
    coeff_flat, c = params[..., :-1], params[..., -1]
    k = coeff_flat.shape[-1] // (_SOS_DEGREE + 1)
    coeffs = jnp.reshape(
        coeff_flat, coeff_flat.shape[:-1] + (k, _SOS_DEGREE + 1)
    )  # (..., K, r+1)

    # poly_k(t) = sum_l a_{kl} t^l ;  G'(t) ∝ eps + sum_k poly_k(t)^2
    powers = t[..., None] ** jnp.arange(_SOS_DEGREE + 1)  # (..., r+1)
    poly_vals = jnp.sum(coeffs * powers[..., None, :], axis=-1)  # (..., K)
    sq_sum = jnp.sum(poly_vals**2, axis=-1)

    # Normalize by the derivative at zero-params so that zero conditioner
    # output yields the identity map (slope 1) while still allowing
    # contraction and expansion as params move.
    norm = _EPS + jnp.sum(coeffs[..., 0] ** 2, axis=-1)

    # Antiderivative of sum_k poly_k^2: squared-poly coefficients via
    # anti-diagonal sums of the coefficient outer product.
    outer = coeffs[..., :, None] * coeffs[..., None, :]  # (..., K, r+1, r+1)
    degree2 = 2 * _SOS_DEGREE
    t_pows = t[..., None] ** jnp.arange(1, degree2 + 2)  # t^{j+1}
    integral = jnp.zeros_like(t)
    for j in range(degree2 + 1):
        b_j = jnp.zeros(t.shape) if t.ndim else jnp.asarray(0.0)
        for l in range(max(0, j - _SOS_DEGREE), min(j, _SOS_DEGREE) + 1):
            b_j = b_j + jnp.sum(outer[..., l, j - l], axis=-1)
        integral = integral + b_j * t_pows[..., j] / (j + 1)

    value = c + (_EPS * t + integral) / norm
    log_grad = jnp.log(_EPS + sq_sum) - jnp.log(norm)
    return value, log_grad


@partial(custom_inverse, inv_argnum=1)
def sos_polynomial_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    """Sum-of-squares polynomial transform (sampling direction; root solve)."""
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    return _solve_increasing(lambda t: _sos_value_and_logdet(params, t)[0], x)


sos_polynomial_bijector.definv_and_logdet(
    lambda params, y, **kwargs: _sos_value_and_logdet(
        jnp.asarray(params), jnp.asarray(y)
    )
)


# ---------------------------------------------------------------------------
# BPF — Bernstein polynomial flow
# ---------------------------------------------------------------------------

_BERNSTEIN_BOUND = 5.0


def _binom(n):
    from math import comb

    return jnp.asarray([comb(n, i) for i in range(n + 1)], dtype=jnp.float32)


def _bernstein_value_and_logdet(params, t, bound=_BERNSTEIN_BOUND):
    m = params.shape[-1]  # degree; m weights -> m+1 increasing coefficients
    weights = jax.nn.softmax(params, axis=-1)
    # theta_0 = 0, theta_m = 1, strictly increasing
    theta = jnp.concatenate(
        [jnp.zeros(params.shape[:-1] + (1,)), jnp.cumsum(weights, axis=-1)], axis=-1
    )

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


@partial(custom_inverse, inv_argnum=1)
def bernstein_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    """Monotone Bernstein-polynomial transform (sampling direction; root solve)."""
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    return _solve_increasing(lambda t: _bernstein_value_and_logdet(params, t)[0], x)


bernstein_bijector.definv_and_logdet(
    lambda params, y, **kwargs: _bernstein_value_and_logdet(
        jnp.asarray(params), jnp.asarray(y)
    )
)


# ---------------------------------------------------------------------------
# GF — logistic-mixture CDF + logistic quantile (Gaussianization kernel layer)
# ---------------------------------------------------------------------------
#
# The quantile is the logistic (logit) rather than the probit of the original
# paper: logit(mixture-CDF) has exactly linear tails, so the map is a
# numerically stable bijection of the whole real line (the probit version
# saturates in float32 beyond |x| ~ 4.8). The two differ only by the fixed
# smooth reparameterization probit∘logistic.


def _mixture_cdf_value_and_logdet(params, t):
    k = params.shape[-1] // 3
    w_logits, mu, raw_s = _blocks(params, (k, k, k))
    log_w = jax.nn.log_softmax(w_logits, axis=-1)
    s = jax.nn.softplus(raw_s) + _EPS

    z = (t[..., None] - mu) / s
    log_cdf = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(z), axis=-1)
    log_1mcdf = jax.nn.logsumexp(log_w + jax.nn.log_sigmoid(-z), axis=-1)
    value = log_cdf - log_1mcdf  # logit(mixture cdf)

    log_pdf_mix = jax.nn.logsumexp(
        log_w + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z) - jnp.log(s), axis=-1
    )
    log_grad = log_pdf_mix - log_cdf - log_1mcdf
    return value, log_grad


@partial(custom_inverse, inv_argnum=1)
def mixture_cdf_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    """Gaussianization kernel transform (sampling direction; root solve)."""
    del kwargs
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    return _solve_increasing(lambda t: _mixture_cdf_value_and_logdet(params, t)[0], x)


mixture_cdf_bijector.definv_and_logdet(
    lambda params, y, **kwargs: _mixture_cdf_value_and_logdet(
        jnp.asarray(params), jnp.asarray(y)
    )
)
