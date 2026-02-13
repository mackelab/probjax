"""
Watson Distribution (:mod:`probjax.stats.watson`)
=================================================

This module contains the Watson distribution, a directional distribution on
the unit hypersphere. The Watson distribution is axial (invariant to sign)
and is parameterised by a mean direction and a concentration parameter.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.special import gammaln, hyp1f1

from probjax.stats.base import rv_exponential_family, rv_spherical
from probjax.stats.constraints import real, spherical
from probjax.stats.utils import normalize_sample_weights
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["watson"]

_EPS = 1e-8


def _normalize_vector(x: ArrayLike) -> Array:
    """Project vectors onto the unit sphere."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + _EPS)


def _orthogonal_unit_vector(mu: Array) -> Array:
    """Construct a unit vector orthogonal to `mu`."""
    dim = mu.shape[-1]
    basis = jnp.eye(dim, dtype=mu.dtype)
    idx = jnp.argmin(jnp.abs(mu))
    v = basis[idx]
    v = v - jnp.dot(v, mu) * mu
    norm_v = jnp.linalg.norm(v)
    return jnp.where(norm_v > _EPS, v / (norm_v + _EPS), basis[(idx + 1) % dim])


def _log_hyp1f1_half(kappa: Array, dim: int) -> Array:
    """Stable computation of log 1F1(1/2, dim/2, kappa)."""
    calc_dtype = jnp.result_type(jnp.asarray(kappa).dtype, jnp.float32)
    kappa_arr = jnp.asarray(kappa, dtype=calc_dtype)

    if dim <= 1:
        return kappa_arr.astype(calc_dtype)

    a = jnp.asarray(0.5, dtype=calc_dtype)
    b = jnp.asarray(0.5 * dim, dtype=calc_dtype)
    a_neg = b - a

    small_thresh = jnp.asarray(1e-3, dtype=calc_dtype)
    large_thresh = jnp.asarray(40.0, dtype=calc_dtype)

    pref_pos = gammaln(b) - gammaln(a)
    pref_neg = gammaln(b) - gammaln(a_neg)

    def series_log(z):
        term1 = (a / b) * z
        term2 = (
            (a * (a + jnp.asarray(1.0, dtype=calc_dtype)))
            / (b * (b + jnp.asarray(1.0, dtype=calc_dtype)))
            * (z**2)
            / jnp.asarray(2.0, dtype=calc_dtype)
        )
        term3 = (
            (a * (a + 1.0) * (a + 2.0))
            / (b * (b + 1.0) * (b + 2.0))
            * (z**3)
            / jnp.asarray(6.0, dtype=calc_dtype)
        )
        term4 = (
            (a * (a + 1.0) * (a + 2.0) * (a + 3.0))
            / (b * (b + 1.0) * (b + 2.0) * (b + 3.0))
            * (z**4)
            / jnp.asarray(24.0, dtype=calc_dtype)
        )
        return jnp.log1p(term1 + term2 + term3 + term4)

    def mid_log(z):
        val = hyp1f1(a, b, z)
        return jnp.log(jnp.maximum(val, jnp.asarray(_EPS, dtype=calc_dtype)))

    def pos_asympt(z_pos, a_param, pref):
        corr1 = (a_param * (b - a_param)) / (z_pos + _EPS)
        corr2 = (
            a_param
            * (a_param + jnp.asarray(1.0, dtype=calc_dtype))
            * (b - a_param)
            * (b - a_param + jnp.asarray(1.0, dtype=calc_dtype))
        ) / (jnp.asarray(2.0, dtype=calc_dtype) * (z_pos**2 + _EPS))
        corr = corr1 + corr2
        corr = jnp.clip(corr, -0.9, None)
        return pref + z_pos + (a_param - b) * jnp.log(z_pos) + jnp.log1p(corr)

    def pos_log(z):
        return pos_asympt(jnp.maximum(z, large_thresh), a, pref_pos)

    def neg_log(z):
        z_pos = jnp.maximum(-z, large_thresh)
        return z + pos_asympt(z_pos, a_neg, pref_neg)

    def scalar_eval(z):
        abs_z = jnp.abs(z)
        small = abs_z <= small_thresh
        large_pos = z >= large_thresh
        large_neg = z <= -large_thresh

        def handle_not_small(_):
            return lax.cond(
                large_pos,
                lambda __: pos_log(z),
                lambda __: lax.cond(
                    large_neg,
                    lambda ___: neg_log(z),
                    lambda ___: mid_log(z),
                    operand=None,
                ),
                operand=None,
            )

        return lax.cond(
            small,
            lambda __: series_log(z),
            handle_not_small,
            operand=None,
        )

    kappa_flat = kappa_arr.reshape(-1)
    log_flat = jax.vmap(scalar_eval)(kappa_flat)
    return log_flat.reshape(kappa_arr.shape)


def _log_normalization(kappa: Array, dim: int) -> Array:
    """Log of the normalisation constant of the Watson distribution."""
    dtype = jnp.result_type(jnp.asarray(kappa).dtype, jnp.float32)
    half_dim = jnp.asarray(0.5 * dim, dtype=dtype)
    log_uniform = (
        gammaln(half_dim)
        - jnp.log(jnp.asarray(2.0, dtype=dtype))
        - half_dim * jnp.log(jnp.asarray(jnp.pi, dtype=dtype))
    )
    log_hyp1f1 = _log_hyp1f1_half(kappa, dim)
    return log_uniform - log_hyp1f1


def _watson_moment_ratio(kappa, dim: int, dtype):
    """Return E[(μᵀX)²] for a Watson distribution with parameter κ."""
    calc_dtype = jnp.result_type(dtype, jnp.float32)
    kappa_arr = jnp.asarray(kappa, dtype=calc_dtype)

    abs_kappa = jnp.abs(kappa_arr)
    small_mask = abs_kappa < jnp.asarray(1e-4, dtype=calc_dtype)
    large_mask = abs_kappa > jnp.asarray(50.0, dtype=calc_dtype)
    mid_mask = ~(small_mask | large_mask)

    def series():
        k = kappa_arr
        dim_val = jnp.asarray(dim, dtype=calc_dtype)
        base = jnp.full_like(k, 1.0 / dim_val, dtype=calc_dtype)
        term2 = jnp.where(
            dim > 2,
            k / (dim_val * (dim_val + jnp.asarray(2.0, dtype=calc_dtype))),
            jnp.zeros_like(k),
        )
        term4 = jnp.where(
            dim > 4,
            (k**2)
            / (
                dim_val
                * (dim_val + jnp.asarray(2.0, dtype=calc_dtype))
                * (dim_val + jnp.asarray(4.0, dtype=calc_dtype))
            )
            * (dim_val + jnp.asarray(6.0, dtype=calc_dtype))
            / (dim_val + jnp.asarray(2.0, dtype=calc_dtype)),
            jnp.zeros_like(k),
        )
        approx = base + term2 + term4
        return approx

    def hyp1f1_ratio():
        k = kappa_arr.astype(calc_dtype)
        a = jnp.asarray(0.5, dtype=calc_dtype)
        b = jnp.asarray(0.5 * dim, dtype=calc_dtype)
        m0 = hyp1f1(a, b, k)
        m1 = hyp1f1(
            a + jnp.asarray(1.0, dtype=calc_dtype),
            b + jnp.asarray(1.0, dtype=calc_dtype),
            k,
        )
        ratio = (a / b) * m1 / m0
        return ratio

    def asymptotic():
        sign = jnp.sign(kappa_arr)
        mag = abs_kappa
        dim_val = jnp.asarray(dim, dtype=calc_dtype)
        ones = jnp.ones_like(kappa_arr, dtype=calc_dtype)
        iso = ones / dim_val
        approx = jnp.where(
            sign >= 0,
            ones
            - (dim_val - jnp.asarray(1.0, dtype=calc_dtype))
            / (jnp.asarray(2.0, dtype=calc_dtype) * mag)
            + (dim_val - jnp.asarray(1.0, dtype=calc_dtype))
            * (dim_val - jnp.asarray(3.0, dtype=calc_dtype))
            / (jnp.asarray(8.0, dtype=calc_dtype) * mag**2 + _EPS),
            iso,
        )
        return approx

    result_series = series()
    result_mid = hyp1f1_ratio()
    result_large = asymptotic()

    result = jnp.where(
        small_mask,
        result_series,
        jnp.where(large_mask, result_large, result_mid),
    )
    return result


def _solve_watson_kappa(target: ArrayLike, dim: int, dtype) -> jnp.ndarray:
    """Invert E[(μᵀX)²] = target for the Watson concentration parameter."""
    target = jnp.asarray(target, dtype=dtype)
    iso = jnp.asarray(1.0 / dim, dtype=dtype)
    eps = jnp.asarray(1e-12, dtype=dtype)
    upper = jnp.asarray(1.0, dtype=dtype) - eps
    target = jnp.clip(target, eps, upper)

    tol = jnp.asarray(1e-8, dtype=dtype)
    max_expand = jnp.int32(60)
    max_iter = jnp.int32(80)

    def close_branch(_):
        return jnp.asarray(0.0, dtype=dtype)

    def solve_branch(_):
        def positive_branch(_):
            lo = jnp.asarray(0.0, dtype=dtype)
            hi = jnp.asarray(1.0, dtype=dtype)
            ratio_hi = _watson_moment_ratio(hi, dim, dtype)

            def expand_cond(state):
                lo_, hi_, ratio_hi_, count_ = state
                return jnp.logical_and(ratio_hi_ <= target, count_ < max_expand)

            def expand_body(state):
                lo_, hi_, ratio_hi_, count_ = state
                hi_new = hi_ * jnp.asarray(2.0, dtype=dtype)
                ratio_hi_new = _watson_moment_ratio(hi_new, dim, dtype)
                return (lo_, hi_new, ratio_hi_new, count_ + 1)

            lo, hi, ratio_hi, _ = lax.while_loop(
                expand_cond, expand_body, (lo, hi, ratio_hi, jnp.int32(0))
            )
            ratio_lo = _watson_moment_ratio(lo, dim, dtype)

            def bisect_cond(state):
                lo_, hi_, ratio_lo_, ratio_hi_, count_ = state
                width = hi_ - lo_
                return jnp.logical_and(
                    count_ < max_iter,
                    width > tol * (jnp.asarray(1.0, dtype=dtype) + jnp.abs(hi_)),
                )

            def bisect_body(state):
                lo_, hi_, ratio_lo_, ratio_hi_, count_ = state
                mid = jnp.asarray(0.5, dtype=dtype) * (lo_ + hi_)
                ratio_mid = _watson_moment_ratio(mid, dim, dtype)
                choose_hi = ratio_mid > target
                hi_new = jnp.where(choose_hi, mid, hi_)
                lo_new = jnp.where(choose_hi, lo_, mid)
                ratio_hi_new = jnp.where(choose_hi, ratio_mid, ratio_hi_)
                ratio_lo_new = jnp.where(choose_hi, ratio_lo_, ratio_mid)
                return (lo_new, hi_new, ratio_lo_new, ratio_hi_new, count_ + 1)

            lo, hi, ratio_lo, ratio_hi, _ = lax.while_loop(
                bisect_cond,
                bisect_body,
                (lo, hi, ratio_lo, ratio_hi, jnp.int32(0)),
            )
            return jnp.asarray(0.5, dtype=dtype) * (lo + hi)

        def negative_branch(_):
            hi = jnp.asarray(0.0, dtype=dtype)
            lo = jnp.asarray(-1.0, dtype=dtype)
            ratio_lo = _watson_moment_ratio(lo, dim, dtype)

            limit = jnp.asarray(-1e6, dtype=dtype)

            def expand_cond(state):
                lo_, ratio_lo_, count_ = state
                cond_lo = lo_ > limit
                return jnp.logical_and(
                    jnp.logical_and(ratio_lo_ >= target, cond_lo),
                    count_ < max_expand,
                )

            def expand_body(state):
                lo_, ratio_lo_, count_ = state
                lo_new = lo_ * jnp.asarray(2.0, dtype=dtype)
                ratio_lo_new = _watson_moment_ratio(lo_new, dim, dtype)
                return (lo_new, ratio_lo_new, count_ + 1)

            lo, ratio_lo, _ = lax.while_loop(
                expand_cond, expand_body, (lo, ratio_lo, jnp.int32(0))
            )
            ratio_hi = _watson_moment_ratio(hi, dim, dtype)

            def bisect_cond(state):
                lo_, hi_, ratio_lo_, ratio_hi_, count_ = state
                width = hi_ - lo_
                return jnp.logical_and(
                    count_ < max_iter,
                    width > tol * (jnp.asarray(1.0, dtype=dtype) + jnp.abs(hi_)),
                )

            def bisect_body(state):
                lo_, hi_, ratio_lo_, ratio_hi_, count_ = state
                mid = jnp.asarray(0.5, dtype=dtype) * (lo_ + hi_)
                ratio_mid = _watson_moment_ratio(mid, dim, dtype)
                choose_hi = ratio_mid > target
                hi_new = jnp.where(choose_hi, mid, hi_)
                lo_new = jnp.where(choose_hi, lo_, mid)
                ratio_hi_new = jnp.where(choose_hi, ratio_mid, ratio_hi_)
                ratio_lo_new = jnp.where(choose_hi, ratio_lo_, ratio_mid)
                return (lo_new, hi_new, ratio_lo_new, ratio_hi_new, count_ + 1)

            lo, hi, ratio_lo, ratio_hi, _ = lax.while_loop(
                bisect_cond,
                bisect_body,
                (lo, hi, ratio_lo, ratio_hi, jnp.int32(0)),
            )
            return jnp.asarray(0.5, dtype=dtype) * (lo + hi)

        return lax.cond(target > iso, positive_branch, negative_branch, operand=None)

    return lax.cond(
        jnp.abs(target - iso) < tol,
        close_branch,
        solve_branch,
        operand=None,
    )


def _sample_watson_direction(
    key: RngKey,
    mu: Array,
    kappa: Array,
) -> Array:
    """Sample a single direction from the Watson distribution."""
    dim = mu.shape[-1]
    beta_a = jnp.array(0.5, dtype=mu.dtype)
    beta_b = jnp.array(0.5 * (dim - 1), dtype=mu.dtype)

    def cond_fn(state):
        done, *_ = state
        return jnp.logical_not(done)

    def body_fn(state):
        done, sample, key = state
        key, key_beta, key_u, key_sign, key_noise = random.split(key, 5)
        t = random.beta(key_beta, beta_a, beta_b)

        log_accept = jnp.where(kappa >= 0, kappa * (t - 1.0), kappa * t)
        u = random.uniform(key_u, dtype=mu.dtype)
        accept = jnp.log(u + _EPS) <= log_accept

        sign = jnp.where(random.bernoulli(key_sign, 0.5), 1.0, -1.0).astype(mu.dtype)
        w = sign * jnp.sqrt(jnp.clip(t, 0.0, 1.0))

        y = random.normal(key_noise, shape=(dim,), dtype=mu.dtype)
        v = y - jnp.dot(y, mu) * mu
        norm_v = jnp.linalg.norm(v)
        fallback = _orthogonal_unit_vector(mu)
        v = jnp.where(norm_v > _EPS, v / (norm_v + _EPS), fallback)

        candidate = w * mu + jnp.sqrt(jnp.maximum(0.0, 1.0 - t)) * v
        sample = jnp.where(accept, candidate, sample)
        return accept, sample, key

    state0 = (False, mu, key)
    _, sample, _ = lax.while_loop(cond_fn, body_fn, state0)
    return _normalize_vector(sample)


class watson_gen(rv_spherical, rv_exponential_family):
    """Watson distribution on the unit sphere.

    Parameters
    ----------
    mean_direction : array_like
        Unit vector representing the principal axis of the distribution.
    kappa : float, optional
        Concentration parameter. Positive values concentrate mass around ±mean,
        negative values concentrate around the orthogonal equator. Default is 0.
    """

    name = "watson"
    parameters = {"mean_direction": spherical, "kappa": real}
    multivariate = True

    @classmethod
    def support(cls, mean_direction=None, **kwargs):
        return spherical

    @classmethod
    def pdf(cls, x: Array, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        return jnp.exp(cls.logpdf(x, mean_direction, kappa, **kwargs))

    @classmethod
    def logpdf(cls, x: Array, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        x = _normalize_vector(jnp.asarray(x))
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa)

        dot_prod = jnp.sum(x * mean_direction, axis=-1)
        dot_prod = jnp.clip(dot_prod, -1.0, 1.0)
        dim = x.shape[-1]
        log_norm = _log_normalization(kappa, dim)
        return kappa * dot_prod**2 + log_norm

    @classmethod
    def _multivariate_batch_event_shape(
        cls,
        mean_direction: Array,
        kappa: Array = 0.0,
        **kwargs,
    ):
        """Infer batch/event shapes for frozen Watson distributions."""
        mean_direction_arr = jnp.asarray(mean_direction)
        if mean_direction_arr.ndim < 1:
            raise ValueError("mean_direction must be at least one-dimensional.")
        kappa_arr = jnp.asarray(kappa)
        batch_shape = jax.lax.broadcast_shapes(
            mean_direction_arr.shape[:-1], kappa_arr.shape
        )
        event_shape = (int(mean_direction_arr.shape[-1]),)
        return tuple(int(dim) for dim in batch_shape), event_shape

    @classmethod
    def rvs(
        cls,
        rng: RngKey,
        mean_direction: Array,
        kappa: Array = 0.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa)

        if mean_direction.ndim < 1:
            raise ValueError("mean_direction must be at least one-dimensional.")

        event_shape = mean_direction.shape[-1:]
        batch_shape = jax.lax.broadcast_shapes(
            mean_direction.shape[:-1], jnp.shape(kappa)
        )

        sample_shape = shape + batch_shape
        total = int(np.prod(sample_shape)) if sample_shape else 1

        keys = random.split(rng, total)

        mean_direction = jnp.broadcast_to(mean_direction, batch_shape + event_shape)
        kappa = jnp.broadcast_to(kappa, batch_shape)

        mean_tiled = jnp.broadcast_to(mean_direction, sample_shape + event_shape)
        kappa_tiled = jnp.broadcast_to(kappa, sample_shape)

        mean_flat = mean_tiled.reshape((total,) + event_shape)
        kappa_flat = kappa_tiled.reshape((total,))

        samples = jax.vmap(_sample_watson_direction)(keys, mean_flat, kappa_flat)
        return samples.reshape(sample_shape + event_shape)

    @classmethod
    def mean(cls, mean_direction: Array, **kwargs):
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        return jnp.zeros_like(mean_direction)

    @classmethod
    def mode(cls, mean_direction: Array, **kwargs):
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        return mean_direction

    @classmethod
    def mean_direction_vector(cls, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        """Representative mean direction (principal axis)."""
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa)
        batch_shape = jax.lax.broadcast_shapes(mean_direction.shape[:-1], kappa.shape)
        return jnp.broadcast_to(mean_direction, batch_shape + mean_direction.shape[-1:])

    @classmethod
    def log_partition(cls, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa)
        dim = mean_direction.shape[-1]
        log_norm = _log_normalization(kappa, dim)
        return -log_norm

    @classmethod
    def natural_parameters(cls, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa)
        batch_shape = jax.lax.broadcast_shapes(mean_direction.shape[:-1], kappa.shape)
        mean_direction = jnp.broadcast_to(
            mean_direction, batch_shape + mean_direction.shape[-1:]
        )
        kappa = jnp.broadcast_to(kappa, batch_shape)
        return kappa[..., None] * mean_direction

    @classmethod
    def sufficient_statistics(cls, x: Array, **kwargs):
        x = _normalize_vector(jnp.asarray(x))
        return jnp.square(x)

    @classmethod
    def mean_direction_dyad(cls, mean_direction: Array, kappa: Array = 0.0, **kwargs):
        """Expected dyadic product :math:`E[XX^T]` for the Watson distribution."""
        mean_direction = _normalize_vector(jnp.asarray(mean_direction))
        kappa = jnp.asarray(kappa, dtype=mean_direction.dtype)

        dim = mean_direction.shape[-1]
        if dim < 1:
            raise ValueError("mean_direction must have at least one dimension.")

        batch_shape = jax.lax.broadcast_shapes(mean_direction.shape[:-1], kappa.shape)
        mean_direction = jnp.broadcast_to(mean_direction, batch_shape + (dim,))
        kappa = jnp.broadcast_to(kappa, batch_shape)

        if dim == 1:
            return jnp.broadcast_to(
                jnp.ones((1, 1), dtype=mean_direction.dtype), batch_shape + (1, 1)
            )

        rho = jnp.asarray(
            _watson_moment_ratio(kappa, dim, mean_direction.dtype),
            dtype=mean_direction.dtype,
        )
        rho = jnp.broadcast_to(rho, batch_shape)

        mu_outer = jnp.einsum("...i,...j->...ij", mean_direction, mean_direction)
        identity = jnp.eye(dim, dtype=mean_direction.dtype)
        identity = jnp.broadcast_to(identity, batch_shape + (dim, dim))

        rho_term = rho[..., None, None] * mu_outer
        perp_scale = ((1.0 - rho) / jnp.asarray(dim - 1, dtype=mean_direction.dtype))[
            ..., None, None
        ]
        perp_term = perp_scale * (identity - mu_outer)
        return rho_term + perp_term

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Estimate mean direction via PCA and concentration via axial moment."""
        data = _normalize_vector(jnp.asarray(data))
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[0]
        dtype = data.dtype

        weights_arr = normalize_sample_weights(
            weights,
            n_samples=n,
            dtype=dtype,
            mismatch_message="weights must have the same number of rows as data",
        )
        if weights_arr is None:
            weights_arr = jnp.ones((n,), dtype=dtype) / jnp.asarray(n, dtype=dtype)

        scatter = (data * weights_arr[:, None]).T @ data
        scatter = 0.5 * (scatter + jnp.swapaxes(scatter, -1, -2))
        eigvals, eigvecs = jnp.linalg.eigh(scatter)
        dim = data.shape[-1]
        iso = jnp.asarray(1.0 / dim, dtype=dtype)

        idx_max = jnp.argmax(eigvals)
        idx_min = jnp.argmin(eigvals)
        rho_max = eigvals[idx_max]
        rho_min = eigvals[idx_min]

        mu_max = eigvecs[:, idx_max]
        mu_min = eigvecs[:, idx_min]
        choose_max = jnp.abs(rho_max - iso) >= jnp.abs(rho_min - iso)
        mu = jnp.where(choose_max, mu_max, mu_min)
        mu = _normalize_vector(mu)

        axial_moment = jnp.sum(weights_arr * jnp.square(data @ mu))
        kappa = _solve_watson_kappa(axial_moment, dim, dtype)
        return mu, kappa


watson = watson_gen(name="watson")
