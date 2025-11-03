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
from jax import lax, random
from jax.scipy.special import gammaln, hyp1f1
from jaxtyping import Array, ArrayLike, PRNGKeyArray
import numpy as np

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, spherical

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


def _log_normalization(kappa: Array, dim: int) -> Array:
    """Log of the normalisation constant of the Watson distribution."""
    half_dim = 0.5 * dim
    log_uniform = gammaln(half_dim) - jnp.log(2.0) - half_dim * jnp.log(jnp.pi)
    log_hyp1f1 = jnp.log(hyp1f1(0.5, half_dim, kappa))
    return log_uniform - log_hyp1f1


def _watson_moment_ratio(kappa: float, dim: int, dtype) -> float:
    """Return E[(μᵀX)²] for a Watson distribution with parameter κ."""
    calc_dtype = jnp.result_type(dtype, jnp.float32)
    kappa_arr = jnp.asarray(kappa, dtype=calc_dtype)
    a = jnp.asarray(0.5, dtype=calc_dtype)
    b = jnp.asarray(0.5 * dim, dtype=calc_dtype)
    m0 = hyp1f1(a, b, kappa_arr)
    m1 = hyp1f1(a + 1.0, b + 1.0, kappa_arr)
    ratio = (a / b) * m1 / m0
    return float(ratio)


def _solve_watson_kappa(target: float, dim: int, dtype) -> jnp.ndarray:
    """Invert E[(μᵀX)²] = target for the Watson concentration parameter."""
    if not np.isfinite(target):
        target = 1.0 / dim

    # Constrain to the open interval (0, 1) for numerical stability.
    eps = 1e-12
    target = float(np.clip(target, eps, 1.0 - eps))
    iso = 1.0 / dim

    if abs(target - iso) < 1e-8:
        return jnp.asarray(0.0, dtype=dtype)

    tol = 1e-8
    max_iter = 80

    if target > iso:
        lo, hi = 0.0, 1.0
        ratio_hi = _watson_moment_ratio(hi, dim, dtype)
        while ratio_hi <= target and hi < 1e6:
            hi *= 2.0
            ratio_hi = _watson_moment_ratio(hi, dim, dtype)
        if ratio_hi <= target:
            return jnp.asarray(hi, dtype=dtype)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            ratio_mid = _watson_moment_ratio(mid, dim, dtype)
            if abs(ratio_mid - target) <= tol:
                return jnp.asarray(mid, dtype=dtype)
            if ratio_mid < target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tol * (1.0 + abs(mid)):
                break
        kappa = 0.5 * (lo + hi)
        return jnp.asarray(kappa, dtype=dtype)

    hi, lo = 0.0, -1.0
    ratio_lo = _watson_moment_ratio(lo, dim, dtype)
    while ratio_lo >= target and lo > -1e6:
        lo *= 2.0
        ratio_lo = _watson_moment_ratio(lo, dim, dtype)
    if ratio_lo >= target:
        return jnp.asarray(lo, dtype=dtype)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ratio_mid = _watson_moment_ratio(mid, dim, dtype)
        if abs(ratio_mid - target) <= tol:
            return jnp.asarray(mid, dtype=dtype)
        if ratio_mid > target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * (1.0 + abs(mid)):
            break
    kappa = 0.5 * (lo + hi)
    return jnp.asarray(kappa, dtype=dtype)


def _sample_watson_direction(
    key: PRNGKeyArray,
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


class watson_gen(rv_continuous, rv_exponential_family):
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
        dim = x.shape[-1]
        log_norm = _log_normalization(kappa, dim)
        return kappa * dot_prod**2 + log_norm

    def freeze(
        self,
        mean_direction: Array,
        kappa: Array = 0.0,
        **kwargs,
    ):
        """Freeze parameters while recording batch and event shapes."""
        rv = super().freeze(mean_direction=mean_direction, kappa=kappa, **kwargs)
        mean_direction_arr = jnp.asarray(mean_direction)
        if mean_direction_arr.ndim < 1:
            raise ValueError("mean_direction must be at least one-dimensional.")
        kappa_arr = jnp.asarray(kappa)
        batch_shape = jax.lax.broadcast_shapes(
            mean_direction_arr.shape[:-1], kappa_arr.shape
        )
        rv._batch_shape = batch_shape
        rv._event_shape = mean_direction_arr.shape[-1:]
        return rv

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
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

        mean_direction = jnp.broadcast_to(
            mean_direction, batch_shape + event_shape
        )
        kappa = jnp.broadcast_to(kappa, batch_shape)

        mean_tiled = jnp.broadcast_to(
            mean_direction, sample_shape + event_shape
        )
        kappa_tiled = jnp.broadcast_to(kappa, sample_shape)

        mean_flat = mean_tiled.reshape((total,) + event_shape)
        kappa_flat = kappa_tiled.reshape((total,))

        samples = jax.vmap(_sample_watson_direction)(
            keys, mean_flat, kappa_flat
        )
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
        batch_shape = jax.lax.broadcast_shapes(
            mean_direction.shape[:-1], kappa.shape
        )
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

        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((n,))
            if weights.shape[0] != n:
                raise ValueError("weights must have the same number of rows as data")
            weights = jnp.clip(weights, 0)
        else:
            weights = jnp.ones((n,), dtype=dtype)

        total_weight = jnp.sum(weights)
        total_weight = jnp.where(
            total_weight > 0, total_weight, jnp.asarray(n, dtype=dtype)
        )
        weights = weights / total_weight

        scatter = (data * weights[:, None]).T @ data
        scatter = 0.5 * (scatter + jnp.swapaxes(scatter, -1, -2))
        eigvals, eigvecs = jnp.linalg.eigh(scatter)
        dim = data.shape[-1]
        iso = 1.0 / dim

        eigvals_np = np.array(eigvals)
        idx_max = int(eigvals_np.argmax())
        idx_min = int(eigvals_np.argmin())

        rho_max = float(eigvals_np[idx_max])
        rho_min = float(eigvals_np[idx_min])

        if abs(rho_max - iso) >= abs(rho_min - iso):
            mu = eigvecs[:, idx_max]
        else:
            mu = eigvecs[:, idx_min]

        mu = _normalize_vector(mu)
        axial_moment = float(jnp.sum(weights * jnp.square(data @ mu)))
        kappa = _solve_watson_kappa(axial_moment, dim, dtype)
        return mu, kappa


watson = watson_gen(name="watson")
