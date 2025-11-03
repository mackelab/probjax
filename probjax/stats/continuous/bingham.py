"""
Bingham Distribution (:mod:`probjax.stats.bingham`)
===================================================

Improved Bingham distribution implementation featuring a deterministic
sphere-integration based normaliser and specialised rejection sampling for
three-dimensional directional data.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.numpy.linalg import eigh
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, PRNGKeyArray
import numpy as np

from probjax.stats.base import rv_continuous, rv_exponential_family
from probjax.stats.constraints import real, spherical, stiefel

__all__ = ["bingham"]

_EPS = 1e-8
_DEFAULT_SAMPLES = 512


@lru_cache(maxsize=None)
def _cached_sphere_samples(dim: int, num_samples: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(num_samples, dim))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return jnp.asarray(samples)


def _sphere_area(dim: int) -> jnp.ndarray:
    dim = jnp.asarray(dim, dtype=jnp.float32)
    return jnp.exp(jnp.log(2.0) + 0.5 * dim * jnp.log(jnp.pi) - gammaln(0.5 * dim))


def _build_parameter_matrix(orientation: Array, concentration: Array) -> Array:
    orientation = jnp.asarray(orientation)
    concentration = jnp.asarray(concentration)
    if orientation.ndim < 2:
        raise ValueError("orientation must be at least two-dimensional.")
    if concentration.ndim == 0:
        concentration = jnp.expand_dims(concentration, axis=-1)

    event_dim = orientation.shape[-1]
    if concentration.shape[-1] != event_dim:
        raise ValueError(
            "concentration vector must have the same length as the orientation dimension."
        )

    batch_shape = jax.lax.broadcast_shapes(
        orientation.shape[:-2], concentration.shape[:-1]
    )
    orientation = jnp.broadcast_to(orientation, batch_shape + orientation.shape[-2:])
    concentration = jnp.broadcast_to(
        concentration, batch_shape + concentration.shape[-1:]
    )

    eye = jnp.eye(event_dim, dtype=orientation.dtype)
    diag = concentration[..., :, None] * eye
    param = orientation @ diag @ jnp.swapaxes(orientation, -1, -2)
    return 0.5 * (param + jnp.swapaxes(param, -1, -2))


def deterministic_sphere_integration(
    kappa: Array,
    beta: Array,
    mu: Array,
    mu_beta: Array,
    n_theta: int = 200,
    n_phi: int = 200,
) -> Array:
    """Deterministic trapezoidal integration on S^2 for quadratic exponentials."""
    dtype = jnp.result_type(kappa, beta, mu, mu_beta)
    thetas = jnp.linspace(0.0, jnp.pi, n_theta, dtype=dtype)
    phis = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, dtype=dtype)
    Theta, Phi = jnp.meshgrid(thetas, phis, indexing="ij")

    sinT = jnp.sin(Theta)
    cosT = jnp.cos(Theta)
    cosP = jnp.cos(Phi)
    sinP = jnp.sin(Phi)

    nx = sinT * cosP
    ny = sinT * sinP
    nz = cosT

    mu = jnp.asarray(mu, dtype=dtype)
    mu_beta = jnp.asarray(mu_beta, dtype=dtype)
    kappa = jnp.asarray(kappa, dtype=dtype)[..., None, None]
    beta = jnp.asarray(beta, dtype=dtype)[..., None, None]

    dot_mu = (
        mu[..., 0][..., None, None] * nx
        + mu[..., 1][..., None, None] * ny
        + mu[..., 2][..., None, None] * nz
    )
    dot_mubeta = (
        mu_beta[..., 0][..., None, None] * nx
        + mu_beta[..., 1][..., None, None] * ny
        + mu_beta[..., 2][..., None, None] * nz
    )

    exponent = kappa * dot_mu**2 + beta * dot_mubeta**2
    integrand = jnp.exp(exponent) * sinT

    dtheta = jnp.pi / (n_theta - 1)
    dphi = 2.0 * jnp.pi / (n_phi - 1)
    integral_phi = _trapz(integrand, dx=dphi, axis=-1)
    integral = _trapz(integral_phi, dx=dtheta, axis=-1)
    return integral


def sample_sphere(key: PRNGKeyArray, n_samples: int, dtype=jnp.float32) -> Array:
    """Sample `n_samples` random unit vectors on S^2."""
    xyz = random.normal(key, shape=(n_samples, 3), dtype=dtype)
    norms = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    return xyz / (norms + _EPS)


def sample_quad_exp_distribution(
    key: PRNGKeyArray,
    kappa: Array,
    beta: Array,
    mu: Array,
    mu_beta: Array,
    max_iter: int = 256,
) -> Array:
    """Rejection sampler for exp(kappa (n·mu)^2 + beta (n·mu_beta)^2) on S^2."""
    dtype = jnp.result_type(kappa, beta, mu, mu_beta)
    kappa = jnp.asarray(kappa, dtype=dtype)
    beta = jnp.asarray(beta, dtype=dtype)
    mu = jnp.asarray(mu, dtype=dtype)
    mu_beta = jnp.asarray(mu_beta, dtype=dtype)

    max_val = jnp.maximum(jnp.maximum(kappa, beta), 0.0)

    def cond_fun(state):
        i, done, *_ = state
        return (i < max_iter) & ~done

    def body_fun(state):
        i, done, rng, sample_val = state
        rng, sk_samp, sk_acc = random.split(rng, 3)
        candidate = sample_sphere(sk_samp, 1, dtype=dtype)[0]
        dot_mu = jnp.dot(candidate, mu)
        dot_mubeta = jnp.dot(candidate, mu_beta)
        exponent = kappa * dot_mu**2 + beta * dot_mubeta**2
        accept_prob = jnp.exp(exponent - max_val)
        uniform = random.uniform(sk_acc, dtype=dtype)
        accepted = uniform < accept_prob
        new_sample = jnp.where(accepted, candidate, sample_val)
        return i + 1, done | accepted, rng, new_sample

    init = (0, False, key, mu)
    _, done, _, final_sample = lax.while_loop(cond_fun, body_fun, init)
    return final_sample


def _sort_axes(concentration: Array, orientation: Array) -> Tuple[Array, Array]:
    """Sort concentration/axes in descending order of concentration."""
    idx = jnp.flip(jnp.argsort(concentration, axis=-1), axis=-1)
    concentration_sorted = jnp.take_along_axis(concentration, idx, axis=-1)
    gather_idx = idx[..., None, :]
    orientation_sorted = jnp.take_along_axis(orientation, gather_idx, axis=-1)
    return concentration_sorted, orientation_sorted


def _bingham_s3_parameters(orientation: Array, concentration: Array):
    """Extract quadratic form parameters for the S^2-specific routines."""
    concentration_sorted, orientation_sorted = _sort_axes(concentration, orientation)
    base = concentration_sorted[..., 2]
    kappa = concentration_sorted[..., 0] - base
    beta = concentration_sorted[..., 1] - base
    mu = orientation_sorted[..., :, 0]
    mu_beta = orientation_sorted[..., :, 1]
    return kappa, beta, mu, mu_beta, base


def _approximate_log_partition_mc(parameter_matrix: Array) -> Array:
    dim = parameter_matrix.shape[-1]
    samples = _cached_sphere_samples(dim, _DEFAULT_SAMPLES)
    quad = jnp.einsum("bi,...ij,bj->...b", samples, parameter_matrix, samples)
    logmeanexp = jax.nn.logsumexp(quad, axis=-1) - jnp.log(samples.shape[0])
    log_area = jnp.log(_sphere_area(dim))
    return log_area + logmeanexp


def _sample_uniform_sphere(key: PRNGKeyArray, dim: int) -> Array:
    vec = random.normal(key, shape=(dim,))
    return vec / (jnp.linalg.norm(vec) + _EPS)


def _sample_bingham_direction_mc(
    key: PRNGKeyArray,
    parameter_matrix: Array,
    lambda_max: Array,
) -> Array:
    dim = parameter_matrix.shape[-1]

    def cond_fn(state):
        done, *_ = state
        return jnp.logical_not(done)

    def body_fn(state):
        done, sample, key = state
        key, key_vec, key_u = random.split(key, 3)
        candidate = _sample_uniform_sphere(key_vec, dim)
        quad = jnp.dot(candidate, parameter_matrix @ candidate)
        log_u = jnp.log(random.uniform(key_u) + _EPS)
        accept = log_u <= quad - lambda_max
        sample = jnp.where(accept, candidate, sample)
        return accept, sample, key

    state0 = (False, jnp.zeros((dim,), dtype=parameter_matrix.dtype), key)
    _, sample, _ = lax.while_loop(cond_fn, body_fn, state0)
    return jnp.where(
        jnp.linalg.norm(sample) < _EPS, _sample_uniform_sphere(key, dim), sample
    )


class bingham_gen(rv_continuous, rv_exponential_family):
    """Bingham distribution on the unit sphere."""

    name = "bingham"
    parameters = {"orientation": stiefel, "concentration": real}
    multivariate = True

    @classmethod
    def support(cls, orientation=None, **kwargs):
        return spherical

    @classmethod
    def pdf(cls, x: Array, orientation: Array, concentration: Array, **kwargs):
        return jnp.exp(cls.logpdf(x, orientation, concentration, **kwargs))

    @classmethod
    def logpdf(cls, x: Array, orientation: Array, concentration: Array, **kwargs):
        x = jnp.asarray(x)
        parameter_matrix = _build_parameter_matrix(orientation, concentration)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + _EPS)
        quad = jnp.einsum("...i,...ij,...j->...", x, parameter_matrix, x)
        log_partition = cls.log_partition(orientation, concentration, **kwargs)
        return quad - log_partition

    def freeze(
        self,
        orientation: Array,
        concentration: Array,
        **kwargs,
    ):
        """Freeze parameters while recording batch and event shapes."""
        rv = super().freeze(orientation=orientation, concentration=concentration, **kwargs)
        orientation_arr = jnp.asarray(orientation)
        if orientation_arr.ndim < 2:
            raise ValueError("orientation must be at least two-dimensional.")
        concentration_arr = jnp.asarray(concentration)
        conc_batch = (
            concentration_arr.shape[:-1] if concentration_arr.ndim > 0 else ()
        )
        batch_shape = jax.lax.broadcast_shapes(orientation_arr.shape[:-2], conc_batch)
        rv._batch_shape = batch_shape
        rv._event_shape = orientation_arr.shape[-1:]
        return rv

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        orientation: Array,
        concentration: Array,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        orientation = jnp.asarray(orientation)
        concentration = jnp.asarray(concentration)
        event_dim = orientation.shape[-1]

        if event_dim == 3:
            batch_shape = jax.lax.broadcast_shapes(
                orientation.shape[:-2], concentration.shape[:-1]
            )
            orientation = jnp.broadcast_to(orientation, batch_shape + (3, 3))
            concentration = jnp.broadcast_to(concentration, batch_shape + (3,))

            kappa, beta, mu, mu_beta, _ = _bingham_s3_parameters(
                orientation, concentration
            )

            sample_shape = shape + batch_shape
            total = int(np.prod(sample_shape)) if sample_shape else 1
            keys = random.split(rng, total)

            kappa_tiled = jnp.broadcast_to(kappa, sample_shape)
            beta_tiled = jnp.broadcast_to(beta, sample_shape)
            mu_tiled = jnp.broadcast_to(mu, sample_shape + (3,))
            mu_beta_tiled = jnp.broadcast_to(mu_beta, sample_shape + (3,))

            samples = jax.vmap(sample_quad_exp_distribution)(
                keys,
                kappa_tiled.reshape(total),
                beta_tiled.reshape(total),
                mu_tiled.reshape((total, 3)),
                mu_beta_tiled.reshape((total, 3)),
            )
            return samples.reshape(sample_shape + (3,))

        parameter_matrix = _build_parameter_matrix(orientation, concentration)
        eigenvalues, _ = eigh(parameter_matrix)
        lambda_max = jnp.max(eigenvalues, axis=-1)

        event_shape = parameter_matrix.shape[-1:]
        batch_shape = parameter_matrix.shape[:-2]
        sample_shape = shape + batch_shape
        total = int(np.prod(sample_shape)) if sample_shape else 1

        keys = random.split(rng, total)

        params_tiled = jnp.broadcast_to(
            parameter_matrix, sample_shape + event_shape + event_shape
        )
        lambda_tiled = jnp.broadcast_to(lambda_max, sample_shape)

        params_flat = params_tiled.reshape((total,) + event_shape + event_shape)
        lambda_flat = lambda_tiled.reshape((total,))

        samples = jax.vmap(_sample_bingham_direction_mc)(keys, params_flat, lambda_flat)
        return samples.reshape(sample_shape + event_shape)

    @classmethod
    def mean(cls, orientation: Array, concentration: Array, **kwargs):
        orientation = jnp.asarray(orientation)
        concentration = jnp.asarray(concentration)
        event_dim = orientation.shape[-1]
        batch_shape = jax.lax.broadcast_shapes(
            orientation.shape[:-2], concentration.shape[:-1]
        )
        return jnp.zeros(batch_shape + (event_dim,), dtype=orientation.dtype)

    @classmethod
    def log_partition(cls, orientation: Array, concentration: Array, **kwargs):
        orientation = jnp.asarray(orientation)
        concentration = jnp.asarray(concentration)
        event_dim = orientation.shape[-1]

        if event_dim == 3:
            batch_shape = jax.lax.broadcast_shapes(
                orientation.shape[:-2], concentration.shape[:-1]
            )
            orientation = jnp.broadcast_to(orientation, batch_shape + (3, 3))
            concentration = jnp.broadcast_to(concentration, batch_shape + (3,))
            kappa, beta, mu, mu_beta, base = _bingham_s3_parameters(
                orientation, concentration
            )
            n_theta = kwargs.pop("n_theta", 200)
            n_phi = kwargs.pop("n_phi", 200)
            integral = deterministic_sphere_integration(
                kappa, beta, mu, mu_beta, n_theta=n_theta, n_phi=n_phi
            )
            return jnp.log(integral) + base

        parameter_matrix = _build_parameter_matrix(orientation, concentration)
        return _approximate_log_partition_mc(parameter_matrix)

    @classmethod
    def natural_parameters(cls, orientation: Array, concentration: Array, **kwargs):
        return _build_parameter_matrix(orientation, concentration)

    @classmethod
    def sufficient_statistics(cls, x: Array, **kwargs):
        x = jnp.asarray(x)
        return jnp.einsum("...i,...j->...ij", x, x)

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Estimate parameters from the sample scatter matrix."""
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[0]
        dtype = data.dtype

        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((n,))
            if weights.shape[0] != n:
                raise ValueError("weights must have the same number of rows as data")
            weights = jnp.clip(weights, 0)
            total = jnp.sum(weights)
            total = jnp.where(total > 0, total, jnp.asarray(n, dtype=dtype))
            weights = weights / total
            scatter = (data * weights[:, None]).T @ data
        else:
            scatter = data.T @ data / n
        eigvals, eigvecs = jnp.linalg.eigh(scatter)
        idx = jnp.flip(jnp.argsort(eigvals))
        eigvals = eigvals[idx]
        orientation = eigvecs[:, idx]
        concentration = eigvals - jnp.mean(eigvals)
        return orientation, concentration


bingham = bingham_gen(name="bingham")
def _trapz(y: Array, dx: Array, axis: int = -1) -> Array:
    """Simple trapezoidal integration along a given axis."""
    sum_y = jnp.sum(y, axis=axis)
    edge0 = jnp.take(y, 0, axis=axis)
    edge1 = jnp.take(y, -1, axis=axis)
    return dx * (sum_y - 0.5 * (edge0 + edge1))
