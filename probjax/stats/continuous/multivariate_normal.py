from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax import core as jax_core

from probjax.stats.base import rv_multivariate
from probjax.stats.constraints import real, symmetric_positive_definite_matrix
from probjax.stats.utils import normalize_sample_weights, row_mean_and_cov
from probjax.utils.linalg import batch_mahalanobis
from probjax.utils.typing import Array, ArrayLike, RngKey


def _asarray_optional(value: Optional[Array]) -> Optional[Array]:
    if value is None:
        return None
    return jnp.asarray(value)


def _ensure_array(name: str, value: Optional[Array]) -> Array:
    if value is None:
        raise ValueError(
            f"Parameter '{name}' must be provided for multivariate normal distribution."
        )
    return jnp.asarray(value)


def _resolve_scale_tril(
    loc: Array,
    cov: Optional[Array],
    precision_matrix: Optional[Array],
    scale_tril: Optional[Array],
) -> Tuple[Array, Array, Tuple[int, ...]]:
    loc_arr = jnp.asarray(loc)
    if loc_arr.ndim < 1:
        raise ValueError("loc must be at least one-dimensional.")

    if scale_tril is not None:
        scale_arr = jnp.asarray(scale_tril)
        if scale_arr.ndim < 2:
            raise ValueError("scale_tril must be at least two-dimensional.")
        batch_shape = jnp.broadcast_shapes(scale_arr.shape[:-2], loc_arr.shape[:-1])
        scale_arr = jnp.broadcast_to(scale_arr, batch_shape + scale_arr.shape[-2:])
        return loc_arr, scale_arr, tuple(int(s) for s in batch_shape)

    if cov is not None:
        cov_arr = jnp.asarray(cov)
        if cov_arr.ndim < 2:
            raise ValueError("covariance matrix must be at least two-dimensional.")
        batch_shape = jnp.broadcast_shapes(cov_arr.shape[:-2], loc_arr.shape[:-1])
        cov_arr = jnp.broadcast_to(cov_arr, batch_shape + cov_arr.shape[-2:])
        scale_arr = jnp.linalg.cholesky(cov_arr)
        return loc_arr, scale_arr, tuple(int(s) for s in batch_shape)

    if precision_matrix is not None:
        precision_arr = jnp.asarray(precision_matrix)
        if precision_arr.ndim < 2:
            raise ValueError("precision_matrix must be at least two-dimensional.")
        batch_shape = jnp.broadcast_shapes(precision_arr.shape[:-2], loc_arr.shape[:-1])
        precision_arr = jnp.broadcast_to(
            precision_arr,
            batch_shape + precision_arr.shape[-2:],
        )
        cov_arr = jnp.linalg.inv(precision_arr)
        scale_arr = jnp.linalg.cholesky(cov_arr)
        return loc_arr, scale_arr, tuple(int(s) for s in batch_shape)

    raise ValueError(
        "At least one of cov, precision_matrix, or scale_tril must be specified."
    )


class multivariate_normal_gen(rv_multivariate):
    """Multivariate normal (also called Gaussian) distribution parameterized by
    a mean vector and a covariance matrix.

    Parameters
    ----------
    loc : array_like
        Mean of the distribution (often referred to as mu)
    cov : array_like, optional
        Covariance matrix
    precision_matrix : array_like, optional
        Precision matrix
    scale_tril : array_like, optional
        Lower triangular matrix with positive diagonal
    """

    name = "multivariate_normal"
    parameters = {
        "loc": real,
        "cov": symmetric_positive_definite_matrix,
        "precision_matrix": symmetric_positive_definite_matrix,
        "scale_tril": symmetric_positive_definite_matrix,
    }
    parameter_aliases = {"covariance_matrix": "cov"}
    multivariate = True

    @classmethod
    def support(cls, loc=None, **kwargs):
        """Support of the multivariate normal distribution.

        Parameters
        ----------
        loc : array_like, optional
            Mean of the distribution. Default is None.

        Returns
        -------
        support : constraint
            Support of the distribution
        """
        return real

    @classmethod
    def pdf(
        cls,
        x: Array,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        **kwargs,
    ):
        """Probability density function of the multivariate normal distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function
        loc : array_like
            Mean of the distribution
        cov : array_like, optional
            Covariance matrix
        precision_matrix : array_like, optional
            Precision matrix
        scale_tril : array_like, optional
            Lower triangular matrix with positive diagonal

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
        return jnp.exp(
            cls.logpdf(
                x,
                loc=loc,
                cov=cov,
                precision_matrix=precision_matrix,
                scale_tril=scale_tril,
                **kwargs,
            )
        )

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the multivariate normal distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        loc : array_like
            Mean of the distribution
        cov : array_like, optional
            Covariance matrix
        precision_matrix : array_like, optional
            Precision matrix
        scale_tril : array_like, optional
            Lower triangular matrix with positive diagonal
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().

        Returns
        -------
        rvs : ndarray
            Random variates of given shape
        """
        loc_arr, scale_arr, batch_shape = _resolve_scale_tril(
            loc, cov, precision_matrix, scale_tril
        )

        event_dim = int(loc_arr.shape[-1])
        sample_shape = shape + batch_shape + (event_dim,)
        eps = random.normal(rng, shape=sample_shape, dtype=loc_arr.dtype)

        loc_reshaped = loc_arr.reshape((1,) * len(shape) + loc_arr.shape)
        loc_broadcast = jnp.broadcast_to(loc_reshaped, shape + loc_arr.shape)

        scale_reshaped = scale_arr.reshape((1,) * len(shape) + scale_arr.shape)
        scale_broadcast = jnp.broadcast_to(scale_reshaped, shape + scale_arr.shape)

        transformed = jnp.einsum("...ij,...j->...i", scale_broadcast, eps)
        return loc_broadcast + transformed

    @classmethod
    def _multivariate_batch_event_shape(
        cls,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        **kwargs,
    ):
        loc_arr, _, batch_shape = _resolve_scale_tril(
            loc, cov, precision_matrix, scale_tril
        )
        event_shape = (int(loc_arr.shape[-1]),)
        return batch_shape, event_shape

    @classmethod
    def logpdf(
        cls,
        x: Array,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        **kwargs,
    ):
        """Log of the probability density function of the multivariate normal distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log probability density function
        loc : array_like
            Mean of the distribution
        cov : array_like, optional
            Covariance matrix
        precision_matrix : array_like, optional
            Precision matrix
        scale_tril : array_like, optional
            Lower triangular matrix with positive diagonal

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at x
        """
        x_arr = jnp.asarray(x)
        loc_arr = jnp.asarray(loc)
        precision_arr = _asarray_optional(precision_matrix)
        scale_arr = _asarray_optional(scale_tril)
        cov_arr = _asarray_optional(cov)

        event_dim = int(loc_arr.shape[-1])
        diff = x_arr - loc_arr

        if precision_arr is not None:
            M = jnp.einsum("...i,...ij,...j->...", diff, precision_arr, diff)
            sign, logdet = jnp.linalg.slogdet(precision_arr)
            sign_nonpos = jnp.any(sign <= 0)
            if not isinstance(sign_nonpos, jax_core.Tracer) and bool(sign_nonpos):
                raise ValueError("precision_matrix must be positive definite.")
            half_log_det = 0.5 * logdet
        elif scale_arr is not None:
            M = batch_mahalanobis(scale_arr, diff)
            half_log_det = jnp.sum(
                jnp.log(jnp.diagonal(scale_arr, axis1=-2, axis2=-1)),
                axis=-1,
            )
        elif cov_arr is not None:
            chol = jnp.linalg.cholesky(cov_arr)
            M = batch_mahalanobis(chol, diff)
            sign, logdet = jnp.linalg.slogdet(cov_arr)
            sign_nonpos = jnp.any(sign <= 0)
            if not isinstance(sign_nonpos, jax_core.Tracer) and bool(sign_nonpos):
                raise ValueError("covariance matrix must be positive definite.")
            half_log_det = 0.5 * logdet
        else:
            raise ValueError(
                "At least one of cov, precision_matrix, or scale_tril must be specified."
            )

        return -0.5 * (event_dim * jnp.log(2 * jnp.pi) + M) - half_log_det

    @classmethod
    def mean(cls, loc: Array, **kwargs):
        """Mean of the multivariate normal distribution.

        Parameters
        ----------
        loc : array_like
            Mean of the distribution

        Returns
        -------
        mean : ndarray
            Mean of the distribution
        """
        return loc

    @classmethod
    def mode(cls, loc: Array, **kwargs):
        """Mode of the multivariate normal distribution.

        Parameters
        ----------
        loc : array_like
            Mean of the distribution

        Returns
        -------
        mode : ndarray
            Mode of the distribution
        """
        return loc

    @classmethod
    def var(
        cls,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        **kwargs,
    ):
        """Variance of the multivariate normal distribution.

        Parameters
        ----------
        loc : array_like
            Mean of the distribution
        cov : array_like, optional
            Covariance matrix
        precision_matrix : array_like, optional
            Precision matrix
        scale_tril : array_like, optional
            Lower triangular matrix with positive diagonal

        Returns
        -------
        var : ndarray
            Variance of the distribution
        """
        cov_arr = _asarray_optional(cov)
        precision_arr = _asarray_optional(precision_matrix)
        scale_arr = _asarray_optional(scale_tril)

        if cov_arr is not None:
            return jnp.diagonal(cov_arr, axis1=-2, axis2=-1)
        if scale_arr is not None:
            cov_from_scale = scale_arr @ jnp.swapaxes(scale_arr, -1, -2)
            return jnp.diagonal(cov_from_scale, axis1=-2, axis2=-1)
        if precision_arr is not None:
            chol = jnp.linalg.cholesky(jnp.linalg.inv(precision_arr))
            cov_from_scale = chol @ jnp.swapaxes(chol, -1, -2)
            return jnp.diagonal(cov_from_scale, axis1=-2, axis2=-1)
        raise ValueError(
            "At least one of cov, precision_matrix, or scale_tril must be specified."
        )

    @classmethod
    def entropy(
        cls,
        loc: Array,
        cov: Optional[Array] = None,
        precision_matrix: Optional[Array] = None,
        scale_tril: Optional[Array] = None,
        **kwargs,
    ):
        """Entropy of the multivariate normal distribution.

        Parameters
        ----------
        loc : array_like
            Mean of the distribution
        cov : array_like, optional
            Covariance matrix
        precision_matrix : array_like, optional
            Precision matrix
        scale_tril : array_like, optional
            Lower triangular matrix with positive diagonal

        Returns
        -------
        entropy : ndarray
            Entropy of the distribution
        """
        cov_arr = _asarray_optional(cov)
        precision_arr = _asarray_optional(precision_matrix)
        scale_arr = _asarray_optional(scale_tril)

        if scale_arr is None:
            if cov_arr is not None:
                scale_arr = jnp.linalg.cholesky(cov_arr)
            elif precision_arr is not None:
                cov_from_precision = jnp.linalg.inv(precision_arr)
                scale_arr = jnp.linalg.cholesky(cov_from_precision)
            else:
                raise ValueError(
                    "At least one of cov, precision_matrix, or scale_tril must be specified."
                )

        half_log_det = jnp.sum(
            jnp.log(jnp.diagonal(scale_arr, axis1=-2, axis2=-1)),
            axis=-1,
        )
        event_dim = int(jnp.asarray(loc).shape[-1])
        return 0.5 * event_dim * (1 + jnp.log(2 * jnp.pi)) + half_log_det

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Closed-form estimators using sample mean and covariance."""
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[..., None]
        dtype = data.dtype

        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=dtype,
            mismatch_message="weights must have the same number of rows as data",
            column=True,
        )
        loc, cov = row_mean_and_cov(data, weights_arr, unbiased_unweighted=False)

        eps = jnp.asarray(1e-6, dtype=cov.dtype)
        cov = cov + eps * jnp.eye(cov.shape[-1], dtype=cov.dtype)
        return loc, cov


multivariate_normal = multivariate_normal_gen(name="multivariate_normal")
