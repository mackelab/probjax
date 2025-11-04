from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, symmetric_positive_definite_matrix
from probjax.utils.linalg import batch_mahalanobis, batch_mv
from probjax.utils.typing import Array, ArrayLike, RngKey


class multivariate_normal_gen(rv_continuous):
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
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
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
    def rvs(
        cls,
        rng: RngKey,
        loc: Array = None,
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
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
        if loc.ndim < 1:
            raise ValueError("loc must be at least one-dimensional.")

        # Choose the most efficient computation path
        if scale_tril is not None:
            # Most efficient path - just need matrix-vector product
            if scale_tril.ndim < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                scale_tril.shape[:-2], loc.shape[:-1]
            )
            scale_tril = jnp.broadcast_to(
                scale_tril, batch_shape + scale_tril.shape[-2:]
            )
        elif cov is not None:
            # Need to compute Cholesky decomposition
            if cov.ndim < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(cov.shape[:-2], loc.shape[:-1])
            cov = jnp.broadcast_to(cov, batch_shape + cov.shape[-2:])
            scale_tril = jnp.linalg.cholesky(cov)
        elif precision_matrix is not None:
            # Need to compute inverse and Cholesky
            if precision_matrix.ndim < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            precision_matrix = jnp.broadcast_to(
                precision_matrix,
                batch_shape + precision_matrix.shape[-2:],
            )
            scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(precision_matrix))
        else:
            raise ValueError(
                "At least one of covariance_matrix, precision_matrix, or scale_tril "
                "must be specified."
            )

        loc = jnp.broadcast_to(loc, batch_shape + loc.shape[-1:])
        event_shape = loc.shape[-1:]
        shape = shape + batch_shape + event_shape

        eps = random.normal(rng, shape=shape, dtype=loc.dtype)
        return loc + batch_mv(scale_tril, eps)

    @classmethod
    def logpdf(
        cls,
        x: Array,
        loc: Array,
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
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
        # Choose the most efficient computation path
        if precision_matrix is not None:
            # Most efficient for logpdf - just need quadratic form
            diff = x - loc
            M = diff @ precision_matrix @ diff
            half_log_det = -0.5 * jnp.sum(
                jnp.log(jnp.diagonal(precision_matrix, axis1=-2, axis2=-1)),
                axis=-1,
            )
        elif scale_tril is not None:
            # Need to compute quadratic form with inverse
            diff = x - loc
            M = batch_mahalanobis(scale_tril, diff)
            half_log_det = jnp.sum(
                jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)),
                axis=-1,
            )
        elif cov is not None:
            # Need to compute inverse and determinant
            diff = x - loc
            scale_tril = jnp.linalg.cholesky(cov)
            M = batch_mahalanobis(scale_tril, diff)
            half_log_det = -0.5 * jnp.log(jnp.linalg.det(cov))
        else:
            raise ValueError(
                "At least one of covariance_matrix, precision_matrix, or scale_tril "
                "must be specified."
            )

        return -0.5 * (loc.shape[-1] * jnp.log(2 * jnp.pi) + M) - half_log_det

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
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
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
        if cov is not None:
            return jnp.diagonal(cov, axis1=-2, axis2=-1)
        else:
            if scale_tril is None:
                precision_matrix = kwargs.get("precision_matrix")
                scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(precision_matrix))
            return jnp.diagonal(scale_tril @ scale_tril.T, axis1=-2, axis2=-1)

    @classmethod
    def entropy(
        cls,
        loc: Array,
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
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
        if scale_tril is None:
            if cov is not None:
                scale_tril = jnp.linalg.cholesky(cov)
            else:
                scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(precision_matrix))

        half_log_det = jnp.sum(
            jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)),
            axis=-1,
        )
        return 0.5 * loc.shape[-1] * (1 + jnp.log(2 * jnp.pi)) + half_log_det

    def freeze(
        self,
        loc: Array,
        cov: Array = None,
        precision_matrix: Array = None,
        scale_tril: Array = None,
        **kwargs,
    ):
        """Freeze the distribution by fixing the parameters.

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
        frozen_dist : multivariate_normal_gen
            Frozen distribution
        """
        rv = super().freeze(
            loc=loc,
            cov=cov,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            **kwargs,
        )
        rv._batch_shape = loc.shape[:-1]
        rv._event_shape = loc.shape[-1:]
        return rv

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
        n = data.shape[0]

        if weights is not None:
            weights = jnp.asarray(weights, dtype=dtype).reshape((n,))
            if weights.shape[0] != n:
                raise ValueError("weights must have the same number of rows as data")
            weights = jnp.clip(weights, 0)
            total = jnp.sum(weights)
            total = jnp.where(total > 0, total, jnp.asarray(n, dtype=dtype))
            weights = weights / total
            loc = jnp.sum(weights[:, None] * data, axis=0)
            centered = data - loc
            cov = (centered * weights[:, None]).T @ centered
        else:
            loc = jnp.mean(data, axis=0)
            centered = data - loc
            denom = jnp.maximum(n - 1, 1)
            cov = centered.T @ centered / denom

        eps = jnp.asarray(1e-6, dtype=cov.dtype)
        cov = cov + eps * jnp.eye(cov.shape[-1], dtype=cov.dtype)
        return loc, cov


multivariate_normal = multivariate_normal_gen(name="multivariate_normal")
