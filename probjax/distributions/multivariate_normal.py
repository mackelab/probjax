import jax.numpy as jnp
import jax
from jax import random
from jax.scipy.special import erfinv, erf

from typing import Optional

from .distribution import Distribution

from probjax.utils.linalg import batch_mv, batch_mahalanobis

__all__ = ["MultivariateNormal"]

def _precision_to_scale_tril(P):
    Lf = jax.lax.cholesky(jnp.flip(P, (-2, -1)))
    L_inv = jnp.transpose(jnp.flip(Lf, (-2, -1)), (-2, -1))
    Id = jnp.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = jax.lax.triangular_solve(L_inv, Id, left_side=False, lower=False)
    return L


class MultivariateNormal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        covariance_matrix (float or ndarray): covariance matrix
        precision_matrix (float or ndarray): precision matrix
        scale_tril (float or ndarray): lower triangular matrix with positive
    """

    arg_constraints = {
        "loc": None,
        "covariance_matrix": None,
        "precision_matrix": None,
        "scale_tril": None,
    }

    def __init__(
        self,
        loc: jnp.array,
        covariance_matrix: Optional[jnp.array] = None,
        precision_matrix: Optional[jnp.array] = None,
        scale_tril: Optional[jnp.array] = None,
    ):
        if loc.ndim < 1:
            raise ValueError("loc must be at least one-dimensional.")

        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.ndim < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                scale_tril.shape[:-2], loc.shape[:-1]
            )
            self.scale_tril = jnp.broadcast_to(
                scale_tril, batch_shape + scale_tril.shape[-2:]
            )
            self.covariance_matrix = None
            self.precision_matrix = None
        elif covariance_matrix is not None:
            if covariance_matrix.ndim < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )

            self.covariance_matrix = jnp.broadcast_to(
                covariance_matrix, batch_shape + covariance_matrix.shape[-2:]
            )
            self.scale_tril = None
            self.precision_matrix = None
        else:
            if precision_matrix.ndim < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = jax.lax.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = jnp.broadcast_to(
                precision_matrix,
                batch_shape + precision_matrix.shape[-2:],
            )
            self.covariance_matrix = None
            self.scale_tril = None

        self.loc = jnp.broadcast_to(loc, batch_shape + loc.shape[-1:])

        event_shape = self.loc.shape[-1:]
        batch_shape = batch_shape
        event_shape = event_shape

        if covariance_matrix is not None:
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        else:  # precision_matrix is not None
            self.scale_tril = _precision_to_scale_tril(self.precision_matrix)

        super().__init__(batch_shape, event_shape)


    @property
    def mean(self) -> jnp.array:
        return self.loc

    @property
    def mode(self) -> jnp.array:
        return self.loc

    @property
    def variance(self) -> jnp.array:
        return (
            self.scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, key, sample_shape=tuple):
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape=shape, dtype=self.loc.dtype)
        return self.loc + batch_mv(self.scale_tril, eps)
        

    def log_prob(self, value: jnp.array):
        diff = value - self.loc
        M = batch_mahalanobis(self.scale_tril, diff)
        half_log_det = jnp.sum(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)),
            axis=-1,
        )
        return -0.5 * (self._event_shape[0] * jnp.log(2 * jnp.pi) + M) - half_log_det
