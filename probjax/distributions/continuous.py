import jax
import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf, gammaln, digamma

from jaxtyping import Array
from typing import Optional
from warnings import warn

from .exponential_family import ExponentialFamily
from .distribution import Distribution
from .constraints import real, positive, unit_interval, square_matrix
from .utils import _precision_to_scale_tril

from probjax.utils.linalg import batch_mv, batch_mahalanobis

__all__ = ["Normal", "MultivariateNormal", "Gamma", "Beta", "Uniform"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import norm, gamma, beta


@register_pytree_node_class
class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        scale (float or ndarray): standard deviation of the distribution
            (often referred to as sigma)
    """

    arg_constraints = {"loc": real, "scale": positive}
    support = real

    def __init__(self, loc: Array | float, scale: Array | float):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        self.loc, self.scale = jnp.broadcast_arrays(loc, scale)

        super().__init__(batch_shape=loc.shape, event_shape=())

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def stddev(self) -> Array:
        return self.scale

    @property
    def variance(self) -> Array:
        return jnp.power(self.stddev, 2)

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        return norm.logpdf(value, self.loc, self.scale)

    def cdf(self, value):
        return norm.cdf(value, self.loc, self.scale)

    def icdf(self, value):
        return norm.ppf(value, self.loc, self.scale)

    def entropy(self):
        return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.scale)


@register_pytree_node_class
class MultivariateNormal(ExponentialFamily):
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
        "loc": real,
        "covariance_matrix": square_matrix,
        # "precision_matrix": square_matrix,
        # "scale_tril": square_matrix,
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
        return jnp.sum(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)**2, axis=-1)

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


@register_pytree_node_class
class Gamma(ExponentialFamily):
    r"""
    Creates a gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Gamma(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # gamma distribution with shape=2 and rate=3
        array([1.3750159], dtype=float32)

    Args:
        alpha (float or ndarray): shape parameter alpha
        beta (float or ndarray): rate parameter beta
    """

    arg_constraints = {"alpha": positive, "beta": positive}
    support = positive

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def concentration(self) -> Array:
        return self.alpha

    @property
    def rate(self) -> Array:
        return self.beta

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(key, self.alpha, shape) / self.beta

    def log_prob(self, value):
        return gamma.logpdf(value * self.beta, self.alpha)

    def cdf(self, value):
        return gamma.cdf(value * self.beta, self.alpha)


@register_pytree_node_class
class Beta(ExponentialFamily):
    r"""
    Creates a beta distribution parameterized by concentration parameters `alpha` and `beta`.

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Beta(jnp.array([2.0]), jnp.array([3.0]))
        >>> m.sample(key)  # beta distribution with alpha=2 and beta=3
        array([0.5302244], dtype=float32)

    Args:
        alpha (float or ndarray): concentration parameter alpha
        beta (float or ndarray): concentration parameter beta
    """

    arg_constraints = {"alpha": positive, "beta": positive}
    support = unit_interval

    def __init__(self, alpha: Array, beta: Array):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha, self.beta = jnp.broadcast_arrays(alpha, beta)

        super().__init__(batch_shape=alpha.shape, event_shape=())

    @property
    def concentration1(self) -> Array:
        return self.alpha

    @property
    def concentration0(self) -> Array:
        return self.beta

    def rsample(self, key, sample_shape: tuple = ()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.beta(key, self.alpha, self.beta, shape)

    def log_prob(self, value):
        return beta.logpdf(value, self.alpha, self.beta)

    def cdf(self, value):
        return beta.cdf(value, self.alpha, self.beta)

    def entropy(self):
        alpha, beta = self.alpha, self.beta
        return (
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1) * digamma(alpha)
            + (beta - 1) * digamma(beta)
            - (alpha + beta - 2) * digamma(alpha + beta)
        )


@register_pytree_node_class
class Uniform(Distribution):
    arg_constraints = {"low": real, "high": real}

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

        if not jnp.all(low < high):
            warn("Some elements of low are not less than corresponding elements of high, we will switch them.")
            self.low = jnp.where(low < high, low, high) - 1e-6


        super().__init__(batch_shape=jnp.shape(low), event_shape=())

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.uniform(key, shape, minval=self.low, maxval=self.high)

    def log_prob(self, value: Array) -> Array:
        return jnp.log(
            jnp.where(
                (value >= self.low) & (value <= self.high),
                1.0 / (self.high - self.low),
                0.0,
            )
        )

    def cdf(self, x: Array) -> Array:
        return jnp.where(
            x < self.low,
            0.0,
            jnp.where(x > self.high, 1.0, (x - self.low) / (self.high - self.low)),
        )

    def icdf(self, q: Array) -> Array:
        return self.low + q * (self.high - self.low)
