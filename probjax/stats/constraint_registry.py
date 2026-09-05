from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import lax

from .constraints import (
    Constraint,
    integer,
    lorentz,
    matrix,
    negative,
    negative_integer,
    positive,
    positive_integer,
    real,
    simplex,
    spherical,
    square_matrix,
    stiefel,
    strict_negative,
    strict_negative_integer,
    strict_positive,
    strict_positive_integer,
    symmetric_positive_definite_matrix,
    unit_interval,
    unit_square,
)

__all__ = [
    "biject_to",
    "transform_to",
    "manifold_registry",
]

expm = jsp_linalg.expm
sqrtm = jsp_linalg.sqrtm

if hasattr(jsp_linalg, "logm"):
    logm = jsp_linalg.logm  # type: ignore[attr-defined]
else:

    def logm(matrix: jnp.ndarray) -> jnp.ndarray:
        """Logarithm of a symmetric positive definite matrix."""
        eigvals, eigvecs = jnp.linalg.eigh(matrix)
        log_eigvals = jnp.log(jnp.clip(eigvals, a_min=1e-12))
        scaled_vecs = eigvecs * log_eigvals[..., None, :]
        return scaled_vecs @ jnp.swapaxes(eigvecs, -1, -2)


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None, *, inverse=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint:
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # Support calling on singleton instances.
        if isinstance(constraint, Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(constraint, Constraint):
            raise TypeError(
                "Expected constraint to be either a Constraint subclass or instance, "
                "but got {}".format(constraint)
            )

        transform = ConstraintTransform(factory, inverse)
        self._registry[constraint] = transform
        return transform

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints['scale']
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)           # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            transform = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return transform


@dataclass(frozen=True)
class ConstraintTransform:
    """Callable transform between unconstrained and constrained values."""

    forward: Callable
    inverse: Optional[Callable] = None

    def __call__(self, value):
        return self.forward(value)

    def inv(self, value):
        if self.inverse is None:
            raise NotImplementedError("This constraint transform has no inverse")
        return self.inverse(value)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


# Register constraints.
def identity(x):
    return x


def generate_matrix(x):
    m = jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape + (x.shape[-1],))
    return m


def generate_pdm(x):
    m = jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape + (x.shape[-1],))
    return m


biject_to.register(real, identity, inverse=identity)
transform_to.register(real)(identity)


transform_to.register(integer)(lax.round)
transform_to.register(positive_integer)(lambda x: lax.abs(lax.round(x)))
transform_to.register(negative_integer)(lambda x: -lax.abs(lax.round(x)))
transform_to.register(strict_positive_integer)(
    lambda x: jnp.maximum(lax.abs(lax.round(x)), 1)
)
transform_to.register(strict_negative_integer)(
    lambda x: -jnp.maximum(lax.abs(lax.round(x)), 1)
)

transform_to.register(positive)(lax.abs)
biject_to.register(positive, lax.exp, inverse=lax.log)

transform_to.register(strict_positive)(
    lambda x: jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(strict_positive, lax.exp, inverse=lax.log)

transform_to.register(negative)(lambda x: -lax.abs(x))
biject_to.register(negative, lambda x: -lax.exp(x), inverse=lambda x: lax.log(-x))

transform_to.register(strict_negative)(
    lambda x: -jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(
    strict_negative,
    lambda x: -lax.exp(x),
    inverse=lambda x: lax.log(-x),
)

transform_to.register(unit_interval)(jax.nn.sigmoid)
biject_to.register(unit_interval, jax.nn.sigmoid, inverse=jax.scipy.special.logit)
transform_to.register(unit_square)(lax.tanh)
biject_to.register(unit_square, lax.tanh, inverse=jnp.arctanh)

transform_to.register(simplex)(jax.nn.softmax)
biject_to.register(simplex, jax.nn.softmax, inverse=jnp.log)
transform_to.register(matrix)(generate_matrix)
transform_to.register(square_matrix)(generate_matrix)
transform_to.register(symmetric_positive_definite_matrix)(generate_pdm)


# Spherical manifold transformations
def spherical_exp_map(x, v):
    """Exponential map for spherical manifold."""
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return x * jnp.cos(norm_v) + v * jnp.sin(norm_v) / (norm_v + 1e-8)


def spherical_log_map(x, y):
    """Logarithmic map for spherical manifold."""
    dot_xy = jnp.sum(x * y, axis=-1, keepdims=True)
    dot_xy = jnp.clip(dot_xy, -1.0, 1.0)  # Ensure numerical stability
    theta = jnp.arccos(dot_xy)
    return theta * (y - x * dot_xy) / (jnp.sin(theta) + 1e-8)


def spherical_transform(x):
    """Transform to spherical manifold."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-8)


# Stiefel manifold transformations
def stiefel_exp_map(x, v):
    """Exponential map for Stiefel manifold."""
    # QR decomposition of the tangent vector
    q, r = jnp.linalg.qr(v)
    # Compute the exponential map
    return x @ expm(r)


def stiefel_log_map(x, y):
    """Logarithmic map for Stiefel manifold."""
    # Compute the tangent vector
    v = y - x @ (x.T @ y)
    return v


def stiefel_transform(x):
    """Transform to Stiefel manifold using QR decomposition."""
    q, _ = jnp.linalg.qr(x)
    return q


# SPD manifold transformations
def spd_exp_map(x, v):
    """Exponential map for SPD manifold."""
    x_sqrt = sqrtm(x)
    x_sqrt_inv = jnp.linalg.inv(x_sqrt)
    return x_sqrt @ expm(x_sqrt_inv @ v @ x_sqrt_inv) @ x_sqrt


def spd_log_map(x, y):
    """Logarithmic map for SPD manifold."""
    x_sqrt = sqrtm(x)
    x_sqrt_inv = jnp.linalg.inv(x_sqrt)
    return x_sqrt @ logm(x_sqrt_inv @ y @ x_sqrt_inv) @ x_sqrt


def spd_transform(x):
    """Map an unconstrained square matrix to a positive-definite matrix."""
    diagonal = jnp.exp(jnp.diagonal(x, axis1=-2, axis2=-1))
    lower = (
        jnp.tril(x, -1) + jnp.eye(x.shape[-1], dtype=x.dtype) * diagonal[..., None, :]
    )
    return lower @ jnp.swapaxes(lower, -1, -2)


def spd_inverse(x):
    """Return an unconstrained representative of a positive-definite matrix."""
    lower = jnp.linalg.cholesky(x)
    diagonal = jnp.log(jnp.diagonal(lower, axis1=-2, axis2=-1))
    return (
        jnp.tril(lower, -1)
        + jnp.eye(x.shape[-1], dtype=x.dtype) * diagonal[..., None, :]
    )


# Lorentz manifold transformations
def lorentz_exp_map(x, v):
    """Exponential map for Lorentz manifold."""
    norm_v = jnp.sqrt(jnp.sum(v[1:] ** 2, axis=-1))
    return jnp.concatenate([
        x[0] * jnp.cosh(norm_v) + v[0] * jnp.sinh(norm_v) / (norm_v + 1e-8),
        x[1:] * jnp.cosh(norm_v) + v[1:] * jnp.sinh(norm_v) / (norm_v + 1e-8),
    ])


def lorentz_log_map(x, y):
    """Logarithmic map for Lorentz manifold."""
    dot_xy = x[0] * y[0] - jnp.sum(x[1:] * y[1:], axis=-1)
    dot_xy = jnp.clip(dot_xy, 1.0, None)  # Ensure numerical stability
    theta = jnp.arccosh(dot_xy)
    return theta * (y - x * dot_xy) / (jnp.sqrt(dot_xy**2 - 1) + 1e-8)


def lorentz_transform(x):
    """Transform to Lorentz manifold."""
    norm = jnp.sqrt(jnp.sum(x[1:] ** 2, axis=-1))
    return jnp.concatenate([jnp.sqrt(1 + norm**2), x[1:]])


# Register the new transformations
transform_to.register(spherical)(spherical_transform)
biject_to.register(spherical, spherical_transform, inverse=identity)

transform_to.register(stiefel)(stiefel_transform)
biject_to.register(stiefel, stiefel_transform, inverse=identity)

transform_to.register(symmetric_positive_definite_matrix)(spd_transform)
biject_to.register(
    symmetric_positive_definite_matrix,
    spd_transform,
    inverse=spd_inverse,
)

transform_to.register(lorentz)(lorentz_transform)
biject_to.register(lorentz, lorentz_transform, inverse=identity)


# Add exponential and log maps to the registry
class ManifoldRegistry:
    """Registry for manifold exponential and logarithmic maps."""

    def __init__(self):
        self._exp_maps = {}
        self._log_maps = {}

    def register_exp_map(self, constraint, exp_map):
        """Register an exponential map for a constraint."""
        self._exp_maps[type(constraint)] = exp_map
        return exp_map

    def register_log_map(self, constraint, log_map):
        """Register a logarithmic map for a constraint."""
        self._log_maps[type(constraint)] = log_map
        return log_map

    def get_exp_map(self, constraint):
        """Get the exponential map for a constraint."""
        return self._exp_maps.get(type(constraint))

    def get_log_map(self, constraint):
        """Get the logarithmic map for a constraint."""
        return self._log_maps.get(type(constraint))


manifold_registry = ManifoldRegistry()

# Register the exponential and logarithmic maps
manifold_registry.register_exp_map(spherical, spherical_exp_map)
manifold_registry.register_log_map(spherical, spherical_log_map)

manifold_registry.register_exp_map(stiefel, stiefel_exp_map)
manifold_registry.register_log_map(stiefel, stiefel_log_map)

manifold_registry.register_exp_map(symmetric_positive_definite_matrix, spd_exp_map)
manifold_registry.register_log_map(symmetric_positive_definite_matrix, spd_log_map)

manifold_registry.register_exp_map(lorentz, lorentz_exp_map)
manifold_registry.register_log_map(lorentz, lorentz_log_map)
