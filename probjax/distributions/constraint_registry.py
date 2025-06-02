import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import sqrtm

from .constraints import (
    Constraint,
    integer,
    matrix,
    negative,
    negative_integer,
    positive,
    positive_definite_matrix,
    positive_integer,
    real,
    simplex,
    square_matrix,
    strict_negative,
    strict_negative_integer,
    strict_positive,
    strict_positive_integer,
    unit_interval,
    unit_square,
    spherical,
    stiefel,
    grassmannian,
    lorentz,
)

__all__ = [
    "biject_to",
    "transform_to",
    "manifold_registry",
]


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
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

        def factory_wrapper(*args):
            out = jax.tree_util.tree_map(factory, args)
            if len(out) == 1:
                return out[0]
            else:
                return out

        self._registry[constraint] = factory_wrapper

        return factory_wrapper

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
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory


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


biject_to.register(real)(identity)
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
biject_to.register(positive)(lax.exp)

transform_to.register(strict_positive)(
    lambda x: jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(strict_positive)(lambda x: lax.exp(x) + jnp.finfo(x.dtype).eps)

transform_to.register(negative)(lambda x: -lax.abs(x))
biject_to.register(negative)(lambda x: -lax.exp(x))

transform_to.register(strict_negative)(
    lambda x: -jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(strict_negative)(lambda x: -lax.exp(x) - jnp.finfo(x.dtype).eps)

transform_to.register(unit_interval)(jax.nn.sigmoid)
biject_to.register(unit_interval)(jax.nn.sigmoid)
transform_to.register(unit_square)(lax.tanh)
biject_to.register(unit_square)(lax.tanh)

transform_to.register(simplex)(jax.nn.softmax)
transform_to.register(matrix)(generate_matrix)
transform_to.register(square_matrix)(generate_matrix)
transform_to.register(symmetric_positive_matrix)(generate_pdm)


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
    return x @ jnp.linalg.expm(r)


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
    return x_sqrt @ jnp.linalg.expm(x_sqrt_inv @ v @ x_sqrt_inv) @ x_sqrt


def spd_log_map(x, y):
    """Logarithmic map for SPD manifold."""
    x_sqrt = sqrtm(x)
    x_sqrt_inv = jnp.linalg.inv(x_sqrt)
    return x_sqrt @ jnp.logm(x_sqrt_inv @ y @ x_sqrt_inv) @ x_sqrt


def spd_transform(x):
    """Transform to SPD manifold."""
    # Ensure symmetry
    x = (x + x.T) / 2
    # Add small diagonal term to ensure positive definiteness
    x = x + jnp.eye(x.shape[0]) * 1e-6
    return x


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
biject_to.register(spherical)(spherical_transform)

transform_to.register(stiefel)(stiefel_transform)
biject_to.register(stiefel)(stiefel_transform)

transform_to.register(symmetric_positive_matrix)(spd_transform)
biject_to.register(symmetric_positive_matrix)(spd_transform)

transform_to.register(lorentz)(lorentz_transform)
biject_to.register(lorentz)(lorentz_transform)


# Add exponential and log maps to the registry
class ManifoldRegistry:
    """Registry for manifold exponential and logarithmic maps."""

    def __init__(self):
        self._exp_maps = {}
        self._log_maps = {}

    def register_exp_map(self, constraint, exp_map):
        self._exp_maps[constraint] = exp_map
        return exp_map

    def register_log_map(self, constraint, log_map):
        self._log_maps[constraint] = log_map
        return log_map

    def get_exp_map(self, constraint):
        return self._exp_maps.get(constraint)

    def get_log_map(self, constraint):
        return self._log_maps.get(constraint)


manifold_registry = ManifoldRegistry()

# Register the exponential and logarithmic maps
manifold_registry.register_exp_map(spherical, spherical_exp_map)
manifold_registry.register_log_map(spherical, spherical_log_map)

manifold_registry.register_exp_map(stiefel, stiefel_exp_map)
manifold_registry.register_log_map(stiefel, stiefel_log_map)

manifold_registry.register_exp_map(symmetric_positive_matrix, spd_exp_map)
manifold_registry.register_log_map(symmetric_positive_matrix, spd_log_map)

manifold_registry.register_exp_map(lorentz, lorentz_exp_map)
manifold_registry.register_log_map(lorentz, lorentz_log_map)
