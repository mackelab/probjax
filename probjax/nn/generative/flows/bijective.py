import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from probjax.nn.sharding import ShardingCfg
from probjax.stats.bijective import rotate
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    RngKeyLike,
)


def skew_symmetric_to_rotation_matrix(skew_params: Array, n: int) -> Array:
    """Convert skew-symmetric parameters to rotation matrix via matrix exponential.

    Args:
        skew_params: Parameters for upper triangular part of skew-symmetric matrix.
            Should have shape (n*(n-1)//2,) for an n×n rotation matrix.
        n: Dimension of the rotation matrix. Must be positive.

    Returns:
        Orthogonal rotation matrix of shape (n, n).

    Raises:
        ValueError: If n is not positive or skew_params has incorrect shape.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    expected_params = n * (n - 1) // 2
    if skew_params.shape[-1] != expected_params:
        raise ValueError(
            f"skew_params must have {expected_params} elements for dimension {n}, "
            f"but got shape {skew_params.shape}"
        )

    # Create upper triangular indices
    i_indices, j_indices = jnp.triu_indices(n, k=1)

    # Create skew-symmetric matrix efficiently
    skew_matrix = jnp.zeros((n, n))
    skew_matrix = skew_matrix.at[i_indices, j_indices].set(skew_params)
    skew_matrix = skew_matrix.at[j_indices, i_indices].set(-skew_params)

    # Compute rotation matrix via matrix exponential
    return jax.scipy.linalg.expm(skew_matrix)


def identity_init(key: RngKeyLike, shape: tuple, dtype: DTypeLike = jnp.float32):
    """Initializer that returns an identity matrix.

    Args:
        key: JAX random key (unused but kept for API compatibility).
        shape: Shape of the array to initialize. Must be 2D and square.
        dtype: Data type of the initialized array. Defaults to float32.

    Returns:
        An identity matrix of the specified shape and dtype.

    Raises:
        ValueError: If shape is not 2D and square.
    """
    del key  # Unused but kept for compatibility
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"Identity initializer requires a square 2D shape, got {shape}"
        )
    return jnp.eye(shape[0], dtype=dtype)


class Affine(nnx.Module):
    """Affine transformation module."""

    def __init__(
        self,
        in_out_features: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        scale_init: Initializer = identity_init,
        bias_init: Initializer = nnx.initializers.zeros,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs,
    ):
        """Affine transformation module that applies scale and bias to input.

        Args:
            in_out_features (int): Input and output dimension. Must be positive.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            scale_init: Initializer for scale parameter. Defaults to identity_init.
            bias_init: Initializer for bias parameter. Defaults to zeros.
            rngs: Random number generators.

        Raises:
            ValueError: If in_out_features is not positive.
        """
        if in_out_features <= 0:
            raise ValueError("in_out_features must be positive")

        self.in_out_features = in_out_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

        self.scale = nnx.Param(
            scale_init(rngs.next(), shape=(in_out_features,), dtype=param_dtype)
        )
        self.bias = nnx.Param(
            bias_init(rngs.next(), shape=(in_out_features,), dtype=param_dtype)
        )

    def __call__(self, x: ArrayLike, *, rng: jax.Array | None = None) -> Array:
        """Apply affine transformation to input.

        Args:
            x: Input array of shape [..., in_out_features].

        Returns:
            Array with same shape as x, with affine transformation applied.
        """
        del rng
        x = jnp.asarray(x)
        scale = self.scale[...].astype(self.dtype) if self.dtype else self.scale[...]
        bias = self.bias[...].astype(self.dtype) if self.dtype else self.bias[...]
        return x * scale + bias


class Flip(nnx.Module):
    """Module for flipping arrays along a specified axis."""

    def __init__(
        self,
        axis: int = -1,
        *,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs = None,
    ):
        """Flip module that reverses elements along a specified axis.

        Args:
            axis (int): Axis along which to flip the array. Defaults to -1 (last axis).
            rngs (nnx.Rngs, optional): Random number generators (unused but kept
                for API compatibility).
        """
        del rngs  # Unused but kept for compatibility
        self.axis = axis
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        """Flip the input array along the specified axis.

        Args:
            x: Input array to flip.
            *args: Additional arguments (unused but kept for compatibility).

        Returns:
            Array with same shape as x, flipped along the specified axis.
        """
        del args, rng
        x = jnp.asarray(x)
        return jnp.flip(x, axis=self.axis)


class Permute(nnx.Module):
    """Module for permuting arrays along a specified axis."""

    def __init__(
        self,
        permutation: Array,
        axis: int = -1,
        *,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs = None,
    ):
        """Permutation module that reorders elements along a specified axis.

        Args:
            permutation (Array): An array of indices specifying the permutation.
                Must contain valid indices for the target axis.
            axis (int): Axis along which to permute. Defaults to -1 (last axis).
            rngs (nnx.Rngs, optional): Random number generators (unused but kept
                for API compatibility).

        Raises:
            ValueError: If permutation contains negative values.
        """
        del rngs  # Unused but kept for compatibility
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)
        permutation = jnp.asarray(permutation)

        if jnp.any(permutation < 0):
            raise ValueError("Permutation indices must be non-negative")

        self.permutation = nnx.Variable(permutation)
        self.axis = axis

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        """Apply permutation to the input array along the specified axis.

        Args:
            x: Input array to permute.
            *args: Additional arguments (unused but kept for compatibility).

        Returns:
            Array with same shape as x, permuted along the specified axis.

        Raises:
            ValueError: If permutation indices are out of bounds for the axis.
        """
        del args, rng
        x = jnp.asarray(x)
        return jnp.take(x, self.permutation[...], axis=self.axis)


class Rotate(nnx.Module):
    """Rotation transformation module."""

    def __init__(
        self,
        in_out_features: int,
        *,
        rotation_matrix: ArrayLike | None = None,
        learnable: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs,
    ):
        """Rotation module that applies orthogonal transformations to input.

        Args:
            in_out_features (int): Input and output dimension. Must be positive.
            rotation_matrix: Pre-specified rotation matrix. If None and not learnable,
                a random orthogonal matrix is generated.
            learnable (bool): Whether the rotation matrix is learnable.
                If True, uses skew-symmetric parameterization. Defaults to False.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            rngs: Random number generators.

        Raises:
            ValueError: If in_out_features is not positive.
        """
        if in_out_features <= 0:
            raise ValueError("in_out_features must be positive")

        self.in_out_features = in_out_features
        self.learnable = learnable
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

        if not learnable:
            if rotation_matrix is None:
                # Generate random orthogonal matrix
                self.rotation_matrix = nnx.Variable(
                    nnx.initializers.orthogonal()(
                        rngs.next(),
                        shape=(in_out_features, in_out_features),
                        dtype=param_dtype,
                    )
                )
            else:
                rotation_matrix = jnp.asarray(rotation_matrix)
                if rotation_matrix.shape != (in_out_features, in_out_features):
                    raise ValueError(
                        f"rotation_matrix shape {rotation_matrix.shape} "
                        f"doesn't match expected shape "
                        f"({in_out_features}, {in_out_features})"
                    )
                self.rotation_matrix = nnx.Variable(rotation_matrix)
        else:
            # Use skew-symmetric matrix parameterization for learnable rotation
            # The matrix exponential of any skew-symmetric matrix is orthogonal
            skew_params_size = in_out_features * (in_out_features - 1) // 2
            self.skew_params = nnx.Param(
                nnx.initializers.normal(stddev=0.1)(
                    rngs.next(), shape=(skew_params_size,), dtype=param_dtype
                )
            )

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        """Apply rotation transformation to input.

        Args:
            x: Input array of shape [..., in_out_features].
            *args: Additional arguments (unused but kept for compatibility).

        Returns:
            Array with same shape as x, with rotation applied.
        """
        del args, rng
        x = jnp.asarray(x)

        if not self.learnable:
            rotation_matrix = jax.lax.stop_gradient(self.rotation_matrix[...])
        else:
            rotation_matrix = skew_symmetric_to_rotation_matrix(
                self.skew_params[...], self.in_out_features
            )

        # Apply dtype conversion if needed
        if self.dtype:
            rotation_matrix = rotation_matrix.astype(self.dtype)
            x = x.astype(self.dtype)

        return rotate(rotation_matrix, x)


class ElementwiseMonotone(nnx.Module):
    """Per-dimension monotone bijector with directly learnable parameters.

    Unlike the coupling/autoregressive conditioners, the bijector parameters
    are plain trainable weights (one parameter vector per dimension) — the
    building block of Gaussianization flows, where expressivity comes from
    alternating elementwise layers with rotations.
    """

    def __init__(
        self,
        in_out_features: int,
        bijector_dim: int,
        bijector,
        *,
        params_init: Initializer = nnx.initializers.zeros,
        param_dtype: DTypeLike | None = None,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs,
    ):
        if in_out_features <= 0:
            raise ValueError("in_out_features must be positive")
        self.in_out_features = in_out_features
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)
        self.params = nnx.Param(
            params_init(
                rngs.next(), shape=(in_out_features, bijector_dim), dtype=param_dtype
            )
        )

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        del args, rng
        return self.bijector(self.params.get_value(), x)
