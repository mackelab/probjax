import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from probjax.core import custom_inverse
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
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Flip module that reverses elements along a specified axis.

        Args:
            axis (int): Axis along which to flip the array. Defaults to -1 (last axis).
            rngs (nnx.Rngs, optional): Random number generators (unused but kept
                for API compatibility).
        """
        del rngs  # Unused but kept for compatibility
        self.axis = axis

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
        rngs: Optional[nnx.Rngs] = None,
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

        return rotate(x, rotation_matrix)


@custom_inverse
def _circulant_diagonal(x, scales, eigenvalues):
    """Apply diagonal/circulant factors without materializing dense matrices."""
    for index in range(eigenvalues.shape[0]):
        x = x * scales[index]
        x = jnp.fft.irfft(jnp.fft.rfft(x) * eigenvalues[index], n=x.shape[-1])
    return x * scales[-1]


@_circulant_diagonal.definv_and_logdet
def _circulant_diagonal_inverse(y, scales, eigenvalues):
    features = y.shape[-1]
    x = y / scales[-1]
    for index in reversed(range(eigenvalues.shape[0])):
        x = jnp.fft.irfft(jnp.fft.rfft(x) / eigenvalues[index], n=features)
        x = x / scales[index]

    # Interior RFFT bins represent conjugate pairs; DC and Nyquist occur once.
    log_amplitudes = jnp.log(jnp.abs(eigenvalues))
    spectral_logdet = 2 * jnp.sum(log_amplitudes) - jnp.sum(log_amplitudes[:, 0])
    if features % 2 == 0:
        spectral_logdet -= jnp.sum(log_amplitudes[:, -1])
    logdet = jnp.sum(jnp.log(scales)) + spectral_logdet
    return x, -logdet * (y.size // features)


class CirculantDiagonal(nnx.Module):
    """Learnable real linear mixing using alternating diagonal/circulant factors.

    Parameters
    ----------
    in_out_features : int
        Size of the final feature axis. Leading axes are batch axes.
    num_diagonals : int, optional
        Number of diagonal factors, with one circulant between adjacent factors.
        Defaults to two. One gives only per-feature scaling.
    init_std : float, optional
        Standard deviation of near-identity parameter initialization. Set to zero
        for the identity. Defaults to 0.05.
    max_log_scale : float, optional
        Bound on each diagonal log-scale and Fourier log-amplitude. Defaults to
        three. Bounds apply per factor, not to the complete layer's condition number.
    dtype, param_dtype : dtype, optional
        Computation and parameter dtypes. Parameters default to float32.
    rngs : nnx.Rngs
        Parameter initialization keys.

    Notes
    -----
    Positive diagonal scales and nonzero Fourier eigenvalues ensure invertibility.
    Interior Fourier bins have learned phases; DC and Nyquist eigenvalues remain
    positive real numbers so outputs are real. This is a restricted parameterization,
    not every possible circulant matrix. Forward and inverse use FFTs, and the
    inverse log-determinant is registered with ProbJax's inverse interpreter.
    """

    def __init__(
        self,
        in_out_features: int,
        *,
        num_diagonals: int = 2,
        init_std: float = 0.05,
        max_log_scale: float = 3.0,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if in_out_features < 1:
            raise ValueError("in_out_features must be positive")
        if num_diagonals < 1:
            raise ValueError("num_diagonals must be positive")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and non-negative")
        if not math.isfinite(max_log_scale) or max_log_scale <= 0:
            raise ValueError("max_log_scale must be finite and positive")
        self.in_out_features = in_out_features
        self.dtype = dtype
        self.max_log_scale = max_log_scale
        init = nnx.initializers.normal(stddev=init_std)
        self.log_scales = nnx.Param(
            init(rngs.params(), (num_diagonals, in_out_features), param_dtype)
        )
        self.log_amplitudes = nnx.Param(
            init(
                rngs.params(),
                (num_diagonals - 1, in_out_features // 2 + 1),
                param_dtype,
            )
        )
        self.phases = nnx.Param(
            init(
                rngs.params(),
                (num_diagonals - 1, (in_out_features - 1) // 2),
                param_dtype,
            )
        )

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        """Mix the final feature axis, preserving leading batch axes."""
        del args, rng
        x = jnp.asarray(x, dtype=self.dtype)
        if x.ndim == 0 or x.shape[-1] != self.in_out_features:
            raise ValueError(
                f"Expected a final feature axis of size {self.in_out_features}"
            )
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError("CirculantDiagonal requires real floating-point inputs")
        log_scales = self.log_scales[...].astype(x.dtype)
        log_amplitudes = self.log_amplitudes[...].astype(x.dtype)
        log_scales = self.max_log_scale * jnp.tanh(log_scales / self.max_log_scale)
        log_amplitudes = self.max_log_scale * jnp.tanh(
            log_amplitudes / self.max_log_scale
        )
        phases = jnp.pad(
            self.phases[...].astype(x.dtype),
            ((0, 0), (1, int(self.in_out_features % 2 == 0))),
        )
        eigenvalues = jnp.exp(log_amplitudes + 1j * phases)
        return _circulant_diagonal(x, jnp.exp(log_scales), eigenvalues)


class ElementwiseMonotone(nnx.Module):
    """Per-dimension monotone bijector with directly learnable parameters.

    Unlike the coupling/autoregressive conditioners, the bijector parameters
    are plain trainable params (one vector per dimension) — the
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
        rngs: nnx.Rngs,
    ):
        if in_out_features <= 0:
            raise ValueError("in_out_features must be positive")
        self.in_out_features = in_out_features
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        self.params = nnx.Param(
            params_init(
                rngs.next(), shape=(in_out_features, bijector_dim), dtype=param_dtype
            )
        )

    def __call__(self, x: ArrayLike, *args, rng: jax.Array | None = None) -> Array:
        del args, rng
        return self.bijector(self.params.get_value(), x)
