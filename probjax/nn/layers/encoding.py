import math
import numbers
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    PrecisionLike,
)

default_embed_init = nnx.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0
)


class PosEncode(nnx.Module):
    """Sinusoidal positional embedding module."""

    def __init__(self, max_seq_len: int = 10_000, *, rngs: nnx.Rngs):
        """Positional embedding module using sinusoidal patterns.

        Args:
            max_seq_len (int): Maximum sequence length for positional encoding.
                Defaults to 10,000.
            rngs (nnx.Rngs, optional): Random number generators (unused but kept
                for API compatibility).
        """
        del rngs
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len

    def __call__(self, x: Array, idx: Optional[Array] = None) -> Array:
        """Add sinusoidal positional encoding to input.

        Args:
            x: Input array of shape [..., seq_len, embedding_dim]
            idx: Optional position indices of shape [seq_len]. If None,
                uses sequential positions starting from 0.

        Returns:
            Array with same shape as x, with positional encoding added.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        seq_len = x.shape[-2]
        token_dim = x.shape[-1]

        if idx is None:
            idx = jnp.arange(seq_len, dtype=jnp.float32)
        else:
            idx = jnp.asarray(idx, dtype=jnp.float32)
            if idx.shape != (seq_len,):
                raise ValueError(
                    f"idx shape {idx.shape} doesn't match sequence length {seq_len}"
                )

        # Create position encoding using the standard formula
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = jnp.exp(
            jnp.arange(0, token_dim, 2, dtype=jnp.float32)
            * (-jnp.log(self.max_seq_len) / token_dim)
        )

        # Reshape for broadcasting: [seq_len, 1] * [token_dim//2]
        pos_encoding = jnp.zeros((seq_len, token_dim), dtype=x.dtype)
        pos_encoding = pos_encoding.at[:, 0::2].set(
            jnp.sin(idx[:, None] * div_term[None, :])
        )
        pos_encoding = pos_encoding.at[:, 1::2].set(
            jnp.cos(idx[:, None] * div_term[None, :])
        )

        # Reshape to match input dimensions
        batch_shape = x.shape[:-2]
        pos_encoding = pos_encoding.reshape(
            (1,) * len(batch_shape) + pos_encoding.shape
        )

        return x + pos_encoding


class RotaryPosEncode(nnx.Module):
    """Rotary positional encoding module supporting 1D and ND coordinates."""

    def __init__(
        self,
        token_dim: int,
        *,
        max_seq_len: int = 4_096,
        base: float = 10_000.0,
        rotary_dim: Optional[int] = None,
        spatial_ndims: int | None = None,
        spatial_shape: int | Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        cache_cos_sin: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        """Rotary positional embedding module (RoPE).

        Args:
            token_dim: Feature dimension of the incoming tensor.
            max_seq_len: Maximum sequence length cached for rotary frequencies.
            base: Exponential base used to compute inverse frequencies.
            rotary_dim: Number of leading features to rotate. Defaults to
                ``token_dim``.
            spatial_ndims: Number of spatial dimensions expected when using
                structured grids. Defaults to 1. Ignored when ``spatial_shape``
                is provided.
            spatial_shape: Optional static spatial shape. When ``None`` the
                module treats inputs as 1D unless an explicit shape is supplied
                at call time.
            dtype: Optional dtype used for cached cos/sin tables.
            cache_cos_sin: If True, precomputes cos/sin tables up to
                ``max_seq_len`` for faster lookups when using sequential
                positions.
            rngs: Random number generators (unused, kept for API consistency).
        """
        del rngs
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.token_dim = token_dim
        self.rotary_dim = rotary_dim or token_dim
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even in order to apply rotations")
        if self.rotary_dim > token_dim:
            raise ValueError("rotary_dim cannot exceed token_dim")
        if spatial_shape is not None:
            if isinstance(spatial_shape, numbers.Integral):
                parsed_shape = (int(spatial_shape),)
            else:
                parsed_shape = tuple(int(dim) for dim in spatial_shape)
            if not parsed_shape:
                raise ValueError("spatial_shape must contain at least one dimension")
            if spatial_ndims is not None and int(spatial_ndims) != len(parsed_shape):
                raise ValueError(
                    "spatial_shape length must match spatial_ndims when both are provided"
                )
            self.spatial_shape = parsed_shape
            self.spatial_ndims = len(parsed_shape)
        else:
            if spatial_ndims is None:
                spatial_ndims = 1
            if spatial_ndims <= 0:
                raise ValueError("spatial_ndims must be positive")
            self.spatial_shape = None
            self.spatial_ndims = int(spatial_ndims)

        self.max_seq_len = max_seq_len
        self.base = base
        self.dtype = dtype

        self._full_inv_freq = self._compute_inv_freq(self.rotary_dim)

        if cache_cos_sin:
            positions = jnp.arange(max_seq_len, dtype=self._full_inv_freq.dtype)
            cos, sin = self._compute_cos_sin(positions, self._full_inv_freq)
            self.cos_cache = nnx.Variable(cos)
            self.sin_cache = nnx.Variable(sin)
        else:
            self.cos_cache = None
            self.sin_cache = None

    def _compute_inv_freq(self, rotary_dim: int) -> Array:
        dtype = self.dtype or jnp.float32
        base = jnp.asarray(self.base, dtype=dtype)
        exponents = jnp.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim
        return 1.0 / (base**exponents)

    def _compute_cos_sin(
        self,
        positions: Array,
        inv_freq: Array,
    ) -> tuple[Array, Array]:
        positions = jnp.asarray(positions, dtype=inv_freq.dtype)
        angles = positions[:, None] * inv_freq[None, :]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        if self.dtype is not None:
            cos = cos.astype(self.dtype)
            sin = sin.astype(self.dtype)
        return cos, sin

    def _lookup_cos_sin_1d(
        self,
        seq_len: int,
        positions: Optional[Array],
        *,
        offset: float,
    ) -> tuple[Array, Array]:
        if (
            positions is None
            and self.cos_cache is not None
            and self.sin_cache is not None
        ):
            offset_int = int(offset)
            if offset_int == offset and offset_int >= 0:
                end = offset_int + seq_len
                if end <= self.max_seq_len:
                    cos = self.cos_cache.value[offset_int:end]
                    sin = self.sin_cache.value[offset_int:end]
                    return cos, sin
        if positions is None:
            positions = jnp.arange(seq_len, dtype=self._full_inv_freq.dtype) + offset
        else:
            positions = jnp.asarray(positions, dtype=self._full_inv_freq.dtype)
        return self._compute_cos_sin(positions, self._full_inv_freq)

    def _apply_rotary(self, x_slice: Array, cos: Array, sin: Array) -> Array:
        cos = cos.reshape((1,) * (x_slice.ndim - 2) + cos.shape)
        sin = sin.reshape((1,) * (x_slice.ndim - 2) + sin.shape)
        even = x_slice[..., ::2]
        odd = x_slice[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = odd * cos + even * sin
        return jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(x_slice.shape)

    def _normalize_offset(self, offset, dims: int) -> tuple[float, ...]:
        if isinstance(offset, numbers.Real):
            return (float(offset),) * dims
        if isinstance(offset, (tuple, list)):
            if len(offset) == dims:
                return tuple(float(o) for o in offset)
            if len(offset) == 1:
                return tuple(float(offset[0]) for _ in range(dims))
        raise TypeError(
            "offset must be a scalar or a sequence matching the positional dimensionality"
        )

    def __call__(
        self,
        x: ArrayLike,
        idx: ArrayLike | None = None,
        *,
        offset=0,
    ) -> Array:
        """Apply rotary positional encoding.

        Args:
            x: Input array of shape [..., seq_len, token_dim].
            idx: Optional positions. For 1D inputs the shape is ``[seq_len]``.
                For ND inputs supply ``[seq_len, ndim]`` coordinates.
            offset: Scalar (1D) or sequence (ND) indicating the starting
                coordinate when ``idx`` is not provided or when a shift is
                required.

        Returns:
            Array with rotary encodings applied to the first ``rotary_dim`` features.

        Examples:
            1. Sequential 1D tokens::

                   x = jnp.zeros((batch, length, model_dim))
                   rope = RotaryPosEncode(model_dim)
                   x = rope(x)  # implicit indices [0, 1, ..., length - 1]

            2. 2D image grid::

                   h, w = 16, 16
                   coords = jnp.stack(jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij"), axis=-1)
                   coords = coords.reshape(h * w, 2)
                   x = jnp.zeros((batch, h * w, model_dim))
                   rope = RotaryPosEncode(model_dim, rotary_dim=64)
                   x = rope(x, idx=coords)
        """
        x = jnp.asarray(x)
        seq_len = x.shape[-2]

        idx_arr = None if idx is None else jnp.asarray(idx)
        if idx_arr is not None and idx_arr.shape[0] != seq_len:
            raise ValueError(
                f"idx shape {idx_arr.shape} doesn't match sequence length {seq_len}"
            )
        if idx_arr is not None and idx_arr.ndim > 2:
            raise ValueError("idx must be rank 1 or 2")

        position_dims = 1 if idx_arr is None or idx_arr.ndim == 1 else idx_arr.shape[-1]

        offsets = self._normalize_offset(offset, position_dims)

        rotary_slice = x[..., : self.rotary_dim]
        remainder = (
            x[..., self.rotary_dim :] if self.rotary_dim < self.token_dim else None
        )

        if position_dims == 1:
            idx_1d = None if idx_arr is None else idx_arr.reshape((seq_len,))
            cos, sin = self._lookup_cos_sin_1d(seq_len, idx_1d, offset=offsets[0])
            rotary_out = self._apply_rotary(rotary_slice, cos, sin)
        else:
            if self.rotary_dim % position_dims != 0:
                raise ValueError(
                    "rotary_dim must be divisible by the number of positional dimensions"
                )
            chunk = self.rotary_dim // position_dims
            if chunk % 2 != 0:
                raise ValueError("rotary_dim per positional dimension must be even")
            if idx_arr is None:
                raise ValueError(
                    "idx must be provided when using ND rotary coordinates"
                )
            inv_freq_axis = self._compute_inv_freq(chunk)
            rotated_parts = []
            for axis in range(position_dims):
                pos_axis = idx_arr[:, axis] + offsets[axis]
                cos_axis, sin_axis = self._compute_cos_sin(pos_axis, inv_freq_axis)
                axis_slice = rotary_slice[..., axis * chunk : (axis + 1) * chunk]
                rotated_parts.append(self._apply_rotary(axis_slice, cos_axis, sin_axis))
            rotary_out = jnp.concatenate(rotated_parts, axis=-1)

        if remainder is not None:
            return jnp.concatenate([rotary_out, remainder], axis=-1)
        return rotary_out

    def _apply_spatial(
        self,
        x: ArrayLike,
        spatial_shape: int | Sequence[int] | None = None,
        *,
        offset=0,
    ) -> Array:
        """Apply rotary encoding over a structured spatial grid.

        Args:
            x: Input array shaped [..., prod(spatial_shape), token_dim].
            spatial_shape: Spatial dimensions that were flattened into the
                sequence axis. When ``None``, the instance uses ``self.spatial_shape``
                if available, otherwise infers a 1D layout.
            offset: Optional scalar or per-dimension offset applied via the
                rotary phase.

        Returns:
            Array with rotary encodings applied along the final-but-one axis.
        """
        x = jnp.asarray(x)
        if spatial_shape is None:
            if self.spatial_shape is not None:
                dims = self.spatial_shape
            elif self.spatial_ndims == 1:
                dims = (x.shape[-2],)
            else:
                raise ValueError(
                    "spatial_shape must be provided for rotary encodings with more than one dimension"
                )
        elif isinstance(spatial_shape, numbers.Integral):
            dims = (int(spatial_shape),)
        else:
            dims = tuple(int(d) for d in spatial_shape)
        if not dims:
            raise ValueError("spatial_shape must contain at least one dimension")
        if len(dims) != self.spatial_ndims:
            raise ValueError(
                f"Expected spatial_shape with {self.spatial_ndims} dims but received {len(dims)}"
            )
        seq_len = math.prod(dims)
        if x.shape[-2] != seq_len:
            raise ValueError(
                f"Expected sequence length {seq_len} for spatial_shape {dims} "
                f"but received {x.shape[-2]}"
            )
        dtype = self._full_inv_freq.dtype
        if len(dims) == 1:
            coords = jnp.arange(seq_len, dtype=dtype)
        else:
            axes = jnp.meshgrid(
                *[jnp.arange(dim, dtype=dtype) for dim in dims], indexing="ij"
            )
            coords = jnp.stack(axes, axis=-1).reshape(seq_len, len(dims))
        return self(x, idx=coords, offset=offset)


class LearnablePosEncode(nnx.Module):
    """Learned positional embedding module."""

    def __init__(
        self,
        in_out_features: int,
        max_seq_len: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        embedding_init: Initializer = default_embed_init,
        rngs: nnx.Rngs,
    ):
        """Learned positional embedding module.

        Args:
            in_out_features (int): Dimension of the input/output features.
            max_seq_len (int): Maximum sequence length.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            embedding_init: Embedding initializer.
            rngs: Random number generators.
        """
        del precision, preferred_element_type  # Unused but kept for API consistency
        if in_out_features <= 0:
            raise ValueError("in_out_features must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.max_seq_len = max_seq_len
        self.embed = nnx.Embed(
            max_seq_len,
            in_out_features,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=embedding_init,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike, idx: ArrayLike | None = None) -> Array:
        """Add learned positional embeddings to input.

        Args:
            x: Input array of shape [..., seq_len, features]
            idx: Optional position indices of shape [seq_len]. If None,
                uses sequential positions starting from 0.

        Returns:
            Array with same shape as x, with positional encoding added.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        x = jnp.asarray(x)
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        if idx is None:
            idx = jnp.arange(seq_len)
        else:
            idx = jnp.asarray(idx)
            if idx.shape != (seq_len,):
                raise ValueError(
                    f"idx shape {idx.shape} doesn't match sequence length {seq_len}"
                )

        pos_emb = self.embed(idx)

        # Reshape to match input dimensions: [..., seq_len, features]
        batch_shape = x.shape[:-2]
        pos_emb = pos_emb.reshape((1,) * len(batch_shape) + pos_emb.shape)

        return x + pos_emb


class GaussianFourierEmbedding(nnx.Module):
    """Gaussian Fourier embedding module for continuous inputs like time."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        learnable: bool = True,
        scale_init: Initializer = nnx.initializers.normal(1.0),
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time or
        other scalar quantities. It can also be used to embed scalar
        features that lie on different scales.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Output dimension of the embedding.
            rngs (nnx.Rngs): Random number generators.
            learnable (bool): Whether the embedding matrix is learnable.
                Defaults to True.
            scale_init (Initializer): Initializer for the frequency matrix.
                Defaults to normal with stddev=1.0.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
        Raises:
            ValueError: If input_dim or output_dim are not positive.
        """
        if in_features <= 0:
            raise ValueError("input_dim must be positive")
        if out_features <= 0:
            raise ValueError("output_dim must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.learnable = learnable
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.preferred_element_type = preferred_element_type

        # Use half_dim to ensure we can create the full output_dim
        half_dim = math.ceil(out_features / 2)

        # Initialize frequency matrix B with shape [half_dim, input_dim]
        P_init = scale_init(
            rngs.next(), shape=(half_dim, in_features), dtype=param_dtype
        )

        if learnable:
            self.P = nnx.Param(P_init)
        else:
            self.P = nnx.Variable(P_init)

    def __call__(self, inputs: ArrayLike) -> Array:
        """Apply Gaussian Fourier embedding to inputs.

        Args:
            inputs: Input array of shape [..., input_dim]

        Returns:
            Array of shape [..., output_dim] with Fourier features.
        """
        inputs = jnp.asarray(inputs)
        P = self.P.value

        # Ensure P has the correct compute dtype
        P = P.astype(self.dtype) if self.dtype else P
        inputs = inputs.astype(self.dtype) if self.dtype else inputs

        if not self.learnable:
            # Ensure P is not updated during backprop
            P = jax.lax.stop_gradient(P)

        # Compute 2π * inputs @ P^T
        frequencies = (2 * jnp.pi / self.in_features) * jnp.dot(
            inputs,
            P.T,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
        )
        # Ensure correct preferred element type
        if self.preferred_element_type:
            frequencies = frequencies.astype(self.preferred_element_type)

        # Apply sin and cos
        cos_features = jnp.cos(frequencies)
        sin_features = jnp.sin(frequencies)

        # Concatenate and truncate to exact output_dim
        features = jnp.concatenate([cos_features, sin_features], axis=-1)
        return features[..., : self.out_features]


class OneHot(nnx.Module):
    """One-hot encoding module."""

    def __init__(self, max_num_sequence: int, *, rngs: nnx.Rngs):
        """One-hot encoding module.

        Args:
            num_tokens (int): Number of distinct tokens/classes.
            rngs (nnx.Rngs, optional): Random number generators (unused but kept
                for API compatibility).

        Raises:
            ValueError: If num_tokens is not positive.
        """
        del rngs  # Unused but kept for compatibility
        if max_num_sequence <= 0:
            raise ValueError("num_tokens must be positive")
        self.num_tokens = max_num_sequence

    def __call__(self, x: ArrayLike) -> Array:
        """One-hot encode the input.

        Args:
            x: Input array of integer indices with shape [...].

        Returns:
            Array of shape [..., num_tokens] with one-hot encoded values.

        Raises:
            ValueError: If input contains indices outside valid range.
        """
        x = jnp.asarray(x)

        # Validate input range
        if jnp.any(x < 0) or jnp.any(x >= self.num_tokens):
            raise ValueError(
                f"Input indices must be in range [0, {self.num_tokens}), "
                f"but got min={jnp.min(x)}, max={jnp.max(x)}"
            )

        return jax.nn.one_hot(x, self.num_tokens)
