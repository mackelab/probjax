import math
from typing import Optional

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

    def __init__(self, max_seq_len: int = 10_000, rngs: nnx.Rngs = None):
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
            precision=precision,
            preferred_element_type=preferred_element_type,
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
