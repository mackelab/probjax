import math
from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from probjax.nn.layers.attention import MultiHeadAttention
from probjax.nn.layers.encoding import PosEncode, RotaryPosEncode
from probjax.nn.layers.fuse import AffineFuse, GatedFuse
from probjax.nn.layers.reg import DropPath
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
    identity_1x1,
)

from probjax.utils.typing import (
    Array,
    DTypeLike,
    ModuleLike,
    ModuleLikeType,
    PrecisionLike,
)


class ConvBlock(nnx.Module):
    """A convolutional block with optional normalization and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        padding: str = "SAME",
        norm_cls: ModuleLikeType | None = nnx.GroupNorm,
        activation: Callable = nnx.silu,
        preactivation: bool = True,
        input_dilation: int | Sequence[int] | None = 1,
        kernel_dilation: int | Sequence[int] | None = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        bias_init: Initializer = nnx.initializers.zeros,
        conv_general_dilated: Callable = jax.lax.conv_general_dilated,
        preferred_element_type: DTypeLike | None = None,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.Rngs,
    ):
        """Initializes the convolutional block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            rngs: Random number generators for initialization.
            kernel_size: Size of the convolutional kernel.
            padding: Padding type for the convolution. Can be 'SAME', 'VALID',
                'CIRCULAR', 'REFLECT' or a sequence of integers (low,high).
            strides: Strides for the convolution.
            precision: Precision for the convolution operation.
            dtype: Data type for the convolution operation.
            params_dtype: Data type for the parameters of the convolution.
            norm_cls: Normalization class to apply, e.g., LayerNorm or GroupNorm.
            activation: Activation function to apply after normalization.
            preferred_element_type: Preferred output data type after convolution.
            **kwargs: Additional keyword arguments for the convolutional layer.

        """
        # This ensures that if global precision rules are set, they
        # are respected, but if they are explicitly given, they are not
        # overridden.
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(nnx.Conv, **precision_kwargs)
        self._mesh = sharding
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            mask=mask,
            kernel_init=kernel_init,
            bias_init=bias_init,
            conv_general_dilated=conv_general_dilated,
            rngs=rngs,
            **precision_kwargs,
        )
        self.preferred_element_type = preferred_element_type
        self.preactivation = preactivation
        self.norm = (
            norm_cls(
                in_features,
                rngs=rngs,
            )
            if norm_cls is not None
            else None
        )
        self.activation = activation

    def __call__(self, x: Array, *, rng: jax.Array | None = None) -> Array:
        """Applies normalization, activation, and convolution."""
        del rng
        if self.preactivation:
            if self.norm is not None:
                x = self.norm(x)
            x = self.activation(x)
            x = self.conv(x).astype(self.preferred_element_type)
        else:
            if self.norm is not None:
                x = self.norm(x)
            x = self.conv(x).astype(self.preferred_element_type)
            x = self.activation(x)
        return x


class ResizeConv(nnx.Module):
    """Resize input spatially, then apply a convolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_shape: Sequence[int],
        *,
        resize_method: str = "bilinear",
        kernel_size: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        padding: str = "SAME",
        input_dilation: int | Sequence[int] | None = 1,
        kernel_dilation: int | Sequence[int] | None = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        bias_init: Initializer = nnx.initializers.zeros,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.Rngs,
    ):
        self.resize_method = resize_method
        self.out_shape = out_shape
        self._mesh = sharding

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(nnx.Conv, **precision_kwargs)

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            mask=mask,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
            **precision_kwargs,
        )
        self.preferred_element_type = preferred_element_type

    def __call__(self, x: Array, *, rng: jax.Array | None = None) -> Array:
        """Resizes input and applies convolution."""
        del rng
        x = jnp.asarray(x)
        shape = x.shape
        if shape[-1] != self.conv.in_features:
            raise ValueError(
                f"Input shape {shape} does not match expected in_features"
                f" {self.conv.in_features}"
            )
        if len(shape) < len(self.out_shape) + 1:
            raise ValueError(
                f"Input shape {shape} does not match expected spatial shape"
                f" {self.out_shape}"
            )
        new_shape = (
            shape[: -len(self.out_shape) - 1]
            + tuple(self.out_shape)
            + (self.conv.in_features,)
        )

        x = jax.image.resize(x, shape=new_shape, method=self.resize_method)
        x = self.conv(x).astype(self.preferred_element_type)
        return x


class RescaleConv(nnx.Module):
    """Resize input spatially, then apply a convolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        resize_factor: float,
        *,
        spatial_dims: int = 2,
        resize_method: str = "bilinear",
        kernel_size: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        padding: str = "SAME",
        input_dilation: int | Sequence[int] | None = 1,
        kernel_dilation: int | Sequence[int] | None = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        bias_init: Initializer = nnx.initializers.zeros,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.Rngs,
    ):
        self.resize_method = resize_method
        self.resize_factor = resize_factor
        self.spatial_dims = spatial_dims
        self._mesh = sharding

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(nnx.Conv, **precision_kwargs)

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            mask=mask,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
            **precision_kwargs,
        )
        self.preferred_element_type = preferred_element_type

    def __call__(self, x: Array, *, rng: jax.Array | None = None) -> Array:
        """Resizes input and applies convolution."""
        del rng
        x = jnp.asarray(x)
        shape = x.shape
        if shape[-1] != self.conv.in_features:
            raise ValueError(
                f"Input shape {shape} does not match expected in_features"
                f" {self.conv.in_features}"
            )
        if len(shape) < self.spatial_dims + 1:
            raise ValueError(
                f"Input shape {shape} does not match expected spatial dims"
                f" {self.spatial_dims}"
            )
        new_shape = (
            shape[: -self.spatial_dims - 1]
            + tuple(
                int(dim * self.resize_factor) + 1
                for dim in shape[-self.spatial_dims - 1 : -1]
            )
            + (self.conv.in_features,)
        )

        x = jax.image.resize(x, shape=new_shape, method=self.resize_method)
        x = self.conv(x).astype(self.preferred_element_type)
        return x


class ResnetBlock(nnx.Module):
    """A residual block with two convolutional layers and optional context."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        context_features: int | None = None,
        rescale_skip: bool = False,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        precision: PrecisionLike | None = None,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype | None = None,
        preferred_element_type=None,
        # Building block choices:
        context_fuse_cls: ModuleLikeType = AffineFuse,
        conv_block_cls: ModuleLikeType = ConvBlock,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initializes the ResNet block with two convolutional layers.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            rngs: Random number generators for initialization.
            conv_block_cls: Convolutional block builder. Must accept in/out features,
                kernel_size, strides, rngs and precision kwargs.
            context_features: Optional number of context features for context fusion.
            context_fuse_cls: Context fusion builder.
            dropout_rate: Standard dropout rate applied inside the block.
            drop_path_rate: Stochastic depth (DropPath) rate for the residual branch.
            kernel_size: Size of the convolutional kernel.
            padding: Padding type for the convolution.
            strides: Strides for the convolution.
            **kwargs: Additional keyword arguments for the convolutional block.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.preferred_element_type = preferred_element_type
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.rescale_skip = rescale_skip
        self._mesh = sharding

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(conv_block_cls, **precision_kwargs)

        if context_features is not None:
            self.context_fuse = context_fuse_cls(
                out_features, context_features, rngs=rngs
            )
        else:
            self.context_fuse = None

        _conv_block = partial(
            conv_block_cls,
            rngs=rngs,
            kernel_size=kernel_size,
            strides=strides,
            **precision_kwargs,
            **kwargs,
        )

        self.conv1 = _conv_block(in_features, out_features)
        self.conv2 = _conv_block(
            out_features, out_features, kernel_init=nnx.initializers.zeros
        )

        self.skip_connection = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1 if isinstance(kernel_size, int) else [1] * len(kernel_size),
            padding="SAME",
            use_bias=False,
            kernel_init=identity_1x1,
            rngs=rngs,
        )

        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        # Use a separate drop path rate for stochastic depth.
        if self.drop_path_rate > 0.0:
            self.dropout_path = DropPath(drop_rate=self.drop_path_rate, rngs=rngs)
        else:
            self.dropout_path = None

    def __call__(
        self,
        inputs: Array,
        context: Array | None = None,
        deterministic: bool = True,
        rng: jax.Array | None = None,
    ) -> Array:
        """Forward pass with optional context fusion and skip connection."""
        # First convolutional layer
        x = self.conv1(inputs, rng=rng)
        # Fuse context if provided
        if context is not None and self.context_fuse is not None:
            x = self.context_fuse(x, context, rng=rng)

        if self.dropout:
            x = self.dropout(x, deterministic=deterministic, rngs=rng)
        # Second convolutional layer
        x = self.conv2(x, rng=rng)

        # Residual connection
        skip_connection = self.skip_connection(inputs).astype(
            self.preferred_element_type
        )
        if self.dropout_path:
            x = self.dropout_path(x, deterministic=deterministic, rng=rng)
        out = x + skip_connection
        if self.rescale_skip:
            # Scale by sqrt(2) to preserve variance when adding
            # two independent, unit-variance variables.
            # See https://arxiv.org/abs/1512.03385
            out = out / jnp.sqrt(2.0)
        return out


class SpatialSelfAttention(nnx.Module):
    """Full-image self-attention for (B, H, W, C) tensors."""

    def __init__(
        self,
        in_features: int,
        rngs: nnx.Rngs,
        *,
        num_spatial_dims: int = 2,
        context_features: int | None = None,
        num_heads: int = 8,
        attn_size: int | None = None,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pos_emb: ModuleLike | None = None,
        precision: PrecisionLike | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        # Base building block choices:
        norm_cls: ModuleLikeType = nnx.GroupNorm,
        mha_cls: ModuleLikeType = MultiHeadAttention,
        sharding: jax.sharding.Mesh | None = None,
    ):
        self.preferred_element_type = preferred_element_type
        self.num_spatial_dims = num_spatial_dims
        self._mesh = sharding
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, None
        )
        precision_kwargs = filter_precision_kwargs(mha_cls, **precision_kwargs)

        # Layers
        self.norm = norm_cls(in_features, rngs=rngs)
        self.attn = mha_cls(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=in_features // num_heads
            if attn_size is None
            else attn_size * num_heads,
            out_features=in_features,
            dropout_rate=dropout_rate,
            rngs=rngs,
            **precision_kwargs,
        )
        if pos_emb is None:
            rotary_dim = (in_features // (2 * max(1, num_spatial_dims))) * (
                2 * max(1, num_spatial_dims)
            )
            if rotary_dim == 0:
                self.pos_emb = PosEncode(rngs=rngs)
            else:
                self.pos_emb = RotaryPosEncode(
                    token_dim=in_features,
                    rotary_dim=rotary_dim,
                    spatial_ndims=num_spatial_dims,
                    rngs=rngs,
                )
        else:
            self.pos_emb = pos_emb

        if context_features is not None:
            self.context_fuse = GatedFuse(in_features, context_features, rngs=rngs)
        else:
            self.context_fuse = None

        # Use a separate drop path rate for stochastic depth.
        if drop_path_rate > 0.0:
            self.dropout_path = DropPath(drop_rate=drop_path_rate, rngs=rngs)
        else:
            self.dropout_path = None

    def __call__(
        self,
        x: Array,
        context: Array | None = None,
        deterministic: bool = True,
        rng: jax.Array | None = None,
    ) -> Array:
        """Applies group normalization and multi-head self-attention.

        Dropout (attention dropout) is controlled via `dropout_rate`. Residual
        stochastic depth is controlled independently via `drop_path_rate`.
        """
        x = jnp.asarray(x)
        b = x.shape[: -self.num_spatial_dims - 1]
        spatial_dims = x.shape[-self.num_spatial_dims - 1 : -1]
        seq_len = math.prod(spatial_dims)
        c = x.shape[-1]
        x = x.reshape(*b, seq_len, c)
        x = self.pos_emb(x, rng=rng)
        x = x.reshape(*b, *spatial_dims, c)
        y = self.norm(x).reshape(*b, seq_len, c)  # →  (B, N, C)  with N = H·W
        y = self.attn(y, deterministic=deterministic, rng=rng)  # MultiHeadAttention
        y = y.reshape(*b, *spatial_dims, c)
        y = y.astype(self.preferred_element_type)
        if self.dropout_path:
            y = self.dropout_path(y, deterministic=deterministic, rng=rng)
        if self.context_fuse is not None and context is not None:
            y = self.context_fuse(x, y, context, deterministic=deterministic, rng=rng)
        else:
            y = x + y
        return y
