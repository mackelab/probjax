from functools import partial
from typing import Callable, Sequence, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Conv, ConvTranspose
from flax.typing import PrecisionLike, Dtype, Array, Initializer
from jax.lax import conv_general_dilated
from jax.typing import DTypeLike

from probjax.nn.attention import MultiHeadAttention
from probjax.nn.utils import AdditiveFuse, AffineFuse, Sequential


def identity_1x1(_, shape, dtype=jnp.float32):
    """Kernel init for a 1×1 Conv that starts as identity.

    Works for (1, 1, C_in, C_out).  If C_in ≠ C_out the extra
    channels are zero-filled.
    """
    k = jnp.zeros(shape, dtype)
    diag = jnp.arange(min(shape[2], shape[3]))
    # set W[0, 0, i, i] = 1
    k = k.at[0, 0, diag, diag].set(1.0)
    return k


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
        norm: type[nnx.LayerNorm] | type[nnx.GroupNorm] | None = nnx.GroupNorm,
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
        conv_general_dilated: Callable = conv_general_dilated,
        preferred_element_type: jnp.dtype | None = None,
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
            norm: Type of normalization to apply, either `LayerNorm` or `GroupNorm`.
            activation: Activation function to apply after normalization.
            preferred_element_type: Preferred output data type after convolution.
            **kwargs: Additional keyword arguments for the convolutional layer.

        """
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            precision=precision,
            dtype=dtype,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            mask=mask,
            kernel_init=kernel_init,
            bias_init=bias_init,
            conv_general_dilated=conv_general_dilated,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # This would be nice to be inside `nnx.Conv`, but it is not
        # currently supported.
        self.preferred_element_type = preferred_element_type
        self.preactivation = preactivation
        self.norm = (
            norm(
                in_features,
                rngs=rngs,
            )
            if norm is not None
            else None
        )
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        """Applies normalization, activation, and convolution."""
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
        rngs: nnx.Rngs,
    ):
        self.resize_method = resize_method
        self.out_shape = out_shape
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
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.preferred_element_type = preferred_element_type

    def __call__(self, x: Array) -> Array:
        """Resizes input and applies convolution."""
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
        dropout_rate: float = 0.0,
        precision: PrecisionLike | None = None,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype | None = None,
        preferred_element_type=None,
        context_fuse: type[AffineFuse] | type[AdditiveFuse] = AffineFuse,
        conv_block: type[nnx.Module] = ConvBlock,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initializes the ResNet block with two convolutional layers.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            rngs: Random number generators for initialization.
            conv_block: Type of convolutional block to use, needs to be a subclass of `nnx.Module`.
                and should accept `kernel_size`, `padding`, and `strides` as keyword arguments.
            context_features: Optional number of context features for context fusion.
            context_fuse: Type of context fusion to use, either `AffineFuse` or `AdditiveFuse`.
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

        if context_features is not None:
            self.context_fuse = context_fuse(out_features, context_features, rngs=rngs)

        _conv_block = partial(
            conv_block,
            rngs=rngs,
            kernel_size=kernel_size,
            strides=strides,
            precision=precision,
            dtype=dtype,
            preferred_element_type=preferred_element_type,
            param_dtype=param_dtype,
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
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self, inputs: Array, context: Array | None = None, deterministic: bool = True
    ):
        """Forward pass with optional context fusion and skip connection."""
        # First convolutional layer
        x = self.conv1(inputs)
        # Fuse context if provided
        if context is not None:
            x = self.context_fuse(x, context)

        if self.dropout:
            x = self.dropout(x, deterministic=deterministic)
        # Second convolutional layer
        x = self.conv2(x)

        # Residual connection
        skip_connection = self.skip_connection(inputs).astype(
            self.preferred_element_type
        )
        out = x + skip_connection
        return out


class SpatialSelfAttention(nnx.Module):
    """Full-image self-attention for (B, H, W, C) tensors."""

    def __init__(
        self,
        in_features: int,
        rngs,
        *,
        num_heads: int = 4,
        attn_size: int | None = None,
        dropout_rate: float = 0.0,
        norm: type[nnx.LayerNorm] | type[nnx.GroupNorm] | None = nnx.GroupNorm,
        mha: type[MultiHeadAttention] | type[nnx.Module] = MultiHeadAttention,
    ):
        self.norm = norm(in_features, rngs=rngs)
        self.attn = mha(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=in_features // num_heads
            if attn_size is None
            else attn_size * num_heads,
            out_features=in_features,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, x: Array, deterministic: bool = True) -> Array:
        """Applies group normalization and multi-head self-attention."""
        *b, h, w, c = x.shape  # (B, H, W, C)
        y = self.norm(x).reshape(*b, h * w, c)  # →  (B, N, C)  with N = H·W
        y = self.attn(y, deterministic=deterministic)  # MultiHeadAttention
        y = y.reshape(*b, h, w, c)
        return x + y


class UNet(nnx.Module):
    """
    Flexible U-Net with pluggable builders for:
      - resnet blocks (default: ResnetBlock)
      - downsampling convs (default: nnx.Conv)
      - upsampling convs (default: nnx.ConvTranspose)
      - attention blocks (default: SpatialSelfAttention, or None to disable)

    You can pass either:
      - *_cls
      - *_factory callables that build a module for a given (in_ch, out_ch).

    Factories take precedence if provided.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Sequence[int],
        *,
        # --- shape/behavior ---
        kernel_size: int | Sequence[int] = 4,
        strides: int | Sequence[int] = 2,
        kernel_size_resnet: int | Sequence[int] = 3,
        strides_resnet: int | Sequence[int] = 1,
        use_attention: bool | Sequence[bool] = False,
        context_features: int | None = None,
        resize_method: str = "bilinear",
        preferred_element_dtype: jnp.dtype | None = None,
        # --- pluggable builders: classes ---
        resnet_block_cls: type[nnx.Module] = ResnetBlock,
        conv_down_cls: type[nnx.Module] | Sequence[type[nnx.Module]] = nnx.Conv,
        conv_up_cls: type[nnx.Module] | Sequence[type[nnx.Module]] = nnx.ConvTranspose,
        attn_cls: type[nnx.Module] = SpatialSelfAttention,
        conv_cls: type[nnx.Module] = nnx.Conv,
        rngs: nnx.Rngs,
    ):
        assert len(out_features) >= 2, "Must have at least 2 output channels"

        self.in_features = in_features
        self.out_features = list(out_features)
        self.num_stages = len(out_features)
        self.resize_method = resize_method  # Triggered if user shapes do not mat
        self.preferred_element_dtype = preferred_element_dtype

        # Normalize/validate attention mask
        if isinstance(use_attention, Sequence):
            assert len(use_attention) == self.num_stages, (
                "`use_attention` list must match number of stages"
            )
            self.attn_mask = list(bool(x) for x in use_attention)
            self.use_attention = any(self.attn_mask)
        else:
            self.attn_mask = [bool(use_attention)] * self.num_stages
            self.use_attention = bool(use_attention)

        # ---------------------------------------------------------------------
        # Builder helpers
        # ---------------------------------------------------------------------
        # ResNet block builder
        _resnet_block = partial(
            resnet_block_cls,
            kernel_size=kernel_size_resnet,
            strides=strides_resnet,
            context_features=context_features,
            rngs=rngs,
        )

        _down_blocks = []
        for i in range(self.num_stages - 1):
            if isinstance(conv_down_cls, Sequence):
                conv_down_cls_i = conv_down_cls[i]
            else:
                conv_down_cls_i = conv_down_cls
            _down_blocks.append(
                partial(
                    conv_down_cls_i,
                    kernel_size=kernel_size,
                    strides=strides,
                    rngs=rngs,
                )
            )

        _up_blocks = []
        for i in range(self.num_stages - 1):
            if isinstance(conv_up_cls, Sequence):
                conv_up_cls_i = conv_up_cls[i]
            else:
                conv_up_cls_i = conv_up_cls
            _up_blocks.append(
                partial(
                    conv_up_cls_i,
                    kernel_size=kernel_size,
                    strides=strides,
                    rngs=rngs,
                )
            )

        # Initial/final projection builders
        _init_final = partial(
            conv_cls,
            kernel_size=1,
            use_bias=False,
            rngs=rngs,
        )

        _attn_block = partial(attn_cls, rngs=rngs)

        # ---------------------------------------------------------------------
        # Down path
        # ---------------------------------------------------------------------
        self.resnet_blocks_down: list[nnx.Module] = []
        self.downsampling_layers: list[nnx.Module] = []
        self.attention_layers_down: list[Optional[nnx.Module]] = []

        for i in range(self.num_stages):
            # ResNet in each stage works on out_features[i]
            self.resnet_blocks_down.append(
                _resnet_block(self.out_features[i], self.out_features[i])
            )
            # Optional attention for this stage
            if self.attn_mask[i]:
                self.attention_layers_down.append(_attn_block(self.out_features[i]))
            else:
                self.attention_layers_down.append(None)

            # Insert a downsample conv between stages (0->1, 1->2, ...)
            if i > 0:
                self.downsampling_layers.append(
                    _down_blocks[i - 1](self.out_features[i - 1], self.out_features[i])
                )

        # ---------------------------------------------------------------------
        # Middle block
        # ---------------------------------------------------------------------
        top_ch = self.out_features[-1]
        self.middle_block1 = _resnet_block(top_ch, top_ch)
        self.middle_block2 = _resnet_block(top_ch, top_ch)
        self.attention_middle = _attn_block(top_ch) if self.attn_mask[-1] else None

        # ---------------------------------------------------------------------
        # Up path (mirror of down)
        # We build resnet_blocks_up to take (ch*2)->ch due to skip concat.
        # Attention order mirrors the down path order.
        # ---------------------------------------------------------------------
        self.resnet_blocks_up: list[nnx.Module] = []
        self.attention_layers_up: list[Optional[nnx.Module]] = []
        self.upsampling_layers: list[nnx.Module] = []

        for i in reversed(range(self.num_stages)):
            ch = self.out_features[i]

            # ResNet block after concatenating skip connection
            self.resnet_blocks_up.append(_resnet_block(ch * 2, ch))

            # Attention layer (mirroring down path)
            if self.attn_mask[self.num_stages - i - 1]:
                self.attention_layers_up.append(_attn_block(ch))
            else:
                self.attention_layers_up.append(None)

            # Upsample conv (skip for bottom-most stage)
            if i > 0:
                self.upsampling_layers.append(
                    _up_blocks[i - 1](ch, self.out_features[i - 1])
                )

        # Initial and final 1x1 projections
        self.conv_initial = _init_final(in_features, self.out_features[0])
        self.conv_final = _init_final(self.out_features[0] * 2, in_features)

    def __call__(
        self,
        inputs: Array,
        context: Optional[Array] = None,
        verbose: bool = False,
        deterministic: bool = True,
    ) -> Array:
        # 1) Initial projection
        x = self.conv_initial(inputs)

        # Stash features before each downsample for skips
        pre_downsampling = [x]

        # 2) Down path
        for i in range(self.num_stages):
            x = self.resnet_blocks_down[i](x, context, deterministic=deterministic)
            if self.attention_layers_down[i] is not None:
                x = self.attention_layers_down[i](x)
            pre_downsampling.append(x)
            if i < self.num_stages - 1:
                if verbose:
                    print("Down:", x.shape)
                x = self.downsampling_layers[i](x).astype(self.preferred_element_dtype)

        # 3) Middle
        x = self.middle_block1(x, context, deterministic=deterministic)
        if self.attention_middle is not None:
            x = self.attention_middle(x)
        x = self.middle_block2(x, context, deterministic=deterministic)
        if verbose:
            print("Mid:", x.shape)

        # 4) Up path
        for idx in range(self.num_stages):
            down = pre_downsampling.pop()
            # Ensure spatial match (covers odd sizes / stride combos)
            if x.shape != down.shape:
                x = jax.image.resize(x, down.shape, method=self.resize_method)
            x = jnp.concatenate([down, x], axis=-1)

            x = self.resnet_blocks_up[idx](x, context, deterministic=deterministic)
            if self.attention_layers_up[idx] is not None:
                x = self.attention_layers_up[idx](x)

            if idx < self.num_stages - 1:
                x = self.upsampling_layers[idx](x).astype(self.preferred_element_dtype)
                if verbose:
                    print("Up:", x.shape)

        # 5) Final projection (+ last skip from very beginning)
        pre_in = pre_downsampling.pop()
        x = jnp.concatenate([pre_in, x], axis=-1)
        x = self.conv_final(x)

        return x
