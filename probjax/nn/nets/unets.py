from functools import partial
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Conv, ConvTranspose
from jax import Array

from probjax.nn.attention import MultiHeadAttention
from probjax.nn.utils import AdditiveFuse, AffineFuse, Sequential


class ConvBlock(nnx.Module):
    """A convolutional block with optional normalization and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs,
        *,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
        norm: type[nnx.LayerNorm] | type[nnx.GroupNorm] | None = nnx.GroupNorm,
        activation: Callable = nnx.silu,
        **kwargs,
    ):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
            **kwargs,
        )

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
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class ResizeConv(nnx.Module):
    """Resize input spatially, then apply a convolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        spatial_shape: Sequence[int],
        rngs,
        *,
        resize_method: str = "bilinear",
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
        **kwargs,
    ):
        self.resize_method = resize_method
        self.spatial_shape = spatial_shape
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            rngs=rngs,
            **kwargs,
        )

    def __call__(self, x: Array) -> Array:
        """Resizes input and applies convolution."""
        shape = x.shape
        if shape[-1] != self.conv.in_features:
            raise ValueError(
                f"Input shape {shape} does not match expected in_features"
                f" {self.conv.in_features}"
            )
        if len(shape) > len(self.spatial_shape) + 1:
            raise ValueError(
                f"Input shape {shape} does not match expected spatial shape"
                f" {self.spatial_shape}"
            )
        new_shape = shape[: -len(self.spatial_shape) - 1] + tuple(self.spatial_shape)

        x = jax.image.resize(
            x,
            shape=new_shape,
            method=self.resize_method,
        )
        x = self.conv(x)
        return x


class ResnetBlock(nnx.Module):
    """A residual block with two convolutional layers and optional context."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs,
        *,
        conv_block: type[ConvBlock] | type[nnx.Module] = ConvBlock,
        context_features: Optional[int] = None,
        context_fuse: type[AffineFuse] | type[AdditiveFuse] = AffineFuse,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
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

        if context_features is not None:
            self.context_fuse = context_fuse(out_features, context_features, rngs=rngs)

        _conv_block = partial(
            conv_block,
            rngs=rngs,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            **kwargs,
        )

        self.conv1 = _conv_block(in_features, out_features)
        self.conv2 = _conv_block(
            out_features, out_features, kernel_init=nnx.initializers.zeros
        )

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

        self.skip_connection = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1 if isinstance(kernel_size, int) else [1] * len(kernel_size),
            padding="SAME",
            use_bias=False,
            kernel_init=identity_1x1,
            rngs=rngs,
        )

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        """Forward pass with optional context fusion and skip connection."""
        # First convolutional layer
        x = self.conv1(inputs)
        # Fuse context if provided
        if context is not None:
            x = self.context_fuse(x, context)
        # Second convolutional layer
        x = self.conv2(x)

        # Residual connection
        skip_connection = self.skip_connection(inputs)
        out = x + skip_connection
        return out


class SpatialSelfAttention(nnx.Module):
    """Full-image self-attention for (B, H, W, C) tensors."""

    def __init__(
        self,
        channels: int,
        rngs,
        *,
        num_heads: int = 4,
        qkv_mult: int = 2,  # multiplier for qkv_features
        num_groups: int | None = 32,
        **mha_kw,
    ):
        g = channels if num_groups is None else min(num_groups, channels)
        self.norm = nnx.GroupNorm(channels, g, rngs=rngs)

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=channels,
            qkv_features=qkv_mult * channels,  # ← still easy to tune
            out_features=channels,
            rngs=rngs,
            **mha_kw,  # keep any extra kwargs
        )

    def __call__(self, x: Array) -> Array:
        """Applies group normalization and multi-head self-attention."""
        *b, h, w, c = x.shape  # (B, H, W, C)
        y = self.norm(x).reshape(*b, h * w, c)  # →  (B, N, C)  with N = H·W
        y = self.attn(y)  # MultiHeadAttention
        y = y.reshape(*b, h, w, c)
        return x + y


class UNet(nnx.Module):
    """A configurable U-Net architecture with optional attention."""

    def __init__(
        self,
        in_features: int,
        out_features: Sequence[int],
        rngs,
        *,
        kernel_size: Union[int, Sequence[int]] = 4,
        strides: Union[int, Sequence[int]] = 2,
        kernel_size_resnet: Union[int, Sequence[int]] = 3,
        strides_resnet: Union[int, Sequence[int]] = 1,
        num_layer_final: int = 0,
        use_attention: bool | Sequence[bool] = False,
        attn_kwargs: Optional[dict] = None,
        activation: Callable = nnx.silu,
        resize_method="bilinear",
        norm: type[nnx.LayerNorm] | type[nnx.GroupNorm] | None = nnx.GroupNorm,
        **kwargs,
    ):
        self.in_features = in_features
        self.num_stages = len(out_features)
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_size_resnet = kernel_size_resnet
        self.strides_resnet = strides_resnet
        self.use_attention = use_attention
        self.activation = activation
        self.resize_method = resize_method
        self.rngs = rngs
        self.kwargs = kwargs

        assert len(out_features) >= 2, "Must have at least 2 output channels"

        # ---------------------------------------------------------------------
        # Building blocks
        # ---------------------------------------------------------------------
        _resnet_block = partial(
            ResnetBlock,
            kernel_size=kernel_size_resnet,
            strides=strides_resnet,
            activation=activation,
            rngs=rngs,
            norm=norm,
            **kwargs,
        )

        _conv_downsampling = partial(
            nnx.Conv,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )

        _conv_upsampling = partial(
            nnx.ConvTranspose,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )

        # ---------------------------------------------------------------------
        # Optional attention blocks
        # ---------------------------------------------------------------------
        if isinstance(use_attention, Sequence):
            assert len(use_attention) == self.num_stages, (
                "`use_attention` list must match number of down stages"
            )
            self.attn_mask = list(use_attention)  # stage-wise mask
            self.use_attention = any(self.attn_mask)  # global flag
        else:
            self.attn_mask = [bool(use_attention)] * self.num_stages
            self.use_attention = bool(use_attention)

        if self.use_attention:  # builder for a single block
            initializer = nnx.initializers.variance_scaling(
                len(out_features), "fan_in", "truncated_normal"
            )
            attn_kwargs = {} if attn_kwargs is None else attn_kwargs
            _attention = lambda ch: SpatialSelfAttention(
                ch, rngs=rngs, kernel_init=initializer, **attn_kwargs
            )

        # prepare empty lists (None where not used, to keep the indexing simple)
        self.attention_layers_down: list[Optional[nnx.Module]] = []
        self.attention_layers_up: list[Optional[nnx.Module]] = []

        # ---------------------------------------------------------------------
        # Down path
        # ---------------------------------------------------------------------
        self.resnet_blocks_down = []
        self.downsampling_layers = []
        for i in range(self.num_stages):
            # ResNet block
            self.resnet_blocks_down.append(
                _resnet_block(out_features[i], out_features[i])
            )
            # Attention block (down)
            if self.attn_mask[i]:
                self.attention_layers_down.append(_attention(out_features[i]))
            else:
                self.attention_layers_down.append(None)
            # Down-sample layer (except for the last stage)
            if i > 0:
                self.downsampling_layers.append(
                    _conv_downsampling(out_features[i - 1], out_features[i])
                )

        # ---------------------------------------------------------------------
        # Middle block (resnets + optional attention)
        # ---------------------------------------------------------------------
        self.middle_block1 = _resnet_block(out_features[-1], out_features[-1])
        self.middle_block2 = _resnet_block(out_features[-1], out_features[-1])
        if self.attn_mask[-1]:
            # single attention for the middle
            self.attention_middle = _attention(out_features[-1])

        # ---------------------------------------------------------------------
        # Up path
        # ---------------------------------------------------------------------
        self.resnet_blocks_up = []
        self.upsampling_layers = []
        for i in range(self.num_stages - 1, -1, -1):
            # Each up block processes (out_features[i]*2) -> out_features[i]
            # because we concatenate skip connections
            self.resnet_blocks_up.append(
                _resnet_block(out_features[i] * 2, out_features[i])
            )
            # Up attention
            if self.attn_mask[self.num_stages - i - 1]:
                self.attention_layers_up.append(_attention(out_features[i]))
            else:
                self.attention_layers_up.append(None)

        # Upsampling conv-transpose
        for i in reversed(range(1, self.num_stages)):
            self.upsampling_layers.append(
                _conv_upsampling(out_features[i], out_features[i - 1])
            )

        # Final and initial conv
        self.conv_initial = nnx.Conv(
            in_features=in_features,
            out_features=out_features[0],
            kernel_size=1,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.conv_final = nnx.Conv(
            in_features=out_features[0] * 2,
            out_features=in_features,
            kernel_size=1,
            padding="SAME",
            use_bias=False,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # Final deep layer (if specified)
        if num_layer_final > 0:
            self.net_final = Sequential(*[
                _resnet_block(
                    in_features=in_features,
                    out_features=in_features,
                    norm=nnx.LayerNorm,
                )
                for _ in range(num_layer_final)
            ])
        else:
            self.net_final = None

    def get_default_block():
        """Returns the default block type (not implemented)."""
        pass

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        """Forward pass through the U-Net with optional context and attention."""
        # ---------------------------------------------------------------------
        # 1) Initial projection
        # ---------------------------------------------------------------------
        x = self.conv_initial(inputs)

        # We'll keep a list of features before each down-sample for use in the up path.
        pre_downsampling = [x]

        # ---------------------------------------------------------------------
        # 2) Down path
        # ---------------------------------------------------------------------
        for i in range(self.num_stages):
            # ResNet block
            x = self.resnet_blocks_down[i](x, context)
            # Attention
            if self.use_attention and self.attention_layers_down[i] is not None:
                x = self.attention_layers_down[i](x)

            pre_downsampling.append(x)
            # Down sample
            if i < self.num_stages - 1:
                x = self.downsampling_layers[i](x)

        # ---------------------------------------------------------------------
        # 3) Middle block
        # ---------------------------------------------------------------------
        x = self.middle_block1(x, context)
        if self.attn_mask[-1]:
            x = self.attention_middle(x)
        x = self.middle_block2(x, context)

        # ---------------------------------------------------------------------
        # 4) Up path
        # ---------------------------------------------------------------------
        # We traverse from top to bottom of the up-sampling path
        for idx in range(self.num_stages):
            # Concatenate with output from downsampling phase
            down = pre_downsampling.pop()
            # Depending on strides and input dimension the shapes can slightly
            # mismatch, hence we will ensure that both will have the same dim.
            x = jax.image.resize(x, down.shape, self.resize_method)
            x = jnp.concatenate([down, x], -1)
            # ResNet block
            x = self.resnet_blocks_up[idx](x, context)

            # Attention
            if self.use_attention and self.attention_layers_up[idx] is not None:
                x = self.attention_layers_up[idx](x)

            # Upsample (except for the very last iteration)
            if idx < self.num_stages - 1:
                x = self.upsampling_layers[idx](x)

        # ---------------------------------------------------------------------
        # 5) Final projection
        # ---------------------------------------------------------------------
        pre_in = pre_downsampling.pop()
        x = jnp.concatenate([pre_in, x], axis=-1)
        x = self.conv_final(x)

        if self.net_final is not None:
            x = self.net_final(x, context)
        return x
