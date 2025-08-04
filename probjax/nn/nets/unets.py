from functools import partial
from typing import Callable, Optional, Sequence, Union

import jax.numpy as jnp
from flax import nnx
from jax import Array

from probjax.nn.attention import MultiHeadAttention


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs,
        *,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 4,
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

        if num_groups is not None:
            self.group_norm = nnx.GroupNorm(out_features, num_groups, rngs=rngs)
        else:
            self.group_norm = None
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        x = self.conv(x)
        if self.group_norm is not None:
            x = self.group_norm(x)
        x = self.activation(x)
        return x


class ResnetBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs,
        *,
        context_features: Optional[int] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
        activation: Callable = nnx.silu,
        **kwargs,
    ):
        self.activation = activation
        self.context_features = context_features
        self.out_features = out_features
        if context_features is not None:
            self.context_linear = nnx.Linear(context_features, out_features, rngs=rngs)

        _conv_block = partial(
            ConvBlock,
            rngs=rngs,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            num_groups=num_groups,
            activation=activation,
            **kwargs,
        )

        self.conv1 = _conv_block(in_features, out_features)
        self.conv2 = _conv_block(out_features, out_features)

        self.skip_connection = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1 if isinstance(kernel_size, int) else [1] * len(kernel_size),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        # First convolutional layer
        x = self.conv1(inputs)

        # Add context if provided
        if context is not None:
            context = self.context_linear(context)
            context = self.activation(context)
            x = x + context

        # Second convolutional layer
        x = self.conv2(x)

        # Residual connection
        skip_connection = self.skip_connection(inputs)
        out = x + skip_connection
        return out


class UNet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        rngs,
        out_features: Sequence[int] = (32, 64, 128),
        *,
        kernel_size: Union[int, Sequence[int]] = 4,
        strides: Union[int, Sequence[int]] = 2,
        num_groups: int = 16,
        kernel_size_resnet: Union[int, Sequence[int]] = 3,
        strides_resnet: Union[int, Sequence[int]] = 1,
        use_bias: bool = True,
        use_attention: bool = False,
        num_heads: int = 4,
        num_features_qkv: int = 10,
        activation: Callable = nnx.silu,
        **kwargs,
    ):
        self.in_features = in_features
        self.num_stages = len(out_features)
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.num_groups = num_groups
        self.kernel_size_resnet = kernel_size_resnet
        self.strides_resnet = strides_resnet
        self.use_bias = use_bias
        self.use_attention = use_attention
        self.activation = activation
        self.rngs = rngs
        self.kwargs = kwargs

        assert len(out_features) >= 2, "Must have at least 2 output channels"
        assert all(o % num_groups == 0 for o in out_features), (
            "Output channels must be divisible by num_groups!"
        )

        # ---------------------------------------------------------------------
        # Initial large kernel conv
        # ---------------------------------------------------------------------
        self.conv_initial = nnx.Conv(
            in_features=in_features,
            out_features=out_features[0],
            kernel_size=kernel_size + 1
            if isinstance(kernel_size, int)
            else [k + 1 for k in kernel_size],
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

        # ---------------------------------------------------------------------
        # Building blocks
        # ---------------------------------------------------------------------
        _resnet_block = partial(
            ResnetBlock,
            kernel_size=kernel_size_resnet,
            strides=strides_resnet,
            num_groups=num_groups,
            activation=activation,
            rngs=rngs,
            use_bias=use_bias,
            **kwargs,
        )

        _conv_downsampling = partial(
            nnx.Conv,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

        _conv_upsampling = partial(
            nnx.ConvTranspose,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

        # ---------------------------------------------------------------------
        # Optional attention blocks
        # ---------------------------------------------------------------------
        if use_attention:
            initializer = nnx.initializers.variance_scaling(
                len(out_features), 'fan_in', 'truncated_normal'
            )
            _attention = lambda o: MultiHeadAttention(
                num_heads=num_heads,
                in_features=o,
                qkv_features=num_features_qkv * num_heads,
                out_features=o,
                kernel_init=initializer,
                rngs=rngs,
            )
            # We define separate lists for down- and up-path attention
            self.attention_layers_down = []
            self.attention_layers_up = []
            self.layer_norms_down = []
            self.layer_norms_up = []

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
            if use_attention:
                o = out_features[i]
                self.attention_layers_down.append(_attention(o))
                self.layer_norms_down.append(nnx.LayerNorm(out_features[i], rngs=rngs))
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
        if use_attention:
            # single attention for the middle
            self.attention_middle = _attention(out_features[-1])
            self.layer_norm_middle = nnx.LayerNorm(out_features[-1], rngs=rngs)

        # ---------------------------------------------------------------------
        # Up path
        # ---------------------------------------------------------------------
        self.resnet_blocks_up = []
        self.upsampling_layers = []
        for i in range(self.num_stages - 1, 0, -1):
            # Each up block processes (out_features[i]*2) -> out_features[i]
            # because we concatenate skip connections
            self.resnet_blocks_up.append(
                _resnet_block(out_features[i] * 2, out_features[i])
            )
            # Up attention
            if use_attention:
                self.attention_layers_up.append(_attention(out_features[i]))
                self.layer_norms_up.append(nnx.LayerNorm(out_features[i], rngs=rngs))

        # Upsampling conv-transpose
        for i in reversed(range(1, self.num_stages)):
            self.upsampling_layers.append(
                _conv_upsampling(out_features[i], out_features[i - 1])
            )

        # ---------------------------------------------------------------------
        # Final conv
        # ---------------------------------------------------------------------
        self.conv_final = nnx.Conv(
            in_features=out_features[0],
            out_features=in_features,
            kernel_size=1,
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        # ---------------------------------------------------------------------
        # 1) Initial projection
        # ---------------------------------------------------------------------
        x = self.conv_initial(inputs)

        # We’ll keep a list of features before each down-sample for use in the up path.
        pre_downsampling = [x]

        # ---------------------------------------------------------------------
        # 2) Down path
        # ---------------------------------------------------------------------
        for i in range(self.num_stages):
            # ResNet block
            x = self.resnet_blocks_down[i](x, context)
            print(x.shape)
            # Attention
            if self.use_attention:
                _x = x.reshape(x.shape[0], -1, x.shape[-1])
                att = self.attention_layers_down[i](_x)
                att = att.reshape(x.shape)
                x = att + x
                x = self.layer_norms_down[i](x)

            # Save for skip connection
            if i < self.num_stages - 1:
                # Down sample
                x = self.downsampling_layers[i](x)
                pre_downsampling.append(x)

        # ---------------------------------------------------------------------
        # 3) Middle block
        # ---------------------------------------------------------------------
        x = self.middle_block1(x, context)
        if self.use_attention:
            _x = x.reshape(x.shape[0], -1, x.shape[-1])
            att = self.attention_middle(_x)
            att = att.reshape(x.shape)
            x = att + x
            x = self.layer_norm_middle(x)
        x = self.middle_block2(x, context)

        # ---------------------------------------------------------------------
        # 4) Up path
        # ---------------------------------------------------------------------
        # We traverse from top to bottom of the up-sampling path
        for idx in range(self.num_stages - 1):
            # Index of the "top" resnet block we’re in
            up_idx = idx

            # Concatenate with output from downsampling phase
            down = pre_downsampling.pop()
            slices = tuple([slice(0, d) for d in down.shape])
            x = jnp.concatenate([down, x[slices]], -1)

            # ResNet block
            x = self.resnet_blocks_up[up_idx](x, context)

            # Attention
            if self.use_attention:
                _x = x.reshape(x.shape[0], -1, x.shape[-1])
                att = self.attention_layers_up[up_idx](_x)
                att = att.reshape(x.shape)
                x = att + x
                x = self.layer_norms_up[up_idx](x)

            # Upsample (except for the very last iteration)
            if up_idx < len(self.upsampling_layers):
                x = self.upsampling_layers[up_idx](x)

        # ---------------------------------------------------------------------
        # 5) Final projection
        # ---------------------------------------------------------------------
        x = self.conv_final(x)
        return x
