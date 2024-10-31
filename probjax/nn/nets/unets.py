from functools import partial
from typing import Callable, Optional, Sequence, Union

from flax import nnx
from jax import Array
import jax.numpy as jnp


class ConvBlock(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs,
        *,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: str = "SAME",
        strides: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
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


class ResnetBlock(nnx.Module, experimental_pytree=True):
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
        self.context_features = context_features
        if context_features is not None:
            self.context_linear = nnx.Linear(context_features, rngs=rngs)

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
            context = self.act(context)
            x = x + context

        # Second convolutional layer
        x = self.conv2(x)

        # Residual connection
        skip_connection = self.skip_connection(inputs)
        out = x + skip_connection
        return out


class UNet(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_features: int,
        rngs,
        out_features: Sequence[int] = [32, 64, 128],
        *,
        kernel_size: Union[int, Sequence[int]] = 4,
        strides: Union[int, Sequence[int]] = 2,
        num_groups: int = 4,
        kernel_size_resnet: Union[int, Sequence[int]] = 3,
        strides_resnet: Union[int, Sequence[int]] = 1,
        use_bias: bool = True,
        use_attention: bool = False,
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
        assert all(
            o % num_groups == 0 for o in out_features
        ), "Output channels must be divisible by num_groups!"

        self.conv_initial = nnx.Conv(
            in_features=in_features,
            out_features=out_features[0],
            kernel_size=kernel_size + 3
            if isinstance(kernel_size, int)
            else [k + 3 for k in kernel_size],
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
            **kwargs,
        )

        _resnet_block = partial(
            ResnetBlock,
            kernel_size=kernel_size_resnet,
            strides=strides_resnet,
            num_groups=num_groups,
            activation=activation,
            rngs=rngs,
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

        # Initialize ResNet blocks and downsampling layers
        self.resnet_blocks_down = []
        self.downsampling_layers = []
        for i in range(0, self.num_stages):
            self.resnet_blocks_down.append(
                _resnet_block(out_features[i], out_features[i])
            )

        for i in range(1, self.num_stages):
            self.downsampling_layers.append(
                _conv_downsampling(out_features[i - 1], out_features[i])
            )

        # Initialize middle block
        self.middle_block1 = _resnet_block(out_features[-1], out_features[-1])
        self.middle_block2 = _resnet_block(out_features[-1], out_features[-1])

        # Initialize upsampling layers
        _conv_upsampling = partial(
            nnx.ConvTranspose,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

        self.resnet_blocks_up = []
        self.upsampling_layers = []
        for i in range(self.num_stages - 1, 0, -1):
            self.resnet_blocks_up.append(
                _resnet_block(out_features[i] * 2, out_features[i])
            )

        for i in reversed(range(1, self.num_stages)):
            self.upsampling_layers.append(
                _conv_upsampling(out_features[i], out_features[i - 1])
            )

        self.conv_final = nnx.Conv(
            in_features=out_features[0],
            out_features=in_features,
            kernel_size=1,
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(self, inputs: Array, context: Optional[Array] = None):
        # Initial convolutional layer, with larger kernel size
        x = self.conv_initial(inputs)

        pre_downsampling = []

        # Downsampling phase
        for i in range(self.num_stages):
            print(x.shape)
            # ResNet blocks
            x = self.resnet_blocks_down[i](x, context)

            # Attention layer
            if self.use_attention:
                att = self.attention(x)
                x = x + att

            # Downsample with strided convolution
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if i != self.num_stages - 1:
                x = self.downsampling_layers[i](x)

        # Middle block
        x = self.middle_block1(x, context)
        if self.use_attention:
            att = self.attention(x)
            x = x + att
        x = self.middle_block2(x, context)

        # Upsampling phase
        for index in range(self.num_stages - 1):
            print(x.shape)
            # Concatenate with output from downsampling phase
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)

            # ResNet blocks
            x = self.resnet_blocks_up[index](x, context)

            if self.use_attention:
                att = self.attention(x)
                x = x + att

            # Upsample with transposed convolution
            if index != len(self.out_features) - 1:
                # # Gurantee that the output is the downsampled input
                # padding = self._get_padding(
                #     input_size=x.shape[1],
                #     output_size=pre_downsampling[-1].shape[1],
                # )
                x = self.upsampling_layers[index](x)

        # Output convolutional layer
        x = self.conv_final(x)
        return x
