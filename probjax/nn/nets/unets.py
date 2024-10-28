from functools import partial
from typing import Callable, Optional, Sequence, Union

import jax
from jax import Array
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from probjax.nn.nets.mlp import MLP
from probjax.nn.attention import MultiHeadAttention


class ConvNDBlock(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]] = 3,
        strides: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
        activation: Callable = nnx.silu,
    ):
        self.conv = nnx.Conv(
            in_features=output_channels,
            features=output_channels,
            kernel_size=kernel_shape,
            strides=strides,
            padding="SAME",
        )

        if num_groups is not None:
            self.group_norm = nnx.GroupNorm(num_groups)
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
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]] = 3,
        strides: Union[int, Sequence[int]] = 1,
        num_groups: Optional[int] = 8,
        activation: Callable = nnx.silu,
    ):
        self.conv1 = ConvNDBlock(
            num_spatial_dims,
            output_channels,
            kernel_shape,
            strides,
            num_groups,
            activation,
        )
        self.conv2 = ConvNDBlock(
            num_spatial_dims,
            output_channels,
            kernel_shape,
            strides,
            num_groups,
            activation,
        )
