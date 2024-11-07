from functools import partial
from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from probjax.core.custom_primitives.custom_inverse import custom_inverse


class Sequential(nnx.Module, experimental_pytree=True):
    def __init__(self, *layers):
        """Sequential module.

        Args:
            layers (nnx.Module): List of layers.
        """
        self.layers = layers

    def __call__(self, x, *args, **kwargs) -> Array:
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class Flip(nnx.Module, experimental_pytree=True):
    def __init__(self, axis: int = -1, rngs=None):
        """Flip the array along an axis.

        Args:
            axis (int, optional): Axis to flip. Defaults to -1.
        """
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(nnx.Module, experimental_pytree=True):
    def __init__(self, permutation: Array, axis: int = -1, rngs=None):
        """Permutes the array along an axis.

        Args:
            permutation (Array): An array of indices to permute.
            axis (int, optional): Axis to permute. Defaults to -1.
        """
        self.permutation = nnx.Variable(permutation)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.take(x, self.permutation, axis=self.axis)


class Rotate(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_out_dim: int,
        rngs,
        *,
        rotation_matrix: Optional[Array] = None,
        learnable: bool = False,
    ):
        """Rotate the array.

        Args:
            rotation_matrix (Array): Rotation matrix.
            name (str, optional): Name of the module. Defaults to "rotate".
        """
        self.in_out_dim = in_out_dim
        self.learnable = learnable
        if not learnable:
            if rotation_matrix is None:
                self.rotation_matrix = nnx.Variable(
                    nnx.initializers.orthogonal()(
                        rngs.next(), shape=(in_out_dim, in_out_dim)
                    )
                )

            else:
                self.rotation_matrix = nnx.Variable(rotation_matrix)
        else:
            raise NotImplementedError(
                "Learnable rotation matrix is not implemented yet."
            )
            # TODO: Matrix exponetial of any ske symetric matrix is orthogonal
            # Use for reparameterization

    def __call__(self, x: Array, *args) -> Array:
        if not self.learnable:
            rotation_matrix = jax.lax.stop_gradient(self.rotation_matrix.value)
        return rotate(rotation_matrix, x)


@partial(custom_inverse, inv_argnum=1)
def rotate(R, x):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


class GaussianFourierEmbedding(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rngs,
        *,
        learnable=True,
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "gaussian_fourier_embedding".
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learnable = learnable
        half_dim = self.output_dim // 2 + 1
        if not learnable:
            self.B = nnx.Variable(
                nnx.initializers.normal(1.0)(rngs.next(), shape=(half_dim, input_dim))
            )
        else:
            self.B = nnx.Param(
                nnx.initializers.normal(1.0)(rngs.next(), shape=(half_dim, input_dim))
            )

    def __call__(self, inputs):
        B = self.B.value
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


class OneHot(nnx.Module, experimental_pytree=True):
    """One hot encoding module."""

    def __init__(self, num_tokens: int, rngs=None):
        """Represents a one hot encoding module.

        Args:
            num_tokens (int): Number of distinct tokens.
        """
        self.num_tokens = num_tokens

    def __call__(self, x: Array, *args) -> Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.num_tokens)
