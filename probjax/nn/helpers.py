import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List
from jaxtyping import Array, PyTree

from probjax.core.custom_primitives.custom_inverse import custom_inverse


class Flip(hk.Module):
    def __init__(self, axis: int = -1, name: str = "flip"):
        """Flip the array along an axis.

        Args:
            axis (int, optional): Axis to flip. Defaults to -1.
            name (str, optional): Name of the module. Defaults to "flip".
        """
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(hk.Module):
    def __init__(self, permutation: Array, axis: int = -1, name: str = "permute"):
        """Permutes the array along an axis.

        Args:
            permutation (Array): An array of indices to permute.
            axis (int, optional): Axis to permute. Defaults to -1.
            name (str, optional): _description_. Defaults to "permute".
        """
        super().__init__(name=name)
        self.permutation = permutation
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.take(x, self.permutation, axis=self.axis)


@custom_inverse
def rotate(R, x):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


class Rotate(hk.Module):
    def __init__(self, key: Array, output_dim: int, name: str = "rotate"):
        """Rotate the array.

        Args:
            rotation_matrix (Array): Rotation matrix.
            name (str, optional): Name of the module. Defaults to "rotate".
        """
        super().__init__(name=name)
        self.rotation_matrix = jax.random.orthogonal(key, output_dim)

    def __call__(self, x: Array, *args) -> Array:
        return rotate(self.rotation_matrix, x)




class SinusoidalEmbedding(hk.Module):
    def __init__(self, dim=32, name=None):
        super().__init__(name=name)
        self.dim = dim

    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return jnp.squeeze(emb, axis=-2)


class TimeEmbedding(hk.Module):
    def __init__(self, dim=32, name=None):
        super().__init__(name=name)
        self.dim = dim

    def __call__(self, inputs):
        se = SinusoidalEmbedding(self.dim)(inputs)

        # Projecting the embedding into a 128 dimensional space
        x = hk.Linear(self.dim)(se)
        x = jax.nn.gelu(x)
        x = hk.Linear(self.dim)(x)

        return x
