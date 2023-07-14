import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List
from jaxtyping import Array, PyTree


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
