import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List
from jaxtyping import Array, PyTree


class Flip(hk.Module):
    def __init__(self, axis: int = -1, name: str = "reverse"):
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(hk.Module):
    def __init__(self, permutation, name: str = "permute"):
        super().__init__(name=name)
        self.permutation = permutation

    def __call__(self, x: Array, *args) -> Array:
        return x[..., self.permutation]
