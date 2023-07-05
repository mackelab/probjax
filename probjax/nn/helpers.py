
import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List
from jaxtyping import Array, PyTree

class Flip(hk.Module):
    def __init__(self, axis: int = -1, name: str = "reverse"):
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, x: Array) -> Array:
        return jnp.flip(x, axis=self.axis)