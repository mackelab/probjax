import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray

from typing import Callable
from jaxtyping import Array


class BaseSDE:
    def __init__(self, drift: Callable, diffusion: Callable, p0) -> None:
        self.drift = drift
        self.diffusion = diffusion
        self.p0 = p0

    def sample(self, key: PRNGKeyArray, t: Array, n: int) -> Array:
        raise NotImplementedError

    def logprob(self, x: Array, t: Array) -> Array:
        raise NotImplementedError
    

