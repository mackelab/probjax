from chex import PRNGKey
from jaxtyping import Array, PyTree


from typing import Any, Callable, NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.scipy.stats as stats
import numpy as np

from functools import partial
import matplotlib.pyplot as plt

import blackjax

from blackjax.base import State, Info


class Params(NamedTuple):
    pass


class MCMCKernel:

    params: Params

    def init_state(self, position: PyTree) -> State:
        raise NotImplementedError("init_state method must be implemented")

    def init_params(self, position: PyTree):
        raise NotImplementedError("init_params method must be implemented")

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[State, Info]:
        raise NotImplementedError("adapt_params method must be implemented")

    def __call__(self, key: PRNGKey, state: State) -> Tuple[State, Info]:
        raise NotImplementedError("__call__ method must be implemented")
