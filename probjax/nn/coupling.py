import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List
from jaxtyping import Array, PyTree


class CouplingMLP(hk.Module):
    def __init__(
        self,
        split_index: int,
        bijector: Callable[[Array, Array], Array],
        num_bijector_params: int,
        context_size: int = 0,
        hidden_dims: List[int] = [
            50,
        ],
        name: str = "coupling_mlp",
        **kwargs,
    ):
        super().__init__(name=name)
        self.split_index = split_index
        self.context_size = context_size
        self.bijector = bijector
        self.num_bijector_params = num_bijector_params
        self._hidden_dims = hidden_dims
        self._mlp_params = kwargs

    def __call__(self, x: Array, context: Array | None = None, *kwargs) -> Array:
        conditionor = hk.nets.MLP(
            [self.split_index + self.context_size]
            + self._hidden_dims
            + [self.num_bijector_params],
            **self._mlp_params,
        )
        x1, x2 = jnp.split(x, [self.split_index], axis=-1)
        y1 = x1
        if context is not None:
            x1 = jnp.concatenate([x1, context], axis=-1)
        params = conditionor(x1)
        y2 = self.bijector(params, x2)
  
        y = jnp.concatenate([y1, y2], axis=-1)
        return y
