from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax.numpy as jnp

from probjax.nn.nets.mlp import MLP


class CouplingMLP(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        split_index: int,
        bij_params_dim: int,
        bijector: Callable,
        rngs,
        *,
        context_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = [50, 50],
        **kwargs,
    ):
        self.context_dim = context_dim if context_dim else 0
        in_dim = split_index
        dims = [in_dim + self.context_dim] + list(hidden_dims) + [bij_params_dim]
        self.conditionor = MLP(dims, rngs=rngs, **kwargs)
        self.split_index = split_index
        self.bijector = bijector

    def __call__(self, x, context=None):
        # Split the input
        x1, x2 = jnp.split(x, [self.split_index], axis=-1)
        # Compute bijector parameters
        if context is not None:
            x1 = jnp.concatenate([x1, context], axis=-1)
        bijector_params = self.conditionor(x1)
        # Apply bijective transformation
        y1 = x1
        y2 = self.bijector(bijector_params, x2)
        return jnp.concatenate([y1, y2], axis=-1)
