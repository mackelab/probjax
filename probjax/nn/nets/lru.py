from typing import Callable, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.nn.nets.simple import MLP
from probjax.nn.layers.lru import LRUBlock


class LRUModel(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        n_layers: int,
        rngs,
        *,
        bidirectional: bool = True,
        dropout: Optional[float] = None,
        norm: nnx.Module = nnx.LayerNorm,
        activation: Callable = jax.nn.gelu,
    ):
        self.bidirectional = bidirectional

        self.in_layer = nnx.Linear(input_dim, model_dim, rngs=rngs)
        self.out_layer = nnx.Linear(model_dim, output_dim, rngs=rngs)
        self.layers = [
            LRUBlock(
                model_dim,
                rngs,
                dropout=dropout,
                norm=norm,
                activation=activation,
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = [
            MLP([model_dim, 2 * model_dim, model_dim], rngs=rngs)
            for _ in range(n_layers)
        ]

    def __call__(self, inputs, *args, **kwargs):
        h = self.in_layer(inputs)
        for i, (layer, mlp) in enumerate(zip(self.layers, self.mlp_layers)):
            if self.bidirectional:
                # Alternate between forward and backward layers
                h = layer(h) if i % 2 == 0 else layer(h[:, ::-1])[:, ::-1]
            else:
                h = layer(h)
            h_new = mlp(h)
            h = h + h_new

        out = self.out_layer(h)
        return out
