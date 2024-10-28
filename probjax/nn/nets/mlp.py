from typing import Optional, Sequence

import jax
from flax import nnx


class MLP(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        dims: Sequence[int],
        rngs: nnx.Rngs,
        *,
        linear: nnx.Linear | nnx.LoRALinear | nnx.Module = nnx.Linear,
        norm: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        **kwargs,
    ):
        self.layers = [
            linear(dims[i], dims[i + 1], rngs=rngs, **kwargs)
            for i in range(len(dims) - 1)
        ]
        self.norm = norm
        if norm is not None:
            self.norm_layers = [
                norm(dims[i + 1], rngs=rngs) for i in range(len(dims) - 2)
            ]
        self.activation = activation
        self.activate_final = activate_final

    def __call__(self, x):
        h = self.layers[0](x)
        h = self.activation(h)
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](h)
            if self.norm is not None:
                h = self.norm_layers[i - 1](h)
            h = self.activation(h)

        out = self.layers[-1](h) if len(self.layers) > 1 else h

        if self.activate_final:
            out = self.activation(out)
        return out
