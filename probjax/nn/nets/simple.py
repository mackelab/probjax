from typing import Optional, Sequence

import jax
from flax import nnx

from probjax.nn.utils import AdditiveFuse, AffineFuse


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


class ResNet(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rngs: nnx.Rngs,
        *,
        hidden_dim: int = 50,
        num_hidden_layers: int = 2,
        context_dim: Optional[int] = None,
        linear: nnx.Linear | nnx.LoRALinear | nnx.Module = nnx.Linear,
        context_fuse: AffineFuse | AdditiveFuse = AffineFuse,
        norm: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        **kwargs,
    ):
        self.in_layer = linear(in_dim, hidden_dim, rngs=rngs, **kwargs)
        self.out_layer = linear(hidden_dim, out_dim, rngs=rngs, **kwargs)
        self.hidden_layers = [
            linear(hidden_dim, hidden_dim, rngs=rngs, **kwargs)
            for _ in range(num_hidden_layers)
        ]
        self.norm = norm
        if norm is not None:
            self.norm_layers = [norm(hidden_dim, rngs=rngs) for _ in range(num_hidden_layers)]
        self.activation = activation
        self.activate_final = activate_final

        if context_dim:
            self.context_init = context_fuse(hidden_dim, context_dim, rngs=rngs)
            self.context_layers = [
                context_fuse(hidden_dim, context_dim, rngs=rngs)
                for _ in range(num_hidden_layers)
            ]

    def __call__(self, x, context=None):
        h = self.in_layer(x)
        if context is not None:
            h = self.context_init(h, context)
        h = self.activation(h)
        for i in range(len(self.hidden_layers)):
            h_old = h
            h = self.hidden_layers[i](h)
            if self.norm is not None:
                h = self.norm_layers[i](h)
            h = self.activation(h)
            if context is not None:
                h = self.context_layers[i](h, context)

            h = h + h_old

        out = self.out_layer(h) if len(self.hidden_layers) > 0 else h

        if self.activate_final:
            out = self.activation(out)
        return out
