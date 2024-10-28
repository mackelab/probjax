from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike


class MaskedLinear(nnx.Linear):
    def __init__(self, in_features, out_features, mask, rngs, **kwargs):
        super().__init__(in_features, out_features, rngs=rngs, **kwargs)
        if mask.shape != (in_features, out_features):
            raise ValueError("Mask shape must be (in_features, out_features)")
        self.mask = nnx.Variable(mask)

    def __call__(self, inputs):
        kernel = self.kernel.value * self.mask.value
        bias = self.bias.value

        inputs, kernel, bias = nnx.nn.dtypes.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MaskedMLP(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        dims: Sequence[int],
        masks: Sequence[ArrayLike],
        rngs: nnx.Rngs,
        *,
        context_dim: Optional[int] = None,
        norm: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.relu,
        activate_final: bool = False,
        **kwargs,
    ):
        self.layers = [
            MaskedLinear(dims[i], dims[i + 1], masks[i], rngs=rngs, **kwargs)
            for i in range(len(dims) - 1)
        ]
        self.norm = norm
        if norm is not None:
            self.norm_layers = [
                norm(dims[i + 1], rngs=rngs) for i in range(len(dims) - 2)
            ]
        self.activation = activation
        self.activate_final = activate_final
        self.context_dim = context_dim
        if context_dim is not None:
            self.context_blocks = [
                nnx.Linear(context_dim, dims[i], rngs=rngs, **kwargs)
                for i in range(1, len(dims) - 1)
            ]

    def __call__(self, x, context=None):
        h = self.layers[0](x)
        h = self.activation(h)
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](h)
            if self.norm is not None:
                h = self.norm_layers[i - 1](h)
            h = self.activation(h)
            if self.context_dim is not None and context is not None:
                h += self.context_blocks[i - 1](context)

        out = self.layers[-1](h) if len(self.layers) > 1 else h

        if self.activate_final:
            out = self.activation(out)
        return out
