from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.nets.masked import MaskedMLP


def get_autoregressive_masks(dims: Sequence[int]):
    masks = []
    for i in range(len(dims) - 1):
        x1 = jnp.arange(dims[i]).reshape(-1, 1) % dims[0] + 1
        x2 = jnp.arange(dims[i + 1]).reshape(1, -1) % dims[0] + 1

        mask = x2 >= x1 if i != 0 else x2 > x1

        masks.append(mask)
    return masks


class AutoregressiveMLP(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_out_dim: int,
        bijector_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        context_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = [50, 50],
        norm: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        **kwargs,
    ):
        dims = [in_out_dim] + list(hidden_dims) + [in_out_dim * bijector_dim]
        masks = get_autoregressive_masks(dims)
        self.in_out_dim = in_out_dim
        self.bijector = bijector
        self.masked_mlp = MaskedMLP(
            dims,
            masks,
            rngs=rngs,
            context_dim=context_dim,
            norm=norm,
            activation=activation,
            activate_final=activate_final,
            **kwargs,
        )

    def __call__(self, x: jax.Array, context=None):
        y = autoregressive_transform(x, self, context)
        return y

    def inverse(self, y: jax.Array, context=None):
        x = y
        log_det = 0.0
        for _ in range(self.in_out_dim):
            bij_params = self.masked_mlp(x, context)  # type: ignore
            bijective_inv = inverse_and_logabsdet(
                lambda x: self.bijector(bij_params, x)
            )
            x, log_det = bijective_inv(y)
        return x, log_det


@custom_inverse
def autoregressive_transform(x, model, context=None):
    bij_params = model.masked_mlp(x, context)
    y = model.bijector(bij_params, x)
    return y


def autoregressive_inv(y, model, context=None):
    return model.inverse(y, context=context)


# Register inverse
autoregressive_transform.definv_and_logdet(autoregressive_inv)
