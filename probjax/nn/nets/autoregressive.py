from functools import partial
from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.nets.masked import MaskedMLP
from probjax.nn.nets.transformer import Transformer


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

    def predict_bij_params(self, x: jax.Array, context=None):
        return self.masked_mlp(x, context)

    def __call__(self, x: jax.Array, context=None):
        y = autoregressive_transform(x, self, context)
        return y

    def inverse(self, y: jax.Array, context=None):
        def scan_fn(carry, _):
            x, log_det = carry
            bij_params = self.masked_mlp(x, context)  # type: ignore
            bijective_inv = inverse_and_logabsdet(partial(self.bijector, bij_params))
            x, log_det_update = bijective_inv(y)
            return (x, log_det + log_det_update), None

        init_carry = (y, 0.0)
        (x, log_det), _ = jax.lax.scan(
            scan_fn, init_carry, None, length=self.in_out_dim
        )
        return x, log_det


class AutoregressiveTransformer(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        transformer: Transformer,
        bijector: Callable,
        rngs: nnx.Rngs,
    ):
        # NOTE: This will assume that the transform is actually autoregressive
        self.transformer = transformer
        self.bijector = bijector
        self.rngs = rngs

    def predict_bij_params(self, x: jax.Array, context=None, y=None):
        return self.transformer(x, y, y, context=context)

    def __call__(self, x: jax.Array, context=None, y=None):
        y = autoregressive_transform(x, self, context, y)
        return y

    def inverse(self, y: jax.Array, context=None):
        def scan_fn(carry, _):
            x, log_det = carry
            bij_params = self.predict_bij_params(x, context, y)
            bijective_inv = inverse_and_logabsdet(partial(self.bijector, bij_params))
            x, log_det_update = bijective_inv(y)
            return (x, log_det + log_det_update), None

        init_carry = (y, 0.0)
        (x, log_det), _ = jax.lax.scan(scan_fn, init_carry, None, length=y.shape[-1])
        return x, log_det


@custom_inverse
def autoregressive_transform(x, model, *args, **kwargs):
    bij_params = model.predict_bij_params(x, *args, **kwargs)
    y = model.bijector(bij_params, x)
    return y


def autoregressive_inv(y, model, *args, **kwargs):
    return model.inverse(y, *args, **kwargs)


# Register inverse
autoregressive_transform.definv_and_logdet(autoregressive_inv)
