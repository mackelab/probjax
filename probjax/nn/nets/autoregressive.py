from functools import partial
from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.attention import flex_attention
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
        self.bijector_inv = inverse_and_logabsdet(bijector, invertible_arg=1)

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

    def forward(self, x: jax.Array, context=None):
        def scan_fn(carry, i):
            x = carry
            bij_params = self.masked_mlp(x, context)  # type: ignore
            x_new = self.bijector(bij_params, x)
            x = x.at[..., i].set(x_new[i])
            return x, None

        Tx = x
        Tx, _ = jax.lax.scan(scan_fn, Tx, jnp.arange(self.in_out_dim))
        return Tx

    def inverse_and_logdet(self, Tx: jax.Array, context=None):
        bij_params = self.masked_mlp(Tx, context)
        return self.bijector_inv(bij_params, Tx)

    def inverse(self, Tx: jax.Array, context=None):
        bij_params = self.masked_mlp(Tx, context)
        return self.bijector_inv(bij_params, Tx)[0]


class AutoregressiveTransformer(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        in_out_dim: int,
        bijector_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        transformer: Optional[Transformer] = None,
        encoder: Optional[nnx.Module] = None,
        decoder: Optional[nnx.Module] = None,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        attn_size: int = 8,
        widening_factor: int = 2,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        self.bijector_inv = inverse_and_logabsdet(bijector, invertible_arg=1)

        if transformer is None:
            transformer = Transformer(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                attn_size=attn_size,
                widening_factor=widening_factor,
                attention_fn=partial(flex_attention, causal=True),
                rngs=rngs,
                context_dim=context_dim,
            )
        self.transformer = transformer
        self.start_token = nnx.Param(jnp.zeros((self.transformer.model_dim,)))

        if encoder is None:
            encoder = nnx.Linear(
                in_out_dim, self.transformer.model_dim, rngs=rngs, use_bias=False
            )
        if decoder is None:
            decoder = nnx.Linear(
                self.transformer.model_dim,
                bijector_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.zeros,
                use_bias=False,
            )
        self.encoder = encoder
        self.decoder = decoder

    def predict_bij_params(self, x: jax.Array, context=None, k=None, v=None, **kwargs):
        start_token = self.start_token.reshape((1,) * (x.ndim - 1) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, x.shape[:-2] + (1,) + (self.transformer.model_dim,)
        )
        x = self.encoder(x)
        x = jnp.concatenate([start_token, x], axis=-2)
        h = self.transformer(x, k, v, context=context, **kwargs)[..., :-1, :]
        bij_params = self.decoder(h)
        return bij_params

    def __call__(self, x: jax.Array, context=None, k=None, v=None, **kwargs):
        y = autoregressive_transform(x, self, k, v, context, **kwargs)
        return y

    def forward(
        self, x: jax.Array, context=None, k=None, v=None, inverse_impl="naive", **kwargs
    ):
        if inverse_impl == "naive":
            def scan_fn(carry, i):
                x = carry
                print(x.shape)
                bij_params = self.predict_bij_params(x, context, k, v, **kwargs)
                x_new = self.bijector(bij_params, x)
                x = x.at[..., i, :].set(x_new[...,i,:])
                return x, None

            Tx = x
            Tx, _ = jax.lax.scan(scan_fn, Tx, jnp.arange(x.shape[-2]))
            return Tx
        elif inverse_impl == "kv_cache":
            pass
        else:
            raise ValueError(f"Invalid inverse implementation: {inverse_impl}")

    def inverse_and_logdet(self, Tx: jax.Array, context=None, k=None, v=None, **kwargs):
        bij_params = self.predict_bij_params(Tx, context, k, v, **kwargs)
        return self.bijector_inv(bij_params, Tx)

    def inverse(self, Tx: jax.Array, context=None, k=None, v=None, **kwargs):
        return self.inverse_and_logdet(Tx, context, k, v, **kwargs)[0]

@custom_inverse
def autoregressive_transform(x, model, *args, **kwargs):
    Tx = model.forward(x, *args, **kwargs)
    return Tx


def autoregressive_inv_and_logdet(Tx, model, *args, **kwargs):
    return model.inverse_and_logdet(Tx, *args, **kwargs)

def autoregressive_inv(Tx, model, *args, **kwargs):
    return model.inverse(Tx, *args, **kwargs)


# Register inverse
autoregressive_transform.definv(autoregressive_inv)
autoregressive_transform.definv_and_logdet(autoregressive_inv_and_logdet)
