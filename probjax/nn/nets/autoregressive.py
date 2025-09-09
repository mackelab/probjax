from functools import partial
from typing import Callable, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.layers.attention import flex_attention
from probjax.nn.nets.masked import MaskedMLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.layers.encoding import PosEncode


def get_autoregressive_masks(dims: Sequence[int]):
    masks = []
    for i in range(len(dims) - 1):
        x1 = jnp.arange(dims[i]).reshape(-1, 1) % dims[0] + 1
        x2 = jnp.arange(dims[i + 1]).reshape(1, -1) % dims[0] + 1

        mask = x2 >= x1 if i != 0 else x2 > x1

        masks.append(mask)
    return masks


class AutoregressiveMLP(nnx.Module):
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
            # Reshape parameters to (batch_dims..., in_out_dim, bijector_dim)
            bij_params = bij_params.reshape(
                bij_params.shape[:-1] + (self.in_out_dim, -1)
            )
            # Get parameters for the i-th dimension using dynamic indexing
            bij_params_i = jax.lax.dynamic_slice(
                bij_params,
                (0,) * (bij_params.ndim - 2) + (i, 0),
                (1,) * (bij_params.ndim - 2) + (1, bij_params.shape[-1]),
            )
            bij_params_i = bij_params_i.reshape(bij_params_i.shape[:-2] + (-1,))
            # Apply bijector to the i-th dimension only
            x_i = jax.lax.dynamic_slice(
                x, (0,) * (x.ndim - 1) + (i,), (1,) * (x.ndim - 1) + (1,)
            )
            x_new_i = self.bijector(bij_params_i, x_i)
            x = x.at[..., i].set(x_new_i[..., 0])
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


class AutoregressiveTransformer(nnx.Module):
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
        num_layers: int = 4,
        attn_size: int = 8,
        widening_factor: int = 2,
        pos_embed: Optional[nnx.Module] = None,
        context_dim: Optional[int] = None,
        **kwargs,
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
                **kwargs,
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
        if pos_embed is None:
            pos_embed = PosEmbed(model_dim, rngs=rngs)
        self.pos_embed = pos_embed

    def predict_bij_params(self, x: jax.Array, context=None, k=None, v=None, **kwargs):
        start_token = self.start_token.reshape((1,) * (x.ndim - 1) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, x.shape[:-2] + (1,) + (self.transformer.model_dim,)
        )
        x = self.encoder(x)
        x = jnp.concatenate([start_token, x], axis=-2)
        x = self.pos_embed(x)
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
                bij_params = self.predict_bij_params(x, context, k, v, **kwargs)
                # Reshape parameters to (batch_dims..., seq_len, bijector_dim)
                bij_params = bij_params.reshape(
                    bij_params.shape[:-1] + (x.shape[-2], -1)
                )
                # Get parameters for the i-th dimension using dynamic indexing
                bij_params_i = jax.lax.dynamic_slice(
                    bij_params,
                    (0,) * (bij_params.ndim - 2) + (i, 0),
                    (1,) * (bij_params.ndim - 2) + (1, bij_params.shape[-1]),
                )
                bij_params_i = bij_params_i.reshape(bij_params_i.shape[:-2] + (-1,))
                # Apply bijector to the i-th dimension only
                x_i = jax.lax.dynamic_slice(
                    x,
                    (0,) * (x.ndim - 2) + (i, 0),
                    (1,) * (x.ndim - 2) + (1, x.shape[-1]),
                )
                x_new_i = self.bijector(bij_params_i, x_i)
                x = x.at[..., i, :].set(x_new_i[..., 0, :])
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


@partial(custom_inverse, static_argnums=(1,))
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
