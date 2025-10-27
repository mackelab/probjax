from functools import partial
from typing import Callable, Literal, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.layers.attention import flex_attention
from probjax.nn.layers.encoding import PosEncode
from probjax.nn.nets.simple import MaskedMLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.pallas_kernels.attention_mask_bias import CausalMask
from probjax.utils.typing import ModuleLikeType


def get_autoregressive_masks(
    dims: Sequence[int],
    *,
    output_order: Literal["interleaved", "grouped"] = "grouped",
) -> list[jax.Array]:
    masks = []
    if output_order not in {"interleaved", "grouped"}:
        raise ValueError(
            f"Unsupported output_order={output_order}. "
            "Expected one of {'interleaved', 'grouped'}."
        )

    input_dim = dims[0]
    for layer_idx in range(len(dims) - 1):
        x1 = jnp.arange(dims[layer_idx]).reshape(-1, 1) % input_dim + 1
        if output_order == "grouped" and layer_idx == len(dims) - 2:
            out_dim = dims[layer_idx + 1]
            if out_dim % input_dim != 0:
                raise ValueError(
                    "Grouped output_order requires final layer size "
                    "to be a multiple of the input dimension."
                )
            repeats = out_dim // input_dim
            x2_vals = jnp.repeat(jnp.arange(input_dim), repeats)
            x2 = x2_vals.reshape(1, -1) + 1
        else:
            x2 = jnp.arange(dims[layer_idx + 1]).reshape(1, -1) % input_dim + 1

        mask = x2 >= x1 if layer_idx != 0 else x2 > x1
        masks.append(mask)
    return masks


class AutoregressiveMLP(nnx.Module):
    def __init__(
        self,
        in_out_features: int,
        bijector_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        context_features: Optional[int] = None,
        hidden_dims: Sequence[int] = [50, 50],
        norm_cls: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        init_last_layer_to_zero: bool = True,
        mlp_cls: ModuleLikeType = MaskedMLP,
        output_order: Literal["interleaved", "grouped"] = "grouped",
        **kwargs,
    ):
        dims = [in_out_features] + list(hidden_dims) + [in_out_features * bijector_dim]
        masks = get_autoregressive_masks(dims, output_order=output_order)
        self.in_out_features = in_out_features
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        self.bijector_inv = inverse_and_logabsdet(bijector, invertible_arg=1)

        self.masked_mlp = mlp_cls(
            dims,
            masks,
            rngs=rngs,
            context_features=context_features,
            norm_cls=norm_cls,
            activation=activation,
            activate_final=activate_final,
            **kwargs,
        )

        if init_last_layer_to_zero:
            assert hasattr(self.masked_mlp, "layers"), 'mlp_cls must have a "layers"'
            self.masked_mlp.layers[-1].kernel.init = nnx.initializers.zeros
            self.masked_mlp.layers[-1].kernel.value = jnp.zeros_like(
                self.masked_mlp.layers[-1].kernel.value
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
            # Get parameters for the i-th dimension using dynamic indexing
            bij_params_i = jax.lax.dynamic_slice(
                bij_params,
                (0,) * (bij_params.ndim - 1) + (i * self.bijector_dim,),
                bij_params.shape[:-1] + (self.bijector_dim,),
            )
            bij_params_i = bij_params_i.reshape(bij_params_i.shape[:-1] + (-1,))
            # Apply bijector to the i-th dimension only
            x_i = jax.lax.dynamic_slice(
                x, (0,) * (x.ndim - 1) + (i,), x.shape[:-1] + (1,)
            )
            x_new_i = self.bijector(bij_params_i, x_i)
            x = x.at[..., i].set(x_new_i[..., 0])
            return x, None

        Tx = x
        Tx, _ = jax.lax.scan(scan_fn, Tx, jnp.arange(self.in_out_features))
        return Tx

    def inverse_and_logdet(self, Tx: jax.Array, context=None):
        bij_params = self.masked_mlp(Tx, context)
        bij_params = jnp.reshape(bij_params, Tx.shape + (self.bijector_dim,))
        x, logdet = jax.vmap(self.bijector_inv)(bij_params, Tx)
        return x, logdet

    def inverse(self, Tx: jax.Array, context=None):
        print("Hey")
        bij_params = self.masked_mlp(Tx, context)
        bij_params = jnp.reshape(
            bij_params,
            bij_params.shape[:-1]
            + (
                self.in_out_features,
                self.bijector_dim,
            ),
        )
        print(bij_params.shape, Tx.shape)
        x = jax.vmap(self.bijector_inv)(bij_params, Tx)[0]
        return x


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
                attention_fn=partial(flex_attention, mask=CausalMask()),
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
            pos_embed = PosEncode(model_dim, rngs=rngs)
        self.pos_embed = pos_embed

    def predict_bij_params(self, x: jax.Array, context=None, k=None, v=None, **kwargs):
        start_token = self.start_token.reshape((1,) * (x.ndim - 1) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, x.shape[:-2] + (1,) + (self.transformer.model_dim,)
        )
        x = self.encoder(x)  # type: ignore
        x = jnp.concatenate([start_token, x], axis=-2)
        x = self.pos_embed(x)  # type: ignore
        h = self.transformer(x, k, v, context=context, **kwargs)[..., :-1, :]  # type: ignore
        bij_params = self.decoder(h)  # type: ignore
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
