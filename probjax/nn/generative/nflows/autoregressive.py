from functools import partial
from typing import Callable, Literal, Mapping, Optional, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet
from probjax.nn.layers.attention import flex_attention
from probjax.nn.layers.encoding import PosEncode
from probjax.nn.layers.ssm import LRUCell
from probjax.nn.nets.ssm import SSMModel
from probjax.nn.nets.simple import MaskedMLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.pallas_kernels import CausalMask
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
            last_kernel = self.masked_mlp.layers[-1].kernel
            last_kernel[...] = jnp.zeros_like(last_kernel[...])

    def predict_bij_params(
        self, x: jax.Array, context=None, *, rng: jax.Array | None = None
    ):
        return self.masked_mlp(x, context, rng=rng)

    def __call__(self, x: jax.Array, context=None, *, rng: jax.Array | None = None):
        y = autoregressive_transform(x, self, context, rng=rng)
        return y

    def forward(self, x: jax.Array, context=None, *, rng: jax.Array | None = None):
        def scan_fn(carry, i):
            x = carry
            bij_params = self.masked_mlp(x, context, rng=rng)  # type: ignore
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

    def inverse_and_logdet(
        self, Tx: jax.Array, context=None, *, rng: jax.Array | None = None
    ):
        """Inverse of a single (unbatched) event; logdet is the scalar total
        over the event dimensions."""
        bij_params = self.masked_mlp(Tx, context, rng=rng)
        bij_params = jnp.reshape(bij_params, Tx.shape + (self.bijector_dim,))
        x, logdet = jax.vmap(self.bijector_inv)(bij_params, Tx)
        return x, jnp.sum(logdet)

    def inverse(self, Tx: jax.Array, context=None, *, rng: jax.Array | None = None):
        return self.inverse_and_logdet(Tx, context, rng=rng)[0]


class AutoregressiveTransformer(nnx.Module):
    def __init__(
        self,
        in_out_dim: int,
        bijector_dim: int,
        bijector: Optional[Callable] = None,
        rngs: Optional[nnx.Rngs] = None,
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
        attention_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        if rngs is None:
            raise ValueError("rngs is required.")
        self.in_out_dim = in_out_dim
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        # A conditioner-only build (see TransformerARConditionerConfig) never
        # invokes the bijection; only predict_bij_params is used.
        self.bijector_inv = (
            inverse_and_logabsdet(bijector, invertible_arg=1)
            if bijector is not None
            else None
        )

        if transformer is None:
            transformer = Transformer(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                attn_size=attn_size,
                widening_factor=widening_factor,
                attention_fn=(
                    attention_fn
                    if attention_fn is not None
                    else partial(flex_attention, mask=CausalMask())
                ),
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

    def predict_bij_params(
        self,
        x: jax.Array,
        context=None,
        k=None,
        v=None,
        *,
        rng: jax.Array | None = None,
        **kwargs,
    ):
        # Causality is the whole contract here: without an explicit mask the
        # transformer's explicit mask=None would override the attention
        # kernel's baked-in default and every position would read the
        # future. The cached decode path is causal by construction, so the
        # full forward must be too, or the two disagree (and the density
        # is invalid).
        if kwargs.get("mask") is None:
            kwargs["mask"] = CausalMask()
        start_token = self.start_token.reshape((1,) * (x.ndim - 1) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, x.shape[:-2] + (1,) + (self.transformer.model_dim,)
        )
        x = self.encoder(x)  # type: ignore
        x = jnp.concatenate([start_token, x], axis=-2)
        x = self.pos_embed(x, rng=rng)  # type: ignore
        h = self.transformer(x, k, v, context=context, rng=rng, **kwargs)[..., :-1, :]  # type: ignore
        bij_params = self.decoder(h)  # type: ignore
        return bij_params

    def init_decode(self, batch_shape, seq_len, dtype=jnp.float32):
        """Reset self-attention KV caches for a cached decode run.

        Shared by the flow ``kv_cache`` forward and autoregressive-model
        sampling: both feed one token per position and read each position's
        parameters back, so both need the same cache lifecycle.
        """
        cache_input_shape = tuple(batch_shape) + (
            seq_len,
            self.transformer.model_dim,
        )
        for block in self.transformer.attention_blocks:
            block.init_cache(cache_input_shape, dtype=dtype)

    def decode_hidden(self, token, pos, context=None, k=None, v=None, **kwargs):
        """Single-token transformer forward with the KV cache.

        ``token`` is a model-space ``(..., 1, model_dim)`` slice (the
        broadcast ``start_token`` at ``pos == 0``); ``pos`` its sequence
        position. Returns ``h`` of the same shape.
        """
        decode_kwargs = dict(kwargs)
        decode_kwargs.pop("decode", None)
        token = self.pos_embed(  # type: ignore
            token,
            idx=jnp.asarray([pos], dtype=jnp.int32),
            rng=decode_kwargs.get("rng"),
        )
        return self.transformer(  # type: ignore
            token, k, v, context=context, decode=True, **decode_kwargs
        )

    def decode_params(self, token, pos, context=None, k=None, v=None, **kwargs):
        """Bijection parameters for one cached-decode position.

        The per-position counterpart to :meth:`predict_bij_params`; returns
        ``(..., bijector_dim)`` for the single position.
        """
        h = self.decode_hidden(token, pos, context, k, v, **kwargs)
        bij_params = self.decoder(h)  # type: ignore
        return bij_params.reshape(bij_params.shape[:-2] + (-1,))

    def __call__(
        self,
        x: jax.Array,
        context=None,
        k=None,
        v=None,
        *,
        rng: jax.Array | None = None,
        **kwargs,
    ):
        # Order must match forward / inverse_and_logdet, which take
        # (x, context, k, v) -- autoregressive_transform forwards *args as-is.
        if self.bijector is None:
            raise ValueError(
                "This AutoregressiveTransformer was built without a bijector "
                "(conditioner-only use); calling it as a transform is not available."
            )
        y = autoregressive_transform(x, self, context, k, v, rng=rng, **kwargs)
        return y

    def forward(
        self, x: jax.Array, context=None, k=None, v=None, inverse_impl="naive", **kwargs
    ):
        if self.bijector is None:
            raise ValueError(
                "This AutoregressiveTransformer was built without a bijector "
                "(conditioner-only use); forward is not available."
            )
        if inverse_impl == "naive":

            def scan_fn(carry, i):
                x = carry
                # (batch_dims..., seq_len, bijector_dim) -- the decoder already
                # emits one parameter block per token.
                bij_params = self.predict_bij_params(x, context, k, v, **kwargs)
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
            seq_len = x.shape[-2]
            batch_shape = x.shape[:-2]
            self.init_decode(batch_shape, seq_len, dtype=x.dtype)

            Tx = x
            start_token = self.start_token.reshape(
                (1,) * len(batch_shape) + (1, self.transformer.model_dim)
            )
            start_token = jnp.broadcast_to(
                start_token,
                batch_shape + (1, self.transformer.model_dim),
            )

            for i in range(seq_len):
                token = start_token
                if i:
                    token = self.encoder(Tx[..., i - 1 : i, :])  # type: ignore
                bij_params_i = self.decode_params(token, i, context, k, v, **kwargs)

                x_i = Tx[..., i : i + 1, :]
                x_new_i = self.bijector(bij_params_i, x_i)
                Tx = Tx.at[..., i, :].set(x_new_i[..., 0, :])

            return Tx
        else:
            raise ValueError(f"Invalid inverse implementation: {inverse_impl}")

    def inverse_and_logdet(self, Tx: jax.Array, context=None, k=None, v=None, **kwargs):
        bij_params = self.predict_bij_params(Tx, context, k, v, **kwargs)
        if self.bijector_inv is None:
            raise ValueError(
                "This AutoregressiveTransformer was built without a bijector "
                "(conditioner-only use); inverse is not available."
            )
        return self.bijector_inv(bij_params, Tx)

    def inverse(self, Tx: jax.Array, context=None, k=None, v=None, **kwargs):
        return self.inverse_and_logdet(Tx, context, k, v, **kwargs)[0]


class AutoregressiveSSM(nnx.Module):
    """Autoregressive conditioner backed by a unidirectional recurrent model."""

    def __init__(
        self,
        in_out_dim: int,
        bijector_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        model: Optional[SSMModel] = None,
        encoder: Optional[nnx.Module] = None,
        decoder: Optional[nnx.Module] = None,
        model_dim: int = 64,
        num_layers: int = 4,
        recurrent_cls: ModuleLikeType = LRUCell,
        recurrent_kwargs: Optional[Mapping] = None,
        context_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.bijector_dim = bijector_dim
        self.bijector = bijector
        self.bijector_inv = inverse_and_logabsdet(bijector, invertible_arg=1)

        if model is None:
            model = SSMModel(
                input_dim=model_dim,
                model_dim=model_dim,
                output_dim=model_dim,
                num_layers=num_layers,
                bidirectional=False,
                recurrent_cls=recurrent_cls,
                recurrent_kwargs=recurrent_kwargs,
                rngs=rngs,
                **kwargs,
            )
        elif model.bidirectional:
            raise ValueError("AutoregressiveSSM requires a unidirectional model")
        self.model = model

        self.start_token = nnx.Param(jnp.zeros((self.model.input_dim,)))
        if encoder is None:
            encoder = nnx.Linear(
                in_out_dim, self.model.input_dim, rngs=rngs, use_bias=False
            )
        if decoder is None:
            decoder = nnx.Linear(
                self.model.output_dim,
                bijector_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.zeros,
                use_bias=False,
            )
        self.encoder = encoder
        self.decoder = decoder
        self.context_proj = (
            nnx.Linear(context_dim, self.model.input_dim, rngs=rngs, use_bias=False)
            if context_dim is not None
            else None
        )

    def predict_bij_params(
        self,
        x: jax.Array,
        context=None,
        *,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
    ):
        start_token = self.start_token.reshape((1,) * (x.ndim - 1) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, x.shape[:-2] + (1, self.model.input_dim)
        )
        x = self.encoder(x)  # type: ignore
        x = jnp.concatenate([start_token, x], axis=-2)
        if context is not None:
            if self.context_proj is None:
                raise ValueError(
                    "context was provided but context_dim is not configured"
                )
            x = x + self.context_proj(context)[..., None, :]  # type: ignore
        h = self.model(x, deterministic=deterministic, rng=rng)[..., :-1, :]
        return self.decoder(h)  # type: ignore

    def __call__(
        self,
        x: jax.Array,
        context=None,
        *,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
    ):
        return autoregressive_transform(
            x, self, context, deterministic=deterministic, rng=rng
        )

    def forward(
        self,
        x: jax.Array,
        context=None,
        *,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
    ):
        def scan_fn(carry, i):
            x = carry
            bij_params = self.predict_bij_params(
                x, context, deterministic=deterministic, rng=rng
            )
            bij_params_i = jax.lax.dynamic_slice(
                bij_params,
                (0,) * (bij_params.ndim - 2) + (i, 0),
                bij_params.shape[:-2] + (1, bij_params.shape[-1]),
            )
            bij_params_i = bij_params_i.reshape(bij_params_i.shape[:-2] + (-1,))
            x_i = jax.lax.dynamic_slice(
                x,
                (0,) * (x.ndim - 2) + (i, 0),
                x.shape[:-2] + (1, x.shape[-1]),
            )
            x_new_i = self.bijector(bij_params_i, x_i)
            x = x.at[..., i, :].set(x_new_i[..., 0, :])
            return x, None

        Tx, _ = jax.lax.scan(scan_fn, x, jnp.arange(x.shape[-2]))
        return Tx

    def inverse_and_logdet(
        self,
        Tx: jax.Array,
        context=None,
        *,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
    ):
        bij_params = self.predict_bij_params(
            Tx, context, deterministic=deterministic, rng=rng
        )
        return self.bijector_inv(bij_params, Tx)

    def inverse(self, Tx: jax.Array, context=None, **kwargs):
        return self.inverse_and_logdet(Tx, context, **kwargs)[0]


@partial(custom_inverse)
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
