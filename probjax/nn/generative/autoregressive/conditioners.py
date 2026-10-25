"""Conditioners: the causal networks that predict each head's parameters.

Every conditioner maps tokens ``(..., input_dim, width)`` -- one per dimension,
``width`` 1 for continuous data or the class count for one-hot discrete data --
to raw parameters ``(..., input_dim, num_params)``, where position ``i`` depends
on ``x_<i`` only. A config holds the hyperparameters and builds the network
once the model knows ``width`` and ``num_params``.

A conditioner that can decode one position at a time (the transformer, with
its KV cache) also provides ``init_decode`` and ``predict_next``; the model
then samples with the cached loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.generative.nflows.autoregressive import AutoregressiveTransformer
from probjax.nn.generative.nflows.config import (
    WidthRule,
    _resolved_model_dim,
    default_hidden_dims,
    resolve_hidden_dims,
)
from probjax.nn.layers.ssm import LRUCell
from probjax.nn.nets.simple import MaskedMLP
from probjax.nn.nets.ssm import SSMModel
from probjax.utils.jaxutils import where

__all__ = [
    "ARConditionerConfig",
    "MLPARConditionerConfig",
    "SSMARConditionerConfig",
    "TransformerARConditionerConfig",
]


# =============================================================================
# Conditioner configs
# =============================================================================


@runtime_checkable
class ARConditionerConfig(Protocol):
    """Builds a causal network ``(..., d, width) -> (..., d, num_params)``."""

    def build(
        self,
        input_dim: int,
        width: int,
        num_params: int,
        *,
        context_features: Optional[int],
        rngs: nnx.Rngs,
    ) -> nnx.Module: ...



def _autoregressive_masks(dims: Sequence[int], num_vars: int) -> list:
    """MADE masks where the input width may exceed the variable count.

    ``get_autoregressive_masks`` derives the degrees from ``dims[0]``, which
    assumes one input unit per autoregressive variable. That breaks for a
    one-hot encoded discrete model, where each variable occupies ``num_classes``
    adjacent units: every slot would get its own degree and dimension ``i``
    would end up conditioned on part of its own value.
    """
    in_features = dims[0]
    if in_features % num_vars != 0:
        raise ValueError(
            f"input width {in_features} is not a multiple of the "
            f"{num_vars} autoregressive variables."
        )
    width = in_features // num_vars
    # slot j belongs to variable j // width
    in_degrees = jnp.repeat(jnp.arange(num_vars), width) + 1

    masks = []
    for layer_idx in range(len(dims) - 1):
        x1 = (
            in_degrees.reshape(-1, 1)
            if layer_idx == 0
            else jnp.arange(dims[layer_idx]).reshape(-1, 1) % num_vars + 1
        )
        out_dim = dims[layer_idx + 1]
        if layer_idx == len(dims) - 2:
            if out_dim % num_vars != 0:
                raise ValueError(
                    "final layer size must be a multiple of the variable count."
                )
            x2 = (
                jnp.repeat(jnp.arange(num_vars), out_dim // num_vars).reshape(1, -1) + 1
            )
        else:
            x2 = jnp.arange(out_dim).reshape(1, -1) % num_vars + 1
        # strict at the input so dimension i never sees x_i
        masks.append(x2 > x1 if layer_idx == 0 else x2 >= x1)
    return masks


class _MaskedConditioner(nnx.Module):
    """MADE: a masked MLP over the flattened tokens."""

    def __init__(self, net: nnx.Module, input_dim: int, num_params: int):
        self.net = net
        self.input_dim = input_dim
        self.num_params = num_params

    def __call__(self, tokens, context=None, *, rng=None):
        flat = tokens.reshape(tokens.shape[:-2] + (-1,))
        out = self.net(flat, context, rng=rng)
        return out.reshape(out.shape[:-1] + (self.input_dim, self.num_params))


@dataclass
class MLPARConditionerConfig(ARConditionerConfig):
    """Masked MLP conditioner (MADE).

    Uses ``get_autoregressive_masks`` and ``MaskedMLP`` directly rather than
    ``AutoregressiveMLP``, whose constructor requires an invertible bijector and
    eagerly builds its inverse -- machinery an AR density head has no use for.

    Only the ``"grouped"`` mask order is used, which puts dimension ``i``'s
    parameters in a contiguous block at ``[i * p : (i + 1) * p]``.
    """

    hidden_dims: "Sequence[int] | WidthRule" = default_hidden_dims
    activation: Callable = jax.nn.gelu
    norm_cls: Optional[type] = None
    init_last_layer_to_zero: bool = True
    #: Training-set size, when known; the width rule scales with it.
    num_examples: Optional[int] = None


    def build(self, input_dim, width, num_params, *, context_features, rngs):
        in_features = input_dim * width
        out_features = input_dim * num_params
        hidden = resolve_hidden_dims(
            self.hidden_dims, in_features, out_features, self.num_examples
        )
        dims = [in_features] + list(hidden) + [out_features]
        masks = _autoregressive_masks(dims, input_dim)
        net = MaskedMLP(
            dims,
            masks,
            rngs=rngs,
            context_features=context_features,
            norm_cls=self.norm_cls,
            activation=self.activation,
        )
        if self.init_last_layer_to_zero:
            last_kernel = net.layers[-1].kernel
            last_kernel[...] = jnp.zeros_like(last_kernel[...])
        return _MaskedConditioner(net, input_dim, num_params)


class _TokenConditioner(nnx.Module):
    """Causal transformer over one token per dimension, with a KV cache."""

    def __init__(self, inner: AutoregressiveTransformer, input_dim: int):
        self.inner = inner
        self.input_dim = input_dim

    def __call__(self, tokens, context=None, *, rng=None):
        return self.inner.predict_bij_params(tokens, context, rng=rng)

    def init_decode(self, batch_shape, dtype=jnp.float32):
        """Reset the inner transformer's KV caches for a decode run."""
        self.inner.init_decode(tuple(batch_shape), self.input_dim, dtype=dtype)

    def predict_next(self, prev_token, pos, context=None, *, rng=None):
        """Raw parameters ``(..., num_params)`` for position ``pos``.

        ``prev_token`` ``(..., width)`` is the previous position's token; at
        ``pos == 0`` its values are ignored (the learned start token is used)
        and only its batch shape is read.

        Written without Python branching on ``pos`` so the loop compiles
        under ``nnx.scan`` as well as running eagerly.
        """
        inner = self.inner
        start = jnp.broadcast_to(
            inner.start_token.reshape(
                (1,) * (prev_token.ndim - 1) + (1, inner.transformer.model_dim)
            ),
            prev_token.shape[:-1] + (1, inner.transformer.model_dim),
        )
        token = where(
            jnp.asarray(pos) == 0,
            start,
            inner.encoder(prev_token[..., None, :]),
        )
        return inner.decode_params(token, pos, context, rng=rng)


@dataclass
class TransformerARConditionerConfig(ARConditionerConfig):
    """Causal-transformer conditioner.

    Wraps :class:`AutoregressiveTransformer` purely for its
    ``predict_bij_params`` path; it is built conditioner-only
    (``bijector=None``), so the bijection is unavailable by construction.
    """

    model_dim: "int | WidthRule" = 256
    num_examples: Optional[int] = None
    num_heads: int = 8
    num_layers: int = 4
    attn_size: int = 32
    widening_factor: int = 4
    #: Attention kernel for the inner transformer. Defaults to the fused
    #: ``flex_attention`` path; pass ``dot_product_attention`` for exact
    #: softmax on CPU, where the fused kernel runs in slow interpret mode.
    attention_fn: Optional[Callable] = None


    def build(self, input_dim, width, num_params, *, context_features, rngs):
        transformer_kwargs: dict = {}
        if self.attention_fn is not None:
            transformer_kwargs["attention_fn"] = self.attention_fn
        inner = AutoregressiveTransformer(
            width,
            num_params,
            None,  # conditioner-only build; only predict_bij_params is used
            rngs,
            context_dim=context_features,
            model_dim=_resolved_model_dim(self, 1, num_params),
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            widening_factor=self.widening_factor,
            **transformer_kwargs,
        )
        return _TokenConditioner(inner, input_dim)


class _SSMConditioner(nnx.Module):
    """Unidirectional SSM over one token per dimension.

    Shift-right with a learned start token, then a causal recurrent stack:
    position ``i`` can only have seen ``x_<i`` by construction, so unlike the
    masked MLP and the transformer no mask is needed.
    """

    def __init__(self, model: SSMModel, width: int, num_params: int, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(width, model.input_dim, rngs=rngs)
        self.start_token = nnx.Param(jnp.zeros((model.input_dim,)))
        self.model = model
        self.decoder = nnx.Linear(
            model.output_dim,
            num_params,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
        )

    def __call__(self, tokens, context=None, *, rng=None):
        h = self.encoder(tokens)
        start = jnp.broadcast_to(
            self.start_token.reshape((1,) * (h.ndim - 1) + (-1,)),
            h.shape[:-2] + (1, h.shape[-1]),
        )
        h = jnp.concatenate([start, h[..., :-1, :]], axis=-2)
        return self.decoder(self.model(h))


@dataclass
class SSMARConditionerConfig(ARConditionerConfig):
    """Unidirectional state-space conditioner.

    A recurrent stack (default: :class:`LRUCell`) over right-shifted tokens.
    Sampling uses the naive ancestral loop -- there is no KV cache to carry,
    so ``use_cache`` falls back to it automatically.
    """

    model_dim: int = 64
    num_layers: int = 2
    #: Recurrent cell class, called as ``recurrent_cls(model_dim, rngs=...,
    #: **recurrent_kwargs)``. :class:`LRUCell` and :class:`MambaCell` both fit.
    recurrent_cls: Optional[Callable] = None
    recurrent_kwargs: Mapping = field(default_factory=dict)


    def build(self, input_dim, width, num_params, *, context_features, rngs):
        del input_dim
        if context_features is not None:
            raise ValueError("SSMARConditionerConfig does not support context yet.")
        model_dim = int(self.model_dim)
        model = SSMModel(
            input_dim=model_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            num_layers=self.num_layers,
            bidirectional=False,
            recurrent_cls=self.recurrent_cls or LRUCell,
            recurrent_kwargs=dict(self.recurrent_kwargs),
            rngs=rngs,
        )
        return _SSMConditioner(model, width, num_params, rngs)
