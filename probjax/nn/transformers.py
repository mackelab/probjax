import jax
import jax.numpy as jnp

import haiku as hk

from jaxtyping import Array, PyTree
from typing import Callable, Optional


# B -> batch size
# T -> sequence length
# D -> embedding dimension


class Transformer(hk.Module):
    """A transformer stack."""

    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
    name: str | None = None  # Optional identifier for the module.

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        dropout_rate: Optional[float] = None,
        widening_factor: int = 4,
        act: Callable = jax.nn.gelu,
        initializer: Optional[hk.initializers.Initializer] = None,
        name: str | None = "transformer",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        if initializer is None:
            initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        self.initializer = initializer
        self.act = act

    def __call__(
        self,
        inputs: Array,  # [B, T, D]
        context: Array,  # [B, D_context]
        mask: Array | None = None,  # [B, T]
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        if mask is not None:
            mask = mask[None, None, :, :]

        h = inputs

        for _ in range(self.num_layers):
            # First the attention block.

            h = self.layer_norm(h)
            h_attn = self.attention_block(h, mask=mask)

            h = h + h_attn

            # Then the dense block.
            h = self.layer_norm(h)
            h_dense = self.dense_block(h, context)
            h = h + h_dense

        out = self.layer_norm(h)

        return out

    @hk.transparent
    def layer_norm(self, x: Array) -> Array:
        """Applies a unique LayerNorm to `x` with default settings."""
        ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        return ln(x)

    @hk.transparent
    def attention_block(self, x: Array, mask: Array | None = None) -> Array:
        attn_block = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.attn_size,
            model_size=x.shape[-1],
            w_init=self.initializer,
        )
        attn = attn_block(x, x, x, mask=mask)

        if self.dropout_rate is not None:
            attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn)

        return attn

    @hk.transparent
    def dense_block(self, x: Array, context: Optional[Array] = None) -> Array:
        model_size = x.shape[-1]

        dense_block = hk.Sequential(
            [
                hk.Linear(self.widening_factor * model_size, w_init=self.initializer),
                self.act,
                hk.Linear(model_size, w_init=self.initializer),
            ]
        )

        x = dense_block(x)
        if self.dropout_rate is not None:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        if context is not None:
            context_emb = hk.Linear(model_size, w_init=self.initializer)(context)
            context_emb = self.act(context_emb)
            while context_emb.ndim < x.ndim:
                context_emb = context_emb[..., None, :]

            x = x + context_emb

        return x
