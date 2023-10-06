import jax
import jax.numpy as jnp

import haiku as hk

from jaxtyping import Array, PyTree
from typing import Optional

from probjax.nn.utils import MultiHeadAttention

# B -> batch size
# T -> sequence length
# D -> embedding dimension


# Layer norm of transformers.
def _layer_norm(x: Array) -> Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


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
        name: str | None = "transformer",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

    def __call__(
        self,
        embeddings: Array,  # [B, T, D]
        mask: Array | None = None,  # [B, T]
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        _, seq_len, model_size = embeddings.shape

        # Compute causal mask for autoregressive sequence modelling.
        # TODO: Do we need this? If we do not want causal masking ?
        if mask is not None:
            # mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
            mask = mask[None, None, :, :]

        h = embeddings
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.attn_size,
                model_size=model_size,
                w_init=initializer,
            )
            h_norm = _layer_norm(h)
            # Self attention, so key, query and value are all the same.
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)

            if self.dropout_rate is not None:
                h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)

            h = h + h_attn

            # Then the dense block.
            dense_block = hk.Sequential(
                [
                    hk.Linear(self.widening_factor * model_size, w_init=initializer),
                    jax.nn.gelu,
                    hk.Linear(model_size, w_init=initializer),
                ]
            )
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            if self.dropout_rate is not None:
                h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
            h = h + h_dense

        return _layer_norm(h)


class OneHot(hk.Module):
    """One hot encoding module."""

    num_tokens: int  # Size of the vocabulary.
    name: str | None = None  # Optional identifier for the module.

    def __init__(self, num_tokens: int, name: str | None = "one_hot_embed"):
        """_summary_

        Args:
            num_tokens (int): Number of distinct tokens.
            name (str | None, optional): Name of the module. Defaults to "one_hot_embed".
        """
        super().__init__(name=name)
        self.num_tokens = num_tokens

    def __call__(self, x: Array, rng=None) -> Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.num_tokens)


class PosEmbed(hk.Module):
    def __init__(self, token_dim: int, max_seq_len: int = 500):
        """Positional embedding module.

        Args:
            token_dim (int): Dimension of the token embedding.
            max_seq_len (int, optional): Maximal length of the sequence. Defaults to 500.
        """
        super().__init__()
        position = jnp.arange(max_seq_len).reshape(-1, 1)
        div_term = jnp.exp(
            jnp.arange(0, token_dim, 2) * (-jnp.log(10000.0) / token_dim)
        )
        pe = jnp.zeros((1, max_seq_len, token_dim))
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x: Array, rng=None) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.shape[1]]
        return x


class LearnedPosEmbed(hk.Module):
    def __init__(self, max_seq_len: int, name: str = "learned_pos_embed"):
        super().__init__(name=name)
        self.max_seq_len = max_seq_len
        self.embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

    def __call__(self, x: Array, rng=None) -> Array:
        """Embeds the input with learned positional embeddings.

        Args:
            x (Array): Input array of shape [B, T, D]
            max_len (int, optional): Maximum length of the sequence. Defaults to 512.

        Returns:
            Array: Output array of shape [B, T, D]
        """
        _, seq_len, embed_dim = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), "Sequence length cannot be greater than max_len"
        positional_embeddings = hk.get_parameter(
            "positional_embeddings", [self.max_seq_len, embed_dim], init=self.embed_init
        )
        positional_embeddings = positional_embeddings[:seq_len, :]
        return x + positional_embeddings[None, :, :]
