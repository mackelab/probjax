import haiku as hk
import jax
import jax.numpy as jnp

from jaxtyping import Array, PyTree

# B -> batch size
# T -> sequence length
# D -> embedding dimension


# Layer norm of transformers.
def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


class OneHot(hk.Module):
    """One hot encoding module."""

    vocab_size: int  # Size of the vocabulary.
    name: str | None = None  # Optional identifier for the module.

    def __init__(self, vocab_size: int, name: str | None = "one_hot_embed"):
        super().__init__(name=name)
        self.vocab_size = vocab_size

    def __call__(self, x: jax.Array) -> jax.Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.vocab_size)

class PosEmbed(hk.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = jnp.arange(max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
        pe = jnp.zeros((1,max_len, d_model))
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x: Array) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.shape[1]]
        return x


class LearnedPosEmbed(hk.Module):

    def __init__(self, max_len: int, name: str = "learned_pos_embed"):
        super().__init__(name=name)
        self.max_len = max_len
        self.embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        """Embeds the input with learned positional embeddings.

        Args:
            x (Array): Input array of shape [B, T, D]
            max_len (int, optional): Maximum length of the sequence. Defaults to 512.

        Returns:
            Array: Output array of shape [B, T, D]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len <= self.max_len, "Sequence length cannot be greater than max_len"
        positional_embeddings = hk.get_parameter(
            "positional_embeddings", [self.max_len, embed_dim], init=self.embed_init
        )
        positional_embeddings = positional_embeddings[:seq_len, :]
        return x + positional_embeddings[None, :, :]


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
        dropout_rate: float,
        widening_factor: int = 4,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

    def __call__(
        self,
        embeddings: jax.Array,  # [B, T, D]
        mask: jax.Array | None = None,  # [B, T]
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        _, seq_len, model_size = embeddings.shape

        # Compute causal mask for autoregressive sequence modelling.
        if mask is not None:
            mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
            causal_mask = jnp.tril(
                jnp.ones((1, 1, seq_len, seq_len))
            )  # [B=1, H=1, T, T]
            mask = mask * causal_mask  # [B, H=1, T, T]

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
            h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
            h = h + h_dense

        return _layer_norm(h)
