from functools import partial
from typing import Callable, Optional, Sequence

import jax
from jax import Array
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from probjax.nn.nets.mlp import MLP
from probjax.nn.attention import MultiHeadAttention


class PosEmbed(nnx.Module, experimental_pytree=True):
    def __init__(self, token_dim: int, max_seq_len: int = 500, rngs=None):
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
        self.pe = nnx.Variable(pe)

    def __call__(self, x: Array) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe.value[:, : x.shape[1]]
        return x


class LearnedPosEmbed(nnx.Module, experimental_pytree=True):
    def __init__(self, dim: int, max_seq_len: int, rngs):
        self.max_seq_len = max_seq_len
        self.embed = nnx.Embed(max_seq_len, dim, rngs=rngs)

    def __call__(self, x: Array, idx=None, rng=None) -> Array:
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
        idx = jnp.arange(seq_len) if idx is None else idx
        pos_emb = self.embed(idx)
        return x + pos_emb[None, :, :]


class Transformer(nnx.Module, experimental_pytree=True):
    """A transformer stack."""

    in_out_dim: int  # Dimensionality of the embedding vectors.
    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.

    def __init__(
        self,
        in_out_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        rngs: nnx.Rngs,
        *,
        context_dim: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable = jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        initializer: Optional[nnx.initializers.Initializer] = None,
        attention_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.initializer = (
            nnx.initializers.variance_scaling(
                2 / self.num_layers, 'fan_in', 'truncated_normal'
            )
            if initializer is None
            else initializer
        )
        self.act = act
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp

        # Layer norms for the attention and dense blocks.
        self.layer_norms1 = [
            nnx.LayerNorm(in_out_dim, rngs=rngs) for _ in range(num_layers)
        ]
        self.layer_norms2 = [
            nnx.LayerNorm(in_out_dim, rngs=rngs) for _ in range(num_layers)
        ]
        self.out_layer_norm = nnx.LayerNorm(in_out_dim, rngs=rngs)

        # Attention block.
        self.attention_blocks = [
            MultiHeadAttention(
                num_heads,
                in_out_dim,
                attn_size * num_heads,
                in_out_dim,
                rngs=rngs,
                kernel_init=self.initializer,
                dropout_rate=dropout_rate if dropout_rate is not None else 0.0,
            )
            for _ in range(num_layers)
        ]

        # Dense block.
        dims = [in_out_dim] + [widening_factor] * num_hidden_layers + [in_out_dim]
        linear = partial(nnx.Linear, kernel_init=self.initializer)
        self.dense_blocks = [
            MLP(
                dims,
                rngs=rngs,
                linear=linear,
                activation=act,
                activate_final=False,
            )
            for i in range(num_layers)
        ]
        if context_dim is not None:
            self.context_blocks = [
                nnx.Linear(
                    context_dim, in_out_dim, rngs=rngs, kernel_init=self.initializer
                )
                for _ in range(num_layers)
            ]
        else:
            self.context_blocks = None

        if dropout_rate is not None:
            self.dropout_dense = [
                nnx.Dropout(rate=dropout_rate, rngs=rngs) for _ in range(num_layers)
            ]
        else:
            self.dropout_dense = None

    def __call__(
        self,
        inputs: Array,  # [B, T, D]
        context: Optional[Array] = None,  # [B, D_context]
        mask: Array | None = None,  # [T, T] or [B, T, T]
        deterministic: bool = False,
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"Mask must have ndim 2 or 3, got {mask.ndim}.")

        h = inputs

        for i in range(self.num_layers):
            # First the attention block.
            h = self.layer_norms1[i](h)
            h_attn = self.attention_blocks[i](h, mask=mask, deterministic=deterministic)

            h = h + h_attn if self.skip_connection_attn else h_attn

            # Then the dense block.
            h = self.layer_norms2[i](h)
            h_dense = self.dense_blocks[i](h)
            if self.context_dim is not None and context is not None:
                h_dense = h_dense + self.context_blocks[i](context)
            if self.dropout_dense is not None:
                h_dense = self.dropout_dense[i](h_dense, deterministic=deterministic)

            h = h + h_dense if self.skip_connection_mlp else h_dense

        out = self.out_layer_norm(h)

        return out
