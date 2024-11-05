from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from probjax.nn.attention import MultiHeadAttention
from probjax.nn.nets.mlp import MLP


class PosEmbed(nnx.Module, experimental_pytree=True):
    def __init__(self, token_dim: int, max_seq_len: int = 10_000, rngs=None):
        """Positional embedding module.

        Args:
            token_dim (int): Dimension of the token embedding.
            max_seq_len (int, optional): Maximal length of the sequence. Defaults to 500.
        """
        super().__init__()
        self.max_seq_len = max_seq_len

    def __call__(self, x: Array, idx: Optional[Array] = None, **kwargs) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if idx is None:
            idx = jnp.arange(x.shape[-2]).reshape(-1, 1)

        token_dim = x.shape[-1]
        div_term = jnp.exp(
            jnp.arange(0, token_dim, 2) * (-jnp.log(self.max_seq_len) / token_dim)
        )

        pe = jnp.zeros((1, x.shape[-2], token_dim))
        pe = pe.at[..., 0::2].set(jnp.sin(idx * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(idx * div_term))

        return x + pe


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

    model_dim: int  # Dimensionality of the embedding vectors.
    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.

    def __init__(
        self,
        model_dim: int,
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
        """Initialize a Transformer model.
        Args:
            model_dim (int): The dimension of the model's hidden states.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            attn_size (int): Size of each attention head.
            rngs (nnx.Rngs): Random number generator state.
            context_dim (Optional[int], optional): Dimension of additional context to be
                concatenated with transformer output. If None, no context is used.
                Defaults to None.
            dropout_rate (Optional[float], optional): Dropout rate. If None, no dropout
                is applied. Defaults to None.
            widening_factor (int, optional): Factor by which to increase the dimension
                in the MLP. Defaults to 4.
            num_hidden_layers (int, optional): Number of hidden layers in the MLP block.
                Defaults to 1.
            act (Callable, optional): Activation function. Defaults to jax.nn.gelu.
            skip_connection_attn (bool, optional): Whether to use skip connections in
                attention blocks. Defaults to True.
            skip_connection_mlp (bool, optional): Whether to use skip connections in
                MLP blocks. Defaults to True.
            initializer (Optional[nnx.initializers.Initializer], optional): Weight
                initializer. If None, uses truncated normal with variance scaling.
                Defaults to None.
            attention_fn (Optional[Callable], optional): Custom attention function.
                If None, uses dot product attention. Defaults to None.
        """
        super().__init__()
        self.model_dim = model_dim
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
            nnx.LayerNorm(model_dim, rngs=rngs) for _ in range(num_layers)
        ]
        self.layer_norms2 = [
            nnx.LayerNorm(model_dim, rngs=rngs) for _ in range(num_layers)
        ]
        self.out_layer_norm = nnx.LayerNorm(model_dim, rngs=rngs)

        # Attention block.
        attention_fn = (
            attention_fn if attention_fn is not None else nnx.dot_product_attention
        )
        self.attention_blocks = [
            MultiHeadAttention(
                num_heads,
                model_dim,
                attn_size * num_heads,
                model_dim,
                rngs=rngs,
                kernel_init=self.initializer,
                dropout_rate=dropout_rate if dropout_rate is not None else 0.0,
                attention_fn=attention_fn,
            )
            for _ in range(num_layers)
        ]

        # Dense block.
        context_dim = context_dim if context_dim is not None else 0
        dims = (
            [model_dim + context_dim]
            + [widening_factor * model_dim] * num_hidden_layers
            + [model_dim]
        )
        linear = partial(nnx.Linear, kernel_init=self.initializer)
        self.dense_blocks = [
            MLP(
                dims,
                rngs=rngs,
                linear=linear,
                activation=act,
                activate_final=True,
            )
            for _ in range(num_layers)
        ]

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
        decode: bool = False,
    ) -> Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim == 4:
                mask = mask
            else:
                raise ValueError(f"Mask must have ndim 2 or 3, got {mask.ndim}.")

        h = inputs

        # Same context for each token in the sequence.
        if context is not None:
            context = context.reshape(h.shape[:-2] + (1, self.context_dim))
            context = jnp.repeat(context, h.shape[-2], axis=-2)

        for i in range(self.num_layers):
            # First the attention block.
            h = self.layer_norms1[i](h)
            h_attn = self.attention_blocks[i](
                h, mask=mask, deterministic=deterministic, decode=decode
            )
            h = h + h_attn if self.skip_connection_attn else h_attn

            # Then the dense block.
            h = self.layer_norms2[i](h)
            if context is not None and self.context_dim is not None:
                h_context = jnp.concatenate([h, context], axis=-1)
            else:
                h_context = h
            h_dense = self.dense_blocks[i](h_context)
            if self.dropout_dense is not None:
                h_dense = self.dropout_dense[i](h_dense, deterministic=deterministic)
            h = h + h_dense if self.skip_connection_mlp else h_dense

        h = self.out_layer_norm(h)

        return h
