import jax
import jax.numpy as jnp

import haiku as hk

from jaxtyping import Array, PyTree

# B -> batch size
# T -> sequence length
# D -> embedding dimension


# Layer norm of transformers.
def _layer_norm(x: Array) -> Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


def _query_chunk_attention(
    query_idx,
    query,
    key,
    value,
    mask,
    bias,
    precision,
    key_chunk_size=4096,
    mask_calc_fn=None,
    bias_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
):
    """Efficient dot-product attention for a single query chunk.

    Based on https://arxiv.org/pdf/2112.05682v2.pdf .

    """

    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(chunk_idx, query, key, value, mask, bias):
        attn_weights = jnp.einsum(
            "...qhd,...khd->...qhk", query, key, precision=precision
        )
        if bias_calc_fn is not None:
            bias = bias_calc_fn(query_idx, chunk_idx, bias, attn_weights, calc_fn_data)
        if bias is not None:
            bias = jnp.einsum("...hqk->...qhk", bias)
            attn_weights = attn_weights + bias
        if mask_calc_fn is not None:
            mask = mask_calc_fn(query_idx, chunk_idx, mask, attn_weights, calc_fn_data)
        if mask is not None:
            big_neg = jnp.finfo(attn_weights.dtype).min
            mask = jnp.einsum("...hqk->...qhk", mask)
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        if weights_calc_fn is not None:
            attn_weights = weights_calc_fn(
                query_idx, chunk_idx, attn_weights, calc_fn_data
            )
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "...vhf,...qhv->...qhf", value, exp_weights, precision=precision
        )
        max_score = jnp.einsum("...qhk->...qh", max_score)
        return exp_values, exp_weights.sum(axis=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key,
            tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features),
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(value.shape[:-3])
            + (key_chunk_size, num_heads, v_features),
        )

        if bias is None:
            bias_chunk = None
        elif bias.shape[-1] == 1:
            bias_chunk = bias
        elif bias.shape[-1] == num_kv:
            bias_chunk = jax.lax.dynamic_slice(
                bias,
                tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(bias.shape[:-3])
                + (bias.shape[-3], bias.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}"
            )

        if mask is None:
            mask_chunk = None
        elif bias.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(mask.shape[:-3])
                + (mask.shape[-3], mask.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"mask.shape[-1] == {mask.shape[-1]} must broadcast with key.shape[-3] == {num_kv}"
            )

        return summarize_chunk(
            chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk
        )

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def efficient_dot_product_attention(
    query,
    key,
    value,
    mask=None,
    bias=None,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size=1024,
    key_chunk_size=4096,
    bias_calc_fn=None,
    mask_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
):
    """Computes efficient dot-product attention given query, key, and value.
    This is efficient version of attention presented in
    https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
    Note: query, key, value needn't have any batch dimensions.
    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of
        `[batch..., kv_length, num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      query_chunk_size: int: query chunks size
      key_chunk_size: int: key chunks size
      bias_calc_fn: a bias calculation callback for each chunk, of form
        `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask_calc_fn: a mask calculation callback for each chunk, of form
        `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
        This can be used for incorporating causal or other large masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      weights_calc_fn: a general attn_weights callback for each chunk, of form
        `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
        attn_weights has shape of
        `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
        This can be used to implement complex weights processing in a memory
        efficient way.
      calc_fn_data: optional pure data to pass to each per-chunk call of
        bias_calc_fn, mask_calc_fn, and weights_calc_fn.
      precision: numerical precision of the computation see `jax.lax.Precision`
              for details.
    Returns:
      Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(query.shape[:-3])
            + (min(query_chunk_size, num_q), num_heads, q_features),
        )

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(mask.shape[:-3])
                + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]),
            )
        else:
            raise TypeError(
                f"mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}"
            )

        if bias is None:
            bias_chunk = None
        elif mask.shape[-2] == 1:
            bias_chunk = bias
        elif bias.shape[-2] == num_q:
            bias_chunk = jax.lax.dynamic_slice(
                bias,
                tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(bias.shape[:-3])
                + (bias.shape[-3], min(query_chunk_size, num_q), bias.shape[-1]),
            )
        else:
            raise TypeError(
                f"bias.shape[-2] == {bias.shape[-2]} must broadcast with query.shape[-3] == {num_q}"
            )

        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(
                chunk_idx,
                query_chunk,
                key,
                value,
                mask_chunk,
                bias_chunk,
                precision=precision,
                key_chunk_size=key_chunk_size,
                bias_calc_fn=bias_calc_fn,
                mask_calc_fn=mask_calc_fn,
                weights_calc_fn=weights_calc_fn,
                calc_fn_data=calc_fn_data,
            ),
        )

    _, res = jax.lax.scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )
    return jnp.concatenate(res, axis=-3)


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
