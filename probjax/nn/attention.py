import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from typing import Callable, Sequence, Optional, Union, Any, Tuple, Iterable

import warnings


class MultiHeadAttention(hk.MultiHeadAttention):
    def __init__(
        self,
        *args,
        save_attention_weights: bool = False,
        attention_method="dense",
        **kwargs,
    ):
        self.save_attention_weights = save_attention_weights
        self.attention_method = attention_method
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        if self.attention_method == "dense":
            attn, attn_weights = dense_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                self.key_size,
                self.save_attention_weights,
            )
        elif self.attention_method == "mem_eff":
            attn = efficient_masked_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                mask,
                self.save_attention_weights,
            )
            attn_weights = None
            return attn
        elif self.attention_method == "sparse":
            attn = efficient_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                mask,
                self.save_attention_weights,
            )
            attn_weights = None
            return attn
        else:
            raise NotImplementedError("Unimplemented attention method")

        if self.save_attention_weights:
            _ = hk.get_state(
                "attn_weights",
                shape=attn_weights.shape,
                dtype=attn_weights.dtype,
                init=hk.initializers.Constant(0.0),
            )
            hk.set_state("attn_weights", attn_weights)

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(
            self.model_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init,
        )
        return final_projection(attn)  # [T', D']


def dense_dot_product_attention(
    query_heads,  # [...,T', H, K]
    key_heads,  # [...,T', H, K]
    value_heads,  # [T, H, V]
    key_size: int,
    mask=None,  # [..., T,T]
    return_attention_weights: bool = False,
):
    *leading_dims, sequence_length, _, _ = query_heads.shape
    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(key_size).astype(key_heads.dtype)

    if mask is not None:
        if mask.ndim != attn_logits.ndim:
            raise ValueError(
                f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                f"{attn_logits.ndim}."
            )
        attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    if return_attention_weights:
        return attn, attn_weights
    else:
        return attn


def efficient_masked_dot_product_attention(
    query_heads,  # [...,T', H, K]
    key_heads,  # [...,T', H, K]
    value_heads,  # [T, H, V]
    indices1,  # Should be the indices where the mask is true
    indices2,
    return_attention_weights: bool = False,
):
    *leading_dims, sequence_length, _, dim = query_heads.shape
    query_heads = jnp.take(
        query_heads, indices1, axis=-3
    )  # [..., E, H, K] Where E is the number of edges
    key_heads = jnp.take(key_heads, indices2, axis=-3)  # [..., E, H, K]
    value_heads = jnp.take(value_heads, indices2, axis=-3)  # [..., E, H, V]

    # Attention logits
    attention_logits = jnp.einsum(
        "...ehd,...ehd->...eh", query_heads, key_heads
    ) / jnp.sqrt(dim).astype(key_heads.dtype)
    attention_logits = attention_logits - jnp.max(
        attention_logits, axis=-2, keepdims=True
    )
    attention_weight = jnp.exp(attention_logits)
    attention_normalizer = jax.ops.segment_sum(
        attention_weight,
        indices1,
        num_segments=sequence_length,
        indices_are_sorted=True,
    )
    attention_normalizer = jnp.take(attention_normalizer, indices1, axis=-2)
    attention_weight = attention_weight / attention_normalizer  # [..., eh]

    # Attention weighted values
    attn = attention_weight[..., None] * value_heads
    attn = jax.ops.segment_sum(
        attn, indices1, num_segments=sequence_length, indices_are_sorted=True
    )
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    if return_attention_weights:
        return attn, attention_weight
    else:
        return attn


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
        elif mask.shape[-1] == 1:
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

    l = math.ceil(num_kv / key_chunk_size)
    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, l, key_chunk_size)
    )
    # l = math.ceil(num_kv / key_chunk_size)
    # overhang = l - num_kv // key_chunk_size

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
    *leading_dims, num_q, num_heads, q_features = query.shape
    num_kv = key.shape[-3]
    l = math.ceil(num_q / query_chunk_size)
    overhang = l - num_q // query_chunk_size

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(leading_dims)
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
                slice_sizes=tuple(leading_dims)
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
        elif mask.shape[-2] == num_q:
            bias_chunk = jax.lax.dynamic_slice(
                bias,
                tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(leading_dims)
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

    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=l)

    if overhang == 0 or l == 1:
        res = jnp.concatenate(res, axis=-3)
        res = jnp.reshape(res, (*leading_dims, num_q, -1))
        return res
    else:
        res_except_last = res[:-1, ..., :, :].reshape(
            (*leading_dims, -1, value.shape[-1])
        )
        res_last = res[-1, ..., overhang:, :, :].reshape(
            (*leading_dims, -1, value.shape[-1])
        )
        res = jnp.concatenate((res_except_last, res_last), axis=-2)
        return res
