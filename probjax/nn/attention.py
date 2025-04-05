import functools
import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.nnx import MultiHeadAttention as FlaxMultiHeadAttention
from flax.nnx import dot_product_attention
from jax.typing import ArrayLike

from probjax.nn.pallas_kernels.attention import BlockSizes, MaskModFn, ScoreModFn, mha

__all__ = [
    "MultiHeadAttention",
    "dot_product_attention_jax",
    "dot_product_attention",
    "memory_efficient_dot_product_attention",
    "sparse_dot_product_attention",
    "flex_attention",
]


class MultiHeadAttention(FlaxMultiHeadAttention):
    def __call__(
        self,
        inputs_q: ArrayLike,
        inputs_k: Optional[ArrayLike] = None,
        inputs_v: Optional[ArrayLike] = None,
        *,
        mask: Optional[ArrayLike] = None,
        deterministic: bool | None = None,
        rngs=None,
        sow_weights: bool = False,
        decode: bool = False,  # This is different from the original implementation
    ):

        return super().__call__(
            inputs_q,
            inputs_k,
            inputs_v,
            mask=mask,
            deterministic=deterministic,
            rngs=rngs,
            sow_weights=sow_weights,
            decode=decode,
        )


def pad_to_power_of_2(arr: ArrayLike, min_size: int = 16) -> ArrayLike:
    """Pad the array to the next power of 2 greater than min_size."""

    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()

    target_shape = list(arr.shape)
    for i in range(-3, 0):
        if target_shape[i] < min_size:
            target_shape[i] = min_size
        else:
            target_shape[i] = next_power_of_2(target_shape[i])

    pad_width = [
        (0, target - current) for current, target in zip(arr.shape, target_shape)
    ]
    return jnp.pad(arr, pad_width)


def flex_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    mask=None,
    bias=None,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    broadcast_dropout=None,
    deterministic=True,
    dtype=None,
    precision=None,
    module=None,  # Required arguments by Flax
    score_mod_fn: ScoreModFn | None = None,
    mask_mod_fn: MaskModFn | None = None,
    sm_scale: Optional[bool] = None,
    enable_gqa: bool = False,
    causal: bool = False,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
    # These can not be used by the pallas backend
    del (
        module,
        dtype,
        precision,
        broadcast_dropout,
        dropout_rate,
        deterministic,
        dropout_rng,
    )
    # Masks must be passed as functions
    if isinstance(mask, Callable):
        mask_mod_fn = mask

    if isinstance(bias, Callable):
        score_mod_fn = bias

    if (query.dtype != key.dtype) or (query.dtype != value.dtype):
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )

    if (query.ndim < 3) or (key.ndim < 3) or (value.ndim < 3):
        raise ValueError(
            f"Expected query, key, and value to all be at least 3 dimensional, but got query.ndim: "
            f"{query.ndim}, key.ndim: {key.ndim}, and value.ndim: {value.ndim} instead."
        )

    if (not enable_gqa) and query.shape[-2] != key.shape[-2]:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.shape[-2]} and Hkv={key.shape[-2]}. "
            f"Try setting enable_gqa=True for GQA."
        )

    if enable_gqa:
        Hq = query.shape[2]
        Hkv = key.shape[2]
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={Hq} and Hkv={Hkv}."
            )

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(query.shape[-1])

    query = query[None] if query.ndim == 3 else query

    _, l_q, h, n = query.shape

    query = pad_to_power_of_2(query)
    key = pad_to_power_of_2(key)
    value = pad_to_power_of_2(value)

    score_mod_fn_grad = None if score_mod_fn is None else jax.grad(score_mod_fn)

    output = mha(
        q=query,
        k=key,
        v=value,
        segment_ids=None,
        sm_scale=sm_scale,
        causal=causal,
        score_mod=score_mod_fn,
        mask_mod=mask_mod_fn,
        score_mod_grad=score_mod_fn_grad,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
    )

    output = output[:, :l_q, :h, :n]

    return output


def dot_product_attention_jax(
    query,
    key,
    value,
    mask=None,
    dtype=None,
    precision=None,
    bias=None,
    local_window_size=None,
    implementation=None,
    is_caual=False,
    query_seq_lengths=None,
    key_value_seq_lengths=None,
    scale=None,
    module=None,
    **kwargs,
):
    if module is not None:
        raise ValueError("Saving attention weights is not supported in JAX backend")

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=1.0 / jnp.sqrt(query.shape[-1]) if scale is None else scale,
        is_causal=is_caual,
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=key_value_seq_lengths,
        local_window_size=local_window_size,
        implementation=implementation,
    )


@partial(jax.jit, static_argnums=(3,))
def sparse_dot_product_attention(
    query_heads,  # [...,T', H, K]
    key_heads,  # [...,T, H, K]
    value_heads,  # [T, H, V]
    mask=None,  # [T', T]
    **kwargs,
):
    """Attention with sparse static mask.

    NOTE: Only efficient for very sparse masks, otherwise use
        dense_dot_product_attention.
    """

    assert isinstance(
        mask, Callable
    ), "Sparse attention requires a (at best sparse) mask, wrapped in a callable"
    assert mask is not None, "Sparse attention requires a (at best sparse) mask"

    *leading_dims, sequence_length, _, dim = query_heads.shape

    indices1, indices2 = np.where(mask())
    query_heads = jnp.take(
        query_heads, indices1, axis=-3, indices_are_sorted=True
    )  # [..., E, H, K] Where E is the number of edges
    key_heads = jnp.take(
        key_heads, indices2, axis=-3, indices_are_sorted=True
    )  # [..., E, H, K]
    value_heads = jnp.take(
        value_heads, indices2, axis=-3, indices_are_sorted=True
    )  # [..., E, H, V]

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

    return attn


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def memory_efficient_dot_product_attention(
    query,  # [..., T', H, K]
    key,  # [..., T, H, K]
    value,  # [..., T, H, V]
    mask=None,  # [..., T', T]
    precision=jax.lax.Precision.DEFAULT,
    query_chunk_size: int = 2048,
    key_chunk_size: int = 2048,
    **kwargs,
):
    """Computes memory efficient dot-product attention given query, key, and value.

    NOTE: Flexattention is way faster then this XLA based implementation.

    Args:
        query: The query tensor of shape (..., num_q, num_heads, q_features).
        key: The key tensor of shape (..., num_k, num_heads, k_features).
        value: The value tensor of shape (..., num_k, num_heads, v_features).
        mask: Optional mask tensor of shape (..., num_q, num_k) or (..., num_q, 1).
        precision: The precision level for computation. Defaults to
            jax.lax.Precision.HIGHEST.
        query_chunk_size: The chunk size for query tensor. Defaults to 512.
        key_chunk_size: The chunk size for key tensor. Defaults to 2048.

    Returns:
        The attention output tensor of shape (..., num_q, -1).
    """
    *leading_dims, num_q, num_heads, q_features = query.shape

    if mask is not None and mask.ndim != query.ndim:
        while mask.ndim < query.ndim:
            mask = mask[None, ...]

    query_chunk_size = greatest_divisor(num_q, query_chunk_size)

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
                tuple([0] * (mask.ndim - 2)) + (chunk_idx, 0),
                slice_sizes=tuple(mask.shape[:-2])
                + (min(query_chunk_size, num_q), mask.shape[-1]),
            )
        else:
            raise TypeError(
                f"mask.shape[-2] == {mask.shape[-2]} must broadcast with "
                f"query.shape[-3] == {num_q}"
            )

        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(
                chunk_idx,
                query_chunk,
                key,
                value,
                mask_chunk,
                precision=precision,
                key_chunk_size=key_chunk_size,
            ),
        )

    l = num_q // query_chunk_size  # noqa: E741
    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=l)

    res = jnp.concatenate(res, axis=-3)
    res = jnp.reshape(res, (*leading_dims, num_q, num_heads, -1))
    return res


def _query_chunk_attention(
    query_idx,
    query,
    key,
    value,
    mask,
    precision,
    key_chunk_size=2048,
):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]

    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    # NOTE: num_kv must be divisible by key_chunk_size
    key_chunk_size = greatest_divisor(num_kv, key_chunk_size)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(chunk_idx, query, key, value, mask):
        attn_weights = jnp.einsum(
            "...qhd,...khd->...qhk", query, key, precision=precision
        )

        if mask is not None:
            mask = jnp.expand_dims(mask, axis=-2)  # [..., T', 1, T]
            attn_weights = jnp.where(mask, attn_weights, -1e30)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "...vhf,...qhv->...qhf", value, exp_weights, precision=precision
        )
        max_score = jnp.squeeze(max_score, axis=-1)
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

        if mask is None:
            mask_chunk = None
        elif mask.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 2)) + (0, chunk_idx),
                slice_sizes=tuple(mask.shape[:-2]) + (mask.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"mask.shape[-1] == {mask.shape[-1]} must broadcast with key.shape[-3]"
                f"== {num_kv}"
            )

        return summarize_chunk(chunk_idx, query, key_chunk, value_chunk, mask_chunk)

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


def greatest_divisor(n, limit):
    for i in range(min(n, limit), 0, -1):
        if n % i == 0:
            return i
    return 1
