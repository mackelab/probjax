import functools
import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.nnx import MultiHeadAttention as FlaxMultiHeadAttention
from flax.nnx import combine_masks, dot_product_attention
from flax.nnx.module import first_from
from jax import lax

from probjax.nn.pallas_kernels.attention import BlockSizes, mha
from probjax.nn.pallas_kernels.attention_mask_bias import (
    AttentionBias,
    AttentionMask,
    QKVLengthMask,
)
from probjax.utils.typing import Array, ArrayLike

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
        inputs_k: ArrayLike | None = None,
        inputs_v: ArrayLike | None = None,
        *,
        mask: ArrayLike | None = None,
        bias: ArrayLike | None = None,
        deterministic: bool | None = None,
        rngs=None,
        sow_weights: bool = False,
        decode: bool | None = False,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
        inputs_q: input queries of shape `[batch_sizes..., length, features]`.
        inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
        inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
        mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
        deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic. The
            ``deterministic`` flag passed into the call method will take precedence
            over the ``deterministic`` flag passed into the constructor.
        rngs: rng key. The rng key passed into the call method will take
            precedence over the rng key passed into the constructor.
        sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection.
        decode: whether to prepare and use an autoregressive cache. The ``decode``
            flag passed into the call method will take precedence over the ``decode``
            flag passed into the constructor.

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
        if rngs is None:
            rngs = self.rngs

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    '`inputs_k` cannot be None if `inputs_v` is not None. '
                    'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
                    'value to `inputs_k` and leave `inputs_v` as None.'
                )
            inputs_k = inputs_q
            if inputs_v is None:
                inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f'Incompatible input dimension, got {inputs_q.shape[-1]} '
                f'but module expects {self.in_features}.'
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
                as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    'Autoregressive cache not initialized, call ``init_cache`` first.'
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    'Autoregressive cache shape error, '
                    'expected query shape %s instead got %s.'
                    % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index[...]
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key[...], key, indices)
            value = lax.dynamic_update_slice(self.cached_value[...], value, indices)
            self.cached_key[...] = key
            self.cached_value[...] = value
            self.cache_index[...] += 1
            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
            )

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""No `deterministic` argument was provided to MultiHeadAttention
                    as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided to __call__ method if "
                        "MultiHeadAttention instance is defined with keep_rngs=False."
                    )
                dropout_rng = rngs.dropout()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            bias=bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out

def pad_to_power_of_2(arr: Array, min_size: int = 16, axis=(-1,)) -> Array:
    """Pad the array to the next power of 2 greater than min_size along given axis."""

    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()

    target_shape = list(arr.shape)
    for ax in axis:
        seq_len = arr.shape[ax]
        if seq_len < min_size:
            target_shape[ax] = min_size
        else:
            target_shape[ax] = next_power_of_2(seq_len)

    pad_width = [
        (0, target - current)
        for current, target in zip(arr.shape, target_shape, strict=False)
    ]
    return jnp.pad(arr, pad_width)



def flex_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = False,
    deterministic=True,
    dtype=None,
    precision=None,
    module=None,  # Required arguments by Flax
    sm_scale: Optional[float] = None,
    enable_gqa: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    block_q_dkv: int = 64,
    block_kv_dkv: int = 64,
    block_q_dq: int = 64,
    block_kv_dq: int = 64,
    backward_pass_impl: str = "triton_fused",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
    # These can not be used by the pallas backend
    del (
        module,
        precision,
        broadcast_dropout,
    )

    if dtype is not None:
        query = query.astype(dtype)
        key = key.astype(dtype)
        value = value.astype(dtype)

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
        sm_scale = 1.0 / math.sqrt(query.shape[-1])

    query = query[None] if query.ndim == 3 else query

    _, l_q, h, n = query.shape
    _, l_kv, _, _ = key.shape
    query = pad_to_power_of_2(query, axis=(-3,-1))
    key = pad_to_power_of_2(key, axis=(-3,-1))
    value = pad_to_power_of_2(value, axis=(-3,-1))

    if query.shape[1] != l_q or key.shape[1] != l_kv:
        # Non power-of-2 sequence lengths, hence padding was applied.
        # But this will bias the results, if we don't mask out the padded
        # positions. So we create a mask for the padded positions.
        mask = (
            QKVLengthMask(
                q_length=l_q,
                kv_length=l_kv,
                block_sparse=False,
            )
            if mask is None
            else mask & QKVLengthMask(q_length=l_q, kv_length=l_kv, block_sparse=False)
        )

    mask = jax.tree_util.tree_map(pad_to_power_of_2, mask) if mask is not None else None

    # Compute backward-compatible block sizes (no external BlockSizes input)
    block_sizes = BlockSizes.init_default(
        query.shape[-3],
        key.shape[-3],
        block_q,
        block_k,
        block_q_dkv,
        block_kv_dkv,
        block_q_dq,
        block_kv_dq,
    )
    # Score modifier gradient is handled via bias classes in pallas kernels.

    # If compiling for CPU, enforce interpret mode
    if jax.default_backend() == "cpu" or (
        not isinstance(query, jax.Array) and query.device.platform == "cpu"
    ):
        interpret = True

    if deterministic:
        dropout_rate = 0.0
        dropout_rng = None

    output = mha(
        q=query,
        k=key,
        v=value,
        mask=mask,  # AttentionMaskBase or None
        bias=bias,  # AttentionBiasBase or None
        rng=dropout_rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        dropout_rate=dropout_rate,
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

    assert isinstance(mask, Callable), (
        "Sparse attention requires a (at best sparse) mask, wrapped in a callable"
    )
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
