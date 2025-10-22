from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from probjax.utils.typing import Array, Callable

__all__ = [
    # Constants
    "NEG_INF",
    "DEFAULT_MASK_VALUE",
    # Typing helpers
    "MaskModFn",
    "ScoreModFn",
    # Segment helpers
    "ensure_tuple_segment_ids",
    # Block helpers
    "ceil_div",
    "block_pair_has_any",
    "fast_blockmask_local_window",
    "fast_blockmask_causal",
    "row_iterators",
    "compute_block_iterators",
    "compute_kv_iterators",
    "compute_block_bounds",
    # Bias helpers
    "make_segment_bias",
    "compute_padding_biases",
    "alibi_get_slopes",
    # Flash helpers
    "FlashMaskFn",
    # Flash attention utilities
    "get_cpu_dot_precision",
    "get_gpu_dot_precision",
    "build_block_mask",
    "KVOffsetInfo",
    "query_iterator_indices",
    "key_value_iterator_indices",
    "build_sliding_window_mask",
    "get_dropout_mask",
    "segment_mask",
]


# A very negative value used to mask logits safely in float32.
# Keep consistent with pallas_kernels/flash_attention.py
NEG_INF = -1e15
DEFAULT_MASK_VALUE = NEG_INF


# Function type aliases following project typing conventions.
MaskModFn = Callable[
    [Array, Array, Array, Array, Optional[Array], Optional[Array]],
    Array,
]
ScoreModFn = Callable[[Array, Array, Array, Array, Array], Array]

# FlashAttention mask fn signature
FlashMaskFn = Callable[[Array, Array], Array]


def ensure_tuple_segment_ids(
    segment_ids: Optional[Tuple[Array, Array] | Array],
) -> Tuple[Optional[Array], Optional[Array]]:
    if segment_ids is None:
        return None, None
    if isinstance(segment_ids, tuple):
        seg_q, seg_k = segment_ids
        return seg_q, seg_k
    return segment_ids, segment_ids


# ------------------------ Block Sparsity Helpers -----------------------------


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return -(-int(a) // int(b))


def block_pair_has_any(
    mask_mod_fn: MaskModFn,
    b_idx: Array,
    h_idx: Array,
    q_start: int,
    k_start: int,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_k: int,
    seg_q: Optional[Array],
    seg_k: Optional[Array],
) -> Array:
    """True if tile [q_start:q_start+block_q, k_start:k_start+block_k] has any valid entries.

    Indices are clipped in-bounds; mask outside valid rows/cols is zeroed post hoc.
    """
    q_rel = jnp.arange(block_q)
    k_rel = jnp.arange(block_k)
    q_idx = jnp.clip(q_start + q_rel, 0, q_len - 1)
    k_idx = jnp.clip(k_start + k_rel, 0, kv_len - 1)
    base = mask_mod_fn(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k)
    valid_q = (q_start + q_rel) < q_len
    valid_k = (k_start + k_rel) < kv_len
    base = base & valid_q[:, None] & valid_k[None, :]
    return jnp.any(base)


def fast_blockmask_local_window(
    *,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_k: int,
    left_window: int,
    right_window: int,
) -> Array:
    """Fast block mask construction for local window masks.

    A (q_block, kv_block) pair is non-empty iff the intervals intersect:
        [j0, j1] intersects [i0 - left_window, i1 + right_window]
    where i0,i1 are q block bounds, j0,j1 are kv block bounds.
    """
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    i = jnp.arange(num_q_blocks)
    j = jnp.arange(num_kv_blocks)
    i0 = i * block_q
    i1 = jnp.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    j1 = jnp.minimum((j + 1) * block_k - 1, kv_len - 1)
    cond_left = jnp.expand_dims(j0, 0) <= jnp.expand_dims(i1 + right_window, 1)
    cond_right = jnp.expand_dims(j1, 0) >= jnp.expand_dims(i0 - left_window, 1)
    return cond_left & cond_right


def fast_blockmask_causal(
    *, q_len: int, kv_len: int, block_q: int, block_k: int
) -> Array:
    """Fast block mask for causal mask (allow k <= q)."""
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    i = jnp.arange(num_q_blocks)
    j = jnp.arange(num_kv_blocks)
    i1 = jnp.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    return jnp.expand_dims(j0, 0) <= jnp.expand_dims(i1, 1)


def row_iterators(mask_row: Array) -> tuple[Array, Array]:
    """Given a boolean row [N], return (indices, size) with padding.

    - indices: int32[N], listing indices where mask_row is True, padded with 0s.
    - size: int32 scalar, number of valid entries.
    """
    n = mask_row.shape[0]
    idx = jnp.nonzero(mask_row, size=n, fill_value=0)[0]
    size = mask_row.astype(jnp.int32).sum()
    return idx.astype(jnp.int32), size.astype(jnp.int32)


def compute_block_iterators(block_mask: Array) -> tuple[Array, Array]:
    """Per-(QB) iterators over non-empty KV blocks.

    Returns:
        kv_block_offset: int32 [B, H, nQB, nKB]
        kv_block_offset_size: int32 [B, H, nQB]
    """

    idx, sz = jax.vmap(row_iterators)(block_mask)
    return idx, sz


def compute_kv_iterators(block_mask: Array) -> tuple[Array, Array]:
    """Per-(B,H,KB) iterators over non-empty Q blocks.

    Returns:
        q_block_offset: int32 [B, H, nKB, nQB]
        q_block_offset_size: int32 [B, H, nKB]
    """
    block_mask_T = jnp.swapaxes(block_mask, -1, -2)
    idx, sz = jax.vmap(row_iterators)(block_mask_T)
    return idx, sz


def compute_block_bounds(
    *, q_len: int, kv_len: int, block_q: int, block_k: int
) -> tuple[Array, Array, Array, Array]:
    """Returns (q_start, q_end, kv_start, kv_end) as int32 arrays."""
    nQB = ceil_div(q_len, block_q)
    nKB = ceil_div(kv_len, block_k)
    q_start = jnp.arange(nQB, dtype=jnp.int32) * int(block_q)
    q_end = jnp.minimum(q_start + int(block_q) - 1, q_len - 1)
    kv_start = jnp.arange(nKB, dtype=jnp.int32) * int(block_k)
    kv_end = jnp.minimum(kv_start + int(block_k) - 1, kv_len - 1)
    return q_start, q_end, kv_start, kv_end


# ------------------------------ Bias Helpers ---------------------------------


def make_segment_bias(source_segments: Array, target_segments: Array) -> Array:
    """Produces -inf outside same nonzero segment, zeros otherwise.

    Returns bias Array [B, 1, L, L].
    """
    same = source_segments[:, None, :] == target_segments[:, :, None]
    nonzero = (source_segments[:, None, :] != 0) & (target_segments[:, :, None] != 0)
    allowed = same & nonzero
    bias = jnp.where(allowed, 0.0, NEG_INF).astype(jnp.float32)
    return bias[:, None, :, :]


def compute_padding_biases(input_ids: Array, *, pad_token_id: int | None) -> Array:
    """Compute logits bias to disable attention to/from paddings: [B, 1, L, L]."""
    batch_size, seq_len = input_ids.shape
    if pad_token_id is None:
        return jnp.zeros([batch_size, 1, seq_len, seq_len], dtype=jnp.float32)
    padding_bias = (input_ids == pad_token_id) * NEG_INF
    return padding_bias[:, None, None, :] + padding_bias[:, None, :, None]


def alibi_get_slopes(num_heads: int) -> Array:
    """ALiBi head slopes as in the paper/codebase, shape [H]."""
    import math

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    def _slopes_list(n: int) -> list[float]:
        if n <= 0:
            raise ValueError("num_heads must be positive for alibi slopes")
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 1 << int(math.floor(math.log2(n)))
        base = get_slopes_power_of_2(closest_power_of_2)
        extra = _slopes_list(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        return base + extra

    slopes = _slopes_list(num_heads)
    return jnp.asarray(slopes, dtype=jnp.float32)


# -------------------- FlashAttention Utilities -------------------------------


def get_cpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """DotAlgorithmPreset for CPU backend for a given dtype."""
    if dtype == jnp.float32:
        return jax.lax.DotAlgorithmPreset.F32_F32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F16
    if dtype == jnp.bfloat16:
        # bfloat16 not supported
        return jax.lax.DotAlgorithmPreset.F16_F16_F16
    raise ValueError(f"Unsupported dtype {dtype}")


def get_gpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """DotAlgorithmPreset for accelerators; accumulates in FP32 and uses TensorCores."""
    if dtype == jnp.float32:
        return jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F32
    if dtype == jnp.bfloat16:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    raise ValueError(f"Unsupported dtype {dtype}")


def get_tpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """DotAlgorithmPreset for TPU backend; accumulates in FP32."""
    if dtype == jnp.float32:
        return jax.lax.DotAlgorithmPreset.F32_F32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F32
    if dtype == jnp.bfloat16:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    raise ValueError(f"Unsupported dtype {dtype}")


def get_dot_precision(device, dtype) -> jax.lax.DotAlgorithmPreset:
    """DotAlgorithmPreset for current backend; accumulates in FP32."""
    if device == "cpu":
        return get_cpu_dot_precision(dtype)
    if device == "tpu":
        return get_tpu_dot_precision(dtype)
    return get_gpu_dot_precision(dtype)


def build_block_mask(
    mask_fn: FlashMaskFn,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_k: int,
) -> np.ndarray:
    """Build block map where True means the block is not fully masked.

    Uses a separate thread to avoid inheriting sharding contexts during compile-time eval.
    """

    def worker():
        num_q_blocks = ceil_div(q_seq_len, block_q)
        num_kv_blocks = ceil_div(kv_seq_len, block_k)
        block_mask_map = np.ones(shape=(num_q_blocks, num_kv_blocks), dtype=np.bool_)
        for i in range(0, q_seq_len, block_q):
            for j in range(0, kv_seq_len, block_k):
                rows = np.arange(i, i + block_q, dtype=np.int32)
                cols = np.arange(j, j + block_k, dtype=np.int32)
                with jax.ensure_compile_time_eval():
                    if not mask_fn(rows[:, None], cols[None, :]).any():
                        block_mask_map[i // block_q, j // block_k] = False
        return block_mask_map

    with ThreadPoolExecutor(1) as pool:
        return pool.submit(worker).result()


class KVOffsetInfo(NamedTuple):
    kv_block_offset: jax.Array
    kv_block_offset_size: jax.Array


def query_iterator_indices(
    block_mask_map: np.ndarray, *, padding: int = 0
) -> KVOffsetInfo:
    """Per-QBlock iterators over non-empty KV blocks for forward pass."""
    num_q_blocks, num_kv_blocks = block_mask_map.shape
    index_offset = np.full((num_q_blocks, num_kv_blocks), padding, dtype=np.int32)
    index_offset_size = np.zeros(shape=(num_q_blocks), dtype=np.int32)
    for i in range(num_q_blocks):
        k = 0
        for j in range(num_kv_blocks):
            if block_mask_map[i, j]:
                index_offset[i, k] = j
                k += 1
        index_offset_size[i] = k
    return KVOffsetInfo(
        kv_block_offset=jnp.asarray(index_offset),
        kv_block_offset_size=jnp.asarray(index_offset_size),
    )


def key_value_iterator_indices(
    block_mask_map: np.ndarray,
) -> Tuple[jax.Array, jax.Array]:
    """Per-KVBlock iterators over non-empty Q blocks for backward pass."""
    num_q_blocks, num_kv_blocks = block_mask_map.shape
    index_offset = np.zeros(shape=(num_kv_blocks, num_q_blocks), dtype=np.int32)
    index_offset_size = np.zeros(shape=(num_kv_blocks), dtype=np.int32)
    for i in range(num_kv_blocks):
        k = 0
        for j in range(num_q_blocks):
            if block_mask_map[j, i]:
                index_offset[i, k] = j
                k += 1
        index_offset_size[i] = k
    return jnp.asarray(index_offset), jnp.asarray(index_offset_size)


def build_sliding_window_mask(
    *,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_k: int,
    sliding_window_size: int,
) -> np.ndarray:
    """Efficient sliding window causal block mask (np.ndarray) for FlashAttention."""
    bm = fast_blockmask_local_window(
        q_len=q_seq_len,
        kv_len=kv_seq_len,
        block_q=block_q,
        block_k=block_k,
        left_window=int(sliding_window_size),
        right_window=0,
    )
    return np.asarray(bm, dtype=np.bool_)


def get_dropout_mask(shape: tuple[int, ...], *, prng_key: jax.Array, rate: float):
    """Returns a bool dropout mask with True indicating dropout."""
    return jax.random.bernoulli(prng_key, rate, shape)


def segment_mask(q_segment_ids: jax.Array, kv_segment_ids: jax.Array) -> jax.Array:
    """True where query and key positions share the same segment id."""
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=-2)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)
