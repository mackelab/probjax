from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from probjax.utils.typing import Array

# A very negative value used to mask logits safely in float32.
# Keep consistent with pallas_kernels/flash_attention.py
NEG_INF = -1e15
DEFAULT_MASK_VALUE = NEG_INF


def use_interpret_mode() -> bool:
    """Check if we should use interpret mode for pallas kernels.

    Returns True when running on CPU, which requires interpret mode for pallas.
    """
    return jax.default_backend() == "cpu"


# ------------------------ Block Sparsity Helpers -----------------------------


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return -(-int(a) // int(b))


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
    i = np.arange(num_q_blocks)
    j = np.arange(num_kv_blocks)
    i0 = i * block_q
    i1 = np.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    j1 = np.minimum((j + 1) * block_k - 1, kv_len - 1)
    cond_left = np.expand_dims(j0, 0) <= np.expand_dims(i1 + right_window, 1)
    cond_right = np.expand_dims(j1, 0) >= np.expand_dims(i0 - left_window, 1)
    return cond_left & cond_right


def fast_blockmask_causal(
    *, q_len: int, kv_len: int, block_q: int, block_k: int
) -> Array:
    """Fast block mask for causal mask (allow k <= q)."""
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    i = np.arange(num_q_blocks)
    j = np.arange(num_kv_blocks)
    i1 = np.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    return np.expand_dims(j0, 0) <= np.expand_dims(i1, 1)


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


def get_dropout_mask(shape: tuple[int, ...], *, prng_key: jax.Array, rate: float):
    """Returns a bool dropout mask with True indicating dropout."""
    return jax.random.bernoulli(prng_key, rate, shape)
