from __future__ import annotations

import functools
import inspect
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from probjax.utils.typing import ArrayLike

# A very negative value used to mask logits safely in float32.
# Keep consistent with kernels/attention.py
NEG_INF = -1e15
DEFAULT_MASK_VALUE = NEG_INF


def dot(a, b, precision=None):
    """2D matmul with float32 (int32 for ints) accumulation.

    Replaces ``pl.dot``, removed in JAX 0.12.
    """
    integer = jnp.issubdtype(jnp.promote_types(a.dtype, b.dtype), jnp.integer)
    return jnp.dot(
        a,
        b,
        precision=precision,
        preferred_element_type=jnp.int32 if integer else jnp.float32,
    )


def use_interpret_mode() -> bool:
    """Check if we should use interpret mode for pallas kernels.

    Returns True when running on CPU, which requires interpret mode for pallas.
    """
    return jax.default_backend() == "cpu"


@functools.lru_cache(maxsize=1)
def pallas_call_supports_backend() -> bool:
    """Whether the installed ``pl.pallas_call`` accepts ``backend=``."""
    try:
        return "backend" in inspect.signature(pl.pallas_call).parameters
    except (TypeError, ValueError):
        return False


def _get_first_available_attr(module, *names):
    """Return the first available attribute from *module*."""
    for name in names:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    joined_names = ", ".join(names)
    raise AttributeError(f"{module.__name__} has none of: {joined_names}")


def _coerce_backend_hint_to_compiler_params(
    *,
    backend: str | None,
    compiler_params,
):
    """Map a backend hint to compiler params for newer JAX Pallas APIs."""
    if compiler_params is not None or backend is None or jax.default_backend() != "gpu":
        return compiler_params
    if backend == "triton":
        from jax.experimental.pallas import triton as pltriton

        params_cls = _get_first_available_attr(
            pltriton, "CompilerParams", "TritonCompilerParams"
        )
        return params_cls()
    if backend == "mosaic_gpu":
        from jax.experimental.pallas import mosaic_gpu as plgpu

        params_cls = _get_first_available_attr(
            plgpu, "CompilerParams", "GPUCompilerParams"
        )
        return params_cls()
    return compiler_params


def pallas_call_compat(*args, backend=None, **kwargs):
    """Call ``pl.pallas_call`` across JAX versions with/without ``backend=``."""
    if backend is not None and pallas_call_supports_backend():
        kwargs["backend"] = backend
    elif backend is not None:
        kwargs["compiler_params"] = _coerce_backend_hint_to_compiler_params(
            backend=backend,
            compiler_params=kwargs.get("compiler_params"),
        )
    return pl.pallas_call(*args, **kwargs)


def _bs(index_map, block_shape):
    """Compatibility wrapper for BlockSpec(index_map, block_shape) call sites."""
    return pl.BlockSpec(block_shape=block_shape, index_map=index_map)


def _tpu_compiler_params(*, dimension_semantics):
    """Returns TPU compiler params when running on TPU, else None."""
    if use_interpret_mode() or jax.default_backend() != "tpu":
        return None
    from jax.experimental.pallas import tpu as pltpu

    return pltpu.CompilerParams(dimension_semantics=dimension_semantics)


def _is_hopper_or_newer_gpu() -> bool:
    if jax.default_backend() != "gpu":
        return False
    kind = jax.devices()[0].device_kind.lower()
    return any(
        tag in kind for tag in ("h100", "h200", "b100", "b200", "hopper", "blackwell")
    )


def _prefer_mosaic_gpu() -> bool:
    val = os.environ.get("PROBJAX_PALLAS_PREFER_MOSAIC_GPU", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def _unsafe_enable_mosaic_gpu() -> bool:
    """Explicit opt-in for the unstable Mosaic GPU path."""
    val = os.environ.get("PROBJAX_PALLAS_UNSAFE_ENABLE_MOSAIC_GPU", "0")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _pallas_backend(prefer_mosaic_gpu: bool | None = None) -> str | None:
    """Selects the explicit Pallas backend for the current JAX platform."""
    if prefer_mosaic_gpu is None:
        prefer_mosaic_gpu = _prefer_mosaic_gpu()
    backend = jax.default_backend()
    if backend == "gpu":
        if (
            prefer_mosaic_gpu
            and _unsafe_enable_mosaic_gpu()
            and _is_hopper_or_newer_gpu()
        ):
            return "mosaic_gpu"
        return "triton"
    if backend == "tpu":
        return "mosaic_tpu"
    return None


# ------------------------ Block Sparsity Helpers -----------------------------


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return -(-int(a) // int(b))


def pad_to_multiple(x: jax.Array, axis: int, multiple: int) -> jax.Array:
    """Zero-pad ``x`` along ``axis`` up to a multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}.")
    rem = -x.shape[axis] % multiple
    if rem == 0:
        return x
    pads = [(0, 0)] * x.ndim
    pads[axis] = (0, rem)
    return jnp.pad(x, pads)


def fast_blockmask_local_window(
    *,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_k: int,
    left_window: int,
    right_window: int,
) -> np.ndarray:
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
) -> np.ndarray:
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
    block_mask_map: ArrayLike, *, padding: int = 0
) -> KVOffsetInfo:
    """Per-QBlock iterators over non-empty KV blocks for forward pass."""
    block_mask_map = np.asarray(block_mask_map)
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
