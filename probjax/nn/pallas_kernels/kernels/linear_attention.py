# Some of the code in this file is adapted from:
# Apple/axlearn
# https://github.com/apple/axlearn/tree/main/axlearn/common/rattention

"""Residual linear attention reference and GPU Pallas implementations."""

from __future__ import annotations

import enum
import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental import pallas as pl

from probjax.utils.jaxutils import where

from ..kernel_utils import pallas_call_compat, use_interpret_mode


class FeatureMap(enum.Enum):
    """Feature maps supported by residual linear attention."""

    SOFTMAX = "softmax"
    RELU = "relu"


class FeatureMapFn(NamedTuple):
    fwd: Callable[[jax.Array], jax.Array]
    bwd: Callable[[jax.Array, jax.Array], jax.Array]


def _inner_float32(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        dtype = args[0].dtype
        return fn(*(arg.astype(jnp.float32) for arg in args)).astype(dtype)

    return wrapper


@_inner_float32
def _softmax(x: jax.Array) -> jax.Array:
    x = x - jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


@_inner_float32
def _softmax_bwd(y: jax.Array, dy: jax.Array) -> jax.Array:
    return y * (dy - jnp.sum(dy * y, axis=-1, keepdims=True))


@_inner_float32
def _softmax2(x: jax.Array) -> jax.Array:
    return jnp.concatenate((_softmax(x), _softmax(-x)), axis=-1)


@_inner_float32
def _softmax2_bwd(y: jax.Array, dy: jax.Array) -> jax.Array:
    y_pos, y_neg = jnp.split(y, 2, axis=-1)
    dy_pos, dy_neg = jnp.split(dy, 2, axis=-1)
    return _softmax_bwd(y_pos, dy_pos) - _softmax_bwd(y_neg, dy_neg)


@_inner_float32
def _relu2(x: jax.Array) -> jax.Array:
    return jnp.concatenate((jax.nn.relu(x), jax.nn.relu(-x)), axis=-1)


@_inner_float32
def _relu2_bwd(y: jax.Array, dy: jax.Array) -> jax.Array:
    y_pos, y_neg = jnp.split(y, 2, axis=-1)
    dy_pos, dy_neg = jnp.split(dy, 2, axis=-1)
    return where(y_pos > 0, dy_pos, 0) - where(y_neg > 0, dy_neg, 0)


def get_feature_map(feature_map: FeatureMap | str) -> FeatureMapFn:
    feature_map = FeatureMap(feature_map)
    if feature_map is FeatureMap.SOFTMAX:
        return FeatureMapFn(_softmax2, _softmax2_bwd)
    if feature_map is FeatureMap.RELU:
        return FeatureMapFn(_relu2, _relu2_bwd)
    raise ValueError(f"Unknown feature map: {feature_map}")


def right_shift_and_zero_pad(x: jax.Array, shift_size: int, axis: int = 2) -> jax.Array:
    """Right-shift ``x`` and fill the vacated positions with zeros."""
    if shift_size < 0:
        raise ValueError("shift_size must be non-negative")
    if shift_size == 0:
        return x
    if x.shape[axis] <= shift_size:
        return jnp.zeros_like(x)
    padding_shape = list(x.shape)
    padding_shape[axis] = shift_size
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(0, x.shape[axis] - shift_size)
    return jnp.concatenate(
        (jnp.zeros(padding_shape, dtype=x.dtype), x[tuple(slices)]), axis=axis
    )


def _repeat_kv_heads(x: jax.Array, num_heads: int) -> jax.Array:
    num_kv_heads = x.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    return jnp.repeat(x, num_heads // num_kv_heads, axis=1)


def _linear_attention_scan_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    feature_map: FeatureMap,
) -> tuple[jax.Array, jax.Array]:
    """Reference recurrence after window shifting."""
    feature_fn = get_feature_map(feature_map).fwd
    q = feature_fn(q)
    k = feature_fn(k)
    k = _repeat_kv_heads(k, q.shape[1])
    v = _repeat_kv_heads(v, q.shape[1])
    dtype = v.dtype

    q = jnp.moveaxis(q, 2, 0)
    k = jnp.moveaxis(k, 2, 0)
    v = jnp.moveaxis(v, 2, 0)

    def step(state, inputs):
        q_t, k_t, v_t = inputs
        state = state + jnp.einsum(
            "bhk,bhv->bhkv", k_t, v_t, preferred_element_type=jnp.float32
        )
        output = jnp.einsum(
            "bhk,bhkv->bhv", q_t, state, preferred_element_type=jnp.float32
        )
        return state, output.astype(dtype)

    final_state, output = jax.lax.scan(step, h0.astype(jnp.float32), (q, k, v))
    return jnp.moveaxis(output, 0, 2), final_state


def _linear_attention_chunked_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    feature_map: FeatureMap,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunk-parallel form of :func:`_linear_attention_scan_core`.

    Splits the sequence into chunks of ``chunk_size`` tokens. Within a chunk the
    causal contribution is a masked ``[c, c]`` matmul; across chunks only the
    ``[F, D]`` chunk sums ``K_n^T V_n`` are prefix-summed. This never builds the
    per-token state history that the step scan (and its VJP) stores, so reverse
    mode keeps ``O(L / c * F * D + L * c)`` residuals instead of ``O(L * F * D)``.
    Grouped-query heads are handled by reshaping queries to
    ``[B, H_kv, G, ...]`` instead of repeating keys/values.
    """
    batch, num_heads, seq_len, _ = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    groups = num_heads // num_kv_heads
    value_dim = v.shape[-1]
    dtype = v.dtype
    feature_fn = get_feature_map(feature_map).fwd
    # Same feature dtype as the scan reference (computed in float32 inside).
    q_features = feature_fn(q)
    k_features = feature_fn(k)
    feature_dim = q_features.shape[-1]

    chunk = max(1, min(chunk_size, seq_len))
    num_chunks = -(-seq_len // chunk)
    padded = num_chunks * chunk
    if padded != seq_len:
        # Zero-padded keys/values add nothing to the state; padded query rows
        # are dropped. Pad after the feature map so padded keys are exactly 0.
        pad = [(0, 0), (0, 0), (0, padded - seq_len), (0, 0)]
        q_features = jnp.pad(q_features, pad)
        k_features = jnp.pad(k_features, pad)
        v = jnp.pad(v, pad)

    if jax.default_backend() == "cpu" and dtype != jnp.float32:
        # XLA:CPU has no bf16 x bf16 -> f32 dot; low precision buys nothing there.
        q_features, k_features, v = (
            x.astype(jnp.float32) for x in (q_features, k_features, v)
        )
    compute_dtype = v.dtype

    qc = q_features.reshape(batch, num_kv_heads, groups, num_chunks, chunk, feature_dim)
    kc = k_features.reshape(batch, num_kv_heads, num_chunks, chunk, feature_dim)
    vc = v.reshape(batch, num_kv_heads, num_chunks, chunk, value_dim)
    state0 = h0.astype(jnp.float32).reshape(
        batch, num_kv_heads, groups, feature_dim, value_dim
    )

    f32 = jnp.float32
    # float32 inputs use full-precision dots: on GPU the default (TF32) costs
    # ~3 mantissa digits, and the feature-map backward (softmax/relu of +/-
    # halves) cancels strongly in dk -- with TF32 its relative error grows to
    # ~8e-2 at L=8192 vs ~5e-4 here. On an RTX 3090 this costs <= ~10% time.
    precision = jax.lax.Precision.HIGHEST if compute_dtype == f32 else None
    einsum = functools.partial(
        jnp.einsum, preferred_element_type=f32, precision=precision
    )
    chunk_kv = einsum("bjncf,bjncd->bjnfd", kc, vc)
    total = jnp.sum(chunk_kv, axis=2)
    # Exclusive prefix sum: state entering chunk n (excluding h0).
    prefix = jnp.cumsum(chunk_kv, axis=2) - chunk_kv

    q32 = qc.astype(f32)
    inter = einsum("bjgncf,bjgfd->bjgncd", q32, state0) + einsum(
        "bjgncf,bjnfd->bjgncd", q32, prefix
    )
    scores = einsum("bjgncf,bjnsf->bjgncs", qc, kc)
    causal = jnp.tril(jnp.ones((chunk, chunk), dtype=bool))
    scores = jnp.where(causal, scores, 0.0).astype(compute_dtype)
    intra = einsum("bjgncs,bjnsd->bjgncd", scores, vc)
    output = (inter + intra).astype(dtype)
    output = output.reshape(batch, num_heads, padded, value_dim)[:, :, :seq_len]
    final_state = state0 + total[:, :, None]
    return output, final_state.reshape(h0.shape)


@functools.partial(
    jax.jit, static_argnames=("window_size", "feature_map", "chunk_size")
)
def residual_linear_attention_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    *,
    window_size: int,
    feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Chunked pure-JAX residual linear attention (any backend, any length).

    Same math as the Pallas kernel, expressed as batched XLA matmuls plus a
    prefix sum over chunk states. Default on CPU and the GPU fallback when no
    Pallas chunk fits in shared memory; also used as the Pallas backward.
    """
    feature_map = FeatureMap(feature_map)
    shift = window_size + 1
    k = right_shift_and_zero_pad(k, shift)
    v = right_shift_and_zero_pad(v, shift)
    return _linear_attention_chunked_core(q, k, v, h0, feature_map, chunk_size)


@functools.partial(jax.jit, static_argnames=("window_size", "feature_map"))
def residual_linear_attention_scan(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    *,
    window_size: int,
    feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
) -> tuple[jax.Array, jax.Array]:
    """Backend-independent reference residual linear attention.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout. Keys and values
    are shifted past the local window so this branch only summarizes tokens
    not covered by local attention.
    """
    feature_map = FeatureMap(feature_map)
    shift = window_size + 1
    k = right_shift_and_zero_pad(k, shift)
    v = right_shift_and_zero_pad(v, shift)
    return _linear_attention_scan_core(q, k, v, h0, feature_map)


def _linear_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    h0_ref,
    output_ref,
    final_state_ref,
    *,
    chunk_size: int,
):
    """One-program-per-head chunked kernel, safe for unordered GPU grids.

    ``q_ref``/``k_ref`` hold feature-mapped queries/keys (float32). The feature
    maps concatenate positive and negative halves, which Triton cannot lower,
    so they are applied outside the kernel.
    """
    seq_len = q_ref.shape[0]
    state = h0_ref[:, :].astype(jnp.float32)
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))

    def chunk_step(chunk_index, current_state):
        start = chunk_index * chunk_size
        q_features = pallas_primitives.load(
            q_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        k_features = pallas_primitives.load(
            k_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        v = pallas_primitives.load(
            v_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        inter = jnp.dot(q_features, current_state)
        intra = jnp.dot(q_features, k_features.T) * causal
        output = inter + jnp.dot(intra, v)
        pallas_primitives.store(
            output_ref,
            (pl.dslice(start, chunk_size), slice(None)),
            output.astype(output_ref.dtype),
        )
        return current_state + jnp.dot(k_features.T, v)

    state = jax.lax.fori_loop(0, seq_len // chunk_size, chunk_step, state)
    final_state_ref[:, :] = state


# Fallback when the device cannot be queried: the smallest opt-in shared memory
# per block among current NVIDIA GPUs (consumer Ampere/Ada, 99 KB).
_DEFAULT_SMEM_BYTES = 101376


def _kernel_smem_bytes(
    chunk_size: int, feature_dim: int, head_dim: int, value_dtype
) -> int:
    """Shared memory Triton requests for one program.

    Fitted to measured requests on an RTX 3090 and checked against every
    failing (power-of-two) case for head_dim 16..128, chunk 16..128, float32
    and bfloat16: the model reproduces Triton's request byte-for-byte (e.g.
    float32 head_dim 64: 160 KiB at chunk 64; bfloat16 head_dim 64 (features
    stored in bfloat16): 144 KiB at chunk 64). At chunk 256 Triton drops
    pipelining and requests less, so the model is conservative there. The
    [feature_dim, head_dim] float32 state is resident for the whole program.
    """
    if _full_fp32_matmul_precision():
        # Full-fp32 dots (no TF32) need larger, non-linearly growing buffers.
        # Conservative bound over measured requests (e.g. head_dim 64: 112 KiB
        # at chunk 32, 384 KiB at chunk 128; head_dim 16: 144 KiB at chunk 128).
        return (
            4 * chunk_size * chunk_size
            + 40 * chunk_size * head_dim
            + (4 * feature_dim * head_dim)
        )
    per_row = (
        2 * feature_dim + (4 if jnp.dtype(value_dtype) == jnp.float32 else 3) * head_dim
    )
    return 4 * chunk_size * per_row + 4 * feature_dim * head_dim


def _full_fp32_matmul_precision() -> bool:
    """Whether ``jax_default_matmul_precision`` (incl. context) disables TF32."""
    precision = jax.config.jax_default_matmul_precision
    return str(precision).lower() in ("highest", "float32", "precision.highest")


def _fit_chunk_size(
    chunk_size: int, feature_dim: int, head_dim: int, value_dtype
) -> int | None:
    """Largest chunk <= ``chunk_size`` (halving, >= 16) whose tiles fit in
    shared memory, or ``None`` if none does on this device."""
    from .attention import _query_triton_smem_limit

    try:
        limit = _query_triton_smem_limit() or _DEFAULT_SMEM_BYTES
    except Exception:
        limit = _DEFAULT_SMEM_BYTES
    while _kernel_smem_bytes(chunk_size, feature_dim, head_dim, value_dtype) > limit:
        if chunk_size <= 16:
            return None
        chunk_size //= 2
    return chunk_size


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _gpu_chunk_size(q: jax.Array, v: jax.Array, chunk_size: int) -> int | None:
    # Pallas-Triton only lowers power-of-two tiles (head_dim 48/80/96 fail).
    if not (_is_pow2(q.shape[-1]) and _is_pow2(v.shape[-1]) and _is_pow2(chunk_size)):
        return None
    feature_dim = 2 * q.shape[-1]  # both feature maps concatenate +/- halves
    return _fit_chunk_size(chunk_size, feature_dim, v.shape[-1], v.dtype)


def _linear_attention_pallas_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    feature_map: FeatureMap,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if seq_len % chunk_size:
        raise ValueError("sequence length must be divisible by chunk_size")
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError("Pallas linear attention requires equal q/k/v head dimensions")
    if h0.shape != (batch_size, num_heads, 2 * head_dim, head_dim):
        raise ValueError("h0 must have shape [batch, heads, 2 * head_dim, head_dim]")

    repeats = num_heads // num_kv_heads
    # Feature maps are per-token, so apply them once outside the kernel. They
    # are computed in float32 and stored in the input dtype (as in the scan
    # reference), so bf16 inputs do not materialize float32 [L, 2D] features.
    feature_fn = get_feature_map(feature_map).fwd
    q_features = feature_fn(q)
    k_features = feature_fn(k)
    feature_dim = q_features.shape[-1]
    if jax.default_backend() == "gpu" and not use_interpret_mode():
        # The chunk only tiles the recurrence; any divisor gives the same result.
        fitted = _gpu_chunk_size(q, v, chunk_size)
        if fitted is None:
            raise ValueError(
                "residual_linear_attention_pallas: no power-of-two chunk fits "
                "this GPU's shared memory (or head_dim/chunk_size is not a power "
                f"of two) for head_dim={head_dim}, chunk_size={chunk_size}, "
                f"dtype={v.dtype}; use implementation='chunked' (or 'auto', which "
                "falls back to it)."
            )
        chunk_size = fitted
    q_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, feature_dim),
        index_map=lambda b, h: (b, h, 0, 0),
    )
    k_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, feature_dim),
        index_map=lambda b, h: (b, h // repeats, 0, 0),
    )
    kv_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, head_dim),
        index_map=lambda b, h: (b, h // repeats, 0, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, head_dim),
        index_map=lambda b, h: (b, h, 0, 0),
    )
    state_spec = pl.BlockSpec(
        block_shape=(None, None, 2 * head_dim, head_dim),
        index_map=lambda b, h: (b, h, 0, 0),
    )
    output_shape = jax.ShapeDtypeStruct(q.shape, v.dtype)
    state_shape = jax.ShapeDtypeStruct(h0.shape, jnp.float32)
    backend = "triton" if jax.default_backend() == "gpu" else None
    kernel = functools.partial(_linear_attention_kernel, chunk_size=chunk_size)
    return pallas_call_compat(
        kernel,
        in_specs=(q_spec, k_spec, kv_spec, state_spec),
        out_specs=(out_spec, state_spec),
        out_shape=(output_shape, state_shape),
        grid=(batch_size, num_heads),
        interpret=use_interpret_mode(),
        backend=backend,
    )(q_features, k_features, v, h0.astype(jnp.float32))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    feature_map: FeatureMap,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    return _linear_attention_pallas_impl(q, k, v, h0, feature_map, chunk_size)


def _linear_attention_pallas_fwd(q, k, v, h0, feature_map, chunk_size):
    output = _linear_attention_pallas_impl(q, k, v, h0, feature_map, chunk_size)
    return output, (q, k, v, h0)


def _linear_attention_pallas_bwd(feature_map, chunk_size, residuals, cotangents):
    # AXLearn's backward passes state between ordered TPU grid programs. Triton
    # grids are unordered, so differentiate the chunked XLA form instead: it
    # recomputes chunk scores and keeps only chunk-boundary states, rather
    # than the per-token state history the step-scan VJP stores.
    q, k, v, h0 = residuals
    _, pullback = jax.vjp(
        lambda q_, k_, v_, h0_: _linear_attention_chunked_core(
            q_, k_, v_, h0_, feature_map, chunk_size
        ),
        q,
        k,
        v,
        h0,
    )
    return pullback(cotangents)


_linear_attention_pallas.defvjp(
    _linear_attention_pallas_fwd, _linear_attention_pallas_bwd
)


@functools.partial(
    jax.jit,
    static_argnames=("window_size", "feature_map", "chunk_size"),
)
def residual_linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    *,
    window_size: int,
    feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
    chunk_size: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Chunked Pallas residual linear attention."""
    feature_map = FeatureMap(feature_map)
    shift = window_size + 1
    k = right_shift_and_zero_pad(k, shift)
    v = right_shift_and_zero_pad(v, shift)
    return _linear_attention_pallas(q, k, v, h0, feature_map, chunk_size)


def residual_linear_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array,
    *,
    window_size: int,
    feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
    chunk_size: int = 128,
    implementation: str = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Dispatch residual linear attention to the selected implementation.

    ``implementation``:

    * ``"auto"`` / ``"chunked"`` -- chunk-parallel pure JAX/XLA (any backend,
      length and head dim). On an RTX 3090 it beats the Pallas forward by
      1.4-2.5x for L >= 2048 (cuBLAS batched matmuls vs one Triton program
      per head), so it is the default on GPU as well as CPU.
    * ``"pallas"`` -- the Pallas kernel (interpret mode on CPU); its backward
      is the VJP of the chunked form.
    * ``"scan"`` -- the per-token ``lax.scan`` reference (slow; its VJP stores
      every ``[2D, D]`` state).
    """
    if implementation not in ("auto", "pallas", "chunked", "scan"):
        raise ValueError(
            "implementation must be 'auto', 'pallas', 'chunked', or 'scan'"
        )
    kwargs = dict(window_size=window_size, feature_map=feature_map)
    if implementation == "scan":
        return residual_linear_attention_scan(q, k, v, h0, **kwargs)
    if implementation == "pallas":
        return residual_linear_attention_pallas(
            q, k, v, h0, chunk_size=chunk_size, **kwargs
        )
    return residual_linear_attention_chunked(
        q, k, v, h0, chunk_size=chunk_size, **kwargs
    )


__all__ = [
    "FeatureMap",
    "get_feature_map",
    "residual_linear_attention",
    "residual_linear_attention_chunked",
    "residual_linear_attention_pallas",
    "residual_linear_attention_scan",
    "right_shift_and_zero_pad",
]
