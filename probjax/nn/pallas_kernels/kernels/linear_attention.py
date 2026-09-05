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
    return jnp.where(y_pos > 0, dy_pos, 0) - jnp.where(y_neg > 0, dy_neg, 0)


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
    feature_map: FeatureMap,
    chunk_size: int,
):
    """One-program-per-head chunked kernel, safe for unordered GPU grids."""
    seq_len, head_dim = q_ref.shape
    feature_fn = get_feature_map(feature_map).fwd
    state = h0_ref[:, :].astype(jnp.float32)
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))

    def chunk_step(chunk_index, current_state):
        start = chunk_index * chunk_size
        q = pallas_primitives.load(
            q_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        k = pallas_primitives.load(
            k_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        v = pallas_primitives.load(
            v_ref, (pl.dslice(start, chunk_size), slice(None))
        ).astype(jnp.float32)
        q_features = feature_fn(q).astype(jnp.float32)
        k_features = feature_fn(k).astype(jnp.float32)
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
    q_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, head_dim),
        index_map=lambda b, h: (b, h, 0, 0),
    )
    kv_spec = pl.BlockSpec(
        block_shape=(None, None, seq_len, head_dim),
        index_map=lambda b, h: (b, h // repeats, 0, 0),
    )
    state_spec = pl.BlockSpec(
        block_shape=(None, None, 2 * head_dim, head_dim),
        index_map=lambda b, h: (b, h, 0, 0),
    )
    output_shape = jax.ShapeDtypeStruct(q.shape, v.dtype)
    state_shape = jax.ShapeDtypeStruct(h0.shape, jnp.float32)
    backend = "triton" if jax.default_backend() == "gpu" else None
    kernel = functools.partial(
        _linear_attention_kernel,
        feature_map=feature_map,
        chunk_size=chunk_size,
    )
    return pallas_call_compat(
        kernel,
        in_specs=(q_spec, kv_spec, kv_spec, state_spec),
        out_specs=(q_spec, state_spec),
        out_shape=(output_shape, state_shape),
        grid=(batch_size, num_heads),
        interpret=use_interpret_mode(),
        backend=backend,
    )(q, k, v, h0.astype(jnp.float32))


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
    # grids are unordered, so use the reference VJP until a GPU-specific
    # one-program-per-head backward kernel is implemented.
    del chunk_size
    q, k, v, h0 = residuals
    _, pullback = jax.vjp(
        lambda q_, k_, v_, h0_: _linear_attention_scan_core(
            q_, k_, v_, h0_, feature_map
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
    """Dispatch residual linear attention to the selected implementation."""
    if implementation not in ("auto", "pallas", "scan"):
        raise ValueError("implementation must be 'auto', 'pallas', or 'scan'")
    use_pallas = implementation == "pallas" or (
        implementation == "auto"
        and jax.default_backend() == "gpu"
        and q.shape[2] % chunk_size == 0
    )
    if use_pallas:
        return residual_linear_attention_pallas(
            q,
            k,
            v,
            h0,
            window_size=window_size,
            feature_map=feature_map,
            chunk_size=chunk_size,
        )
    return residual_linear_attention_scan(
        q,
        k,
        v,
        h0,
        window_size=window_size,
        feature_map=feature_map,
    )


__all__ = [
    "FeatureMap",
    "get_feature_map",
    "residual_linear_attention",
    "residual_linear_attention_pallas",
    "residual_linear_attention_scan",
    "right_shift_and_zero_pad",
]
