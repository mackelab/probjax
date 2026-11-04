"""Split-KV ("flash-decoding") softmax attention forward for few queries.

With few queries (decode steps, cross-attention onto a long memory) the regular
flash forward launches ``B * Hq * ceil(Lq / block_q)`` programs -- often far
fewer than the GPU has SMs -- and each walks the whole KV sequence alone. This
kernel instead partitions the KV sequence over ``num_splits`` extra programs.
Each program writes a partial output normalised by its own softmax sum plus the
partial log-sum-exp; a tiny XLA reduction combines the splits exactly.

GQA/MQA: the query heads that share a KV head are packed into the rows of one
query tile (``R = (Hq // Hkv) * Lq`` rows), so every KV tile is loaded once per
group instead of once per query head, and no K/V copy is ever made.

The kernel handles any ``Lq``/``Lkv`` (keys past the end are masked in the
kernel, no padded K/V copy). It covers unmasked, bias-free, dropout-free
attention; :func:`probjax.nn.pallas_kernels.kernels.attention.mha` dispatches
here only for such calls, and only in the forward pass -- under
differentiation the regular kernels run (see ``splitkv_attention``).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from ..kernel_utils import NEG_INF, dot, get_dot_precision

_LOG2E = 1.4426950408889634
# RTX 3090-class GPUs have 82 SMs; the heuristic only needs the order of
# magnitude, so a fixed target avoids a device query at trace time.
_TARGET_PROGRAMS = 2 * 82
_MIN_SPLIT_LEN = 256
_SMEM_BUDGET = 88 * 1024


def _splitkv_kernel(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    lse_ref,
    *,
    sm_scale: float,
    kv_len: int,
    split_len: int,
    num_splits: int,
    block_k: int,
    head_dim: int,
    v_dim: int,
    block_r: int,
    block_d: int,
    block_dv: int,
):
    split = pl.program_id(0) % num_splits
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)
    d_mask = jnp.arange(block_d)[None] < head_dim
    dv_mask = jnp.arange(block_dv)[None] < v_dim

    q = pallas_primitives.load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0)
    start = split * split_len

    def body(i, carry):
        o_prev, m_prev, l_prev = carry
        span_k = start + i * block_k + jnp.arange(block_k)
        valid = span_k < kv_len
        k = pallas_primitives.load(
            k_ref,
            (pl.dslice(start + i * block_k, block_k), slice(None)),
            mask=valid[:, None] & d_mask,
            other=0,
        )
        qk = dot(q, k.T, precision=precision) * (sm_scale * _LOG2E)
        qk = jnp.where(valid[None, :], qk, NEG_INF)
        m_next = jnp.maximum(m_prev, qk.max(axis=-1))
        corr = jnp.exp2(m_prev - m_next)
        p = jnp.exp2(qk - m_next[:, None])
        l_next = corr * l_prev + p.sum(axis=-1)
        v = pallas_primitives.load(
            v_ref,
            (pl.dslice(start + i * block_k, block_k), slice(None)),
            mask=valid[:, None] & dv_mask,
            other=0,
        )
        o_next = corr[:, None] * o_prev + dot(p.astype(v.dtype), v, precision=precision)
        return o_next, m_next, l_next

    o = jnp.zeros((block_r, block_dv), jnp.float32)
    m = jnp.full((block_r,), NEG_INF, jnp.float32)
    l_ = jnp.zeros((block_r,), jnp.float32)
    # Static trip count: every split but possibly the last is full; keys past
    # ``kv_len`` are masked above.
    o, m, l_ = lax.fori_loop(0, split_len // block_k, body, (o, m, l_))
    safe_l = jnp.where(l_ == 0.0, 1.0, l_)
    o_ref[...] = (o / safe_l[:, None]).astype(o_ref.dtype)
    # Base-2 LSE; an empty split gets -inf weight in the combine.
    lse_ref[...] = jnp.where(l_ == 0.0, NEG_INF, m + jnp.log2(safe_l))


def default_num_splits(
    batch: int, num_kv_heads: int, row_blocks: int, kv_len: int, block_k: int
) -> int:
    """Splits so that ``B * Hkv * row_blocks * splits`` fills the GPU.

    Each split keeps at least ``_MIN_SPLIT_LEN`` keys so the per-split loop
    amortises the combine.
    """
    programs = batch * num_kv_heads * row_blocks
    want = max(1, math.ceil(_TARGET_PROGRAMS / programs))
    most = max(1, kv_len // max(_MIN_SPLIT_LEN, block_k))
    return int(min(want, most))


def _pack_queries(q: jax.Array, num_kv_heads: int) -> jax.Array:
    """``[B, Lq, Hq, D] -> [B, Hkv, G * Lq, D]`` with row ``g * Lq + t``."""
    b, lq, hq, d = q.shape
    g = hq // num_kv_heads
    q = q.reshape(b, lq, num_kv_heads, g, d)
    return jnp.transpose(q, (0, 2, 3, 1, 4)).reshape(b, num_kv_heads, g * lq, d)


def _unpack_rows(o: jax.Array, lq: int, num_heads: int) -> jax.Array:
    """Inverse of :func:`_pack_queries` for ``[B, Hkv, G * Lq, Dv]``."""
    b, hkv, _, dv = o.shape
    g = num_heads // hkv
    o = o.reshape(b, hkv, g, lq, dv)
    return jnp.transpose(o, (0, 3, 1, 2, 4)).reshape(b, lq, num_heads, dv)


def splitkv_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float,
    num_splits: int | None = None,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    interpret: bool = False,
    return_lse: bool = False,
):
    """Split-KV attention forward. ``q [B, Lq, Hq, D]``, ``k/v [B, Lkv, Hkv, *]``.

    Returns ``[B, Lq, Hq, Dv]`` in ``q.dtype`` (and the natural-log LSE
    ``[B, Hq, Lq]`` when ``return_lse``).
    """
    b, lq, hq, d = q.shape
    _, lkv, hkv, _ = k.shape
    dv = v.shape[-1]
    if hq % hkv:
        raise ValueError(f"Hq={hq} must be a multiple of Hkv={hkv}.")
    rows = (hq // hkv) * lq
    # One row tile when it fits; Triton dots need >= 16 rows.
    block_r = max(16, pl.next_power_of_2(min(rows, 64)))
    row_blocks = pl.cdiv(rows, block_r)
    rows_pad = row_blocks * block_r
    block_k = min(block_k, max(16, pl.next_power_of_2(lkv)))
    block_d = pl.next_power_of_2(d)
    block_dv = pl.next_power_of_2(dv)
    # Keep the pipelined K/V tiles plus the Q tile inside small-SMEM parts
    # (~99 KB opt-in on sm_86).
    itemsize = jnp.dtype(q.dtype).itemsize
    while (
        block_k > 16
        and (num_stages * block_k * (block_d + block_dv) + block_r * block_d) * itemsize
        > _SMEM_BUDGET
    ):
        block_k //= 2
    if num_splits is None:
        num_splits = default_num_splits(b, hkv, row_blocks, lkv, block_k)
    split_len = pl.cdiv(pl.cdiv(lkv, num_splits), block_k) * block_k
    num_splits = pl.cdiv(lkv, split_len)  # drop empty trailing splits

    qp = _pack_queries(q, hkv)
    if rows_pad != rows:
        qp = jnp.pad(qp, ((0, 0), (0, 0), (0, rows_pad - rows), (0, 0)))

    kernel = functools.partial(
        _splitkv_kernel,
        sm_scale=float(sm_scale),
        kv_len=lkv,
        split_len=split_len,
        num_splits=num_splits,
        block_k=block_k,
        head_dim=d,
        v_dim=dv,
        block_r=block_r,
        block_d=block_d,
        block_dv=block_dv,
    )
    grid = (row_blocks * num_splits, b, hkv)
    o_part, lse_part = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec(
                (None, None, block_r, block_d),
                lambda s, i, h: (i, h, s // num_splits, 0),
            ),
            pl.BlockSpec((None, lkv, None, block_d), lambda s, i, h: (i, 0, h, 0)),
            pl.BlockSpec((None, lkv, None, block_dv), lambda s, i, h: (i, 0, h, 0)),
        ],
        out_specs=[
            pl.BlockSpec(
                (None, None, None, block_r, block_dv),
                lambda s, i, h: (i, h, s % num_splits, s // num_splits, 0),
            ),
            pl.BlockSpec(
                (None, None, None, block_r),
                lambda s, i, h: (i, h, s % num_splits, s // num_splits),
            ),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, hkv, num_splits, rows_pad, block_dv), jnp.float32),
            jax.ShapeDtypeStruct((b, hkv, num_splits, rows_pad), jnp.float32),
        ],
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps, num_stages=num_stages
        ),
        interpret=interpret,
        name="mha_splitkv_forward",
    )(qp, k, v)

    # Exact log-sum-exp combine over the splits (base 2).
    m = jnp.max(lse_part, axis=2, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.exp2(lse_part - m)  # [B, Hkv, S, R]
    denom = jnp.sum(w, axis=2)
    out = (
        jnp.einsum("bhsr,bhsrd->bhrd", w, o_part)
        / jnp.where(denom == 0, 1.0, denom)[..., None]
    )
    out = _unpack_rows(out[:, :, :rows, :dv], lq, hq).astype(q.dtype)
    if not return_lse:
        return out
    lse = (m[:, :, 0] + jnp.log2(denom)) / _LOG2E  # natural log, [B, Hkv, R]
    lse = lse[:, :, :rows].reshape(b, hkv, hq // hkv, lq).reshape(b, hq, lq)
    return out, lse


def splitkv_attention(q, k, v, *, sm_scale: float, fallback, **kwargs):
    """Split-KV forward; differentiation runs ``fallback(q, k, v)`` instead.

    The split-KV kernel has no backward. ``jax.custom_vjp`` uses the primal
    (split-KV) function when nothing is differentiated, and the fallback's own
    VJP (the regular flash kernels) under ``jax.grad`` / ``jax.vjp``.
    """

    @jax.custom_vjp
    def attn(q, k, v):
        return splitkv_forward(q, k, v, sm_scale=sm_scale, **kwargs)

    def fwd(q, k, v):
        return jax.vjp(fallback, q, k, v)

    def bwd(vjp_fn, do):
        return vjp_fn(do)

    attn.defvjp(fwd, bwd)
    return attn(q, k, v)
