"""XLA-native (non-Pallas) attention used by :func:`mha` / :func:`mla` on CPU.

On GPU the Triton kernels in :mod:`.attention` are used.  On CPU those kernels
only run in Pallas *interpret* mode, which is correct but orders of magnitude
too slow.  This module provides an XLA implementation with the same semantics:

* masks and biases are evaluated exactly like the kernels evaluate them: the
  mask/bias callables are called per ``(batch, head)`` on index tiles, with the
  mask data / dense-bias data sliced through the objects' own Pallas
  ``BlockSpec`` index maps (so every mask/bias class that works with the kernel
  works here, including learnable biases -- their gradients flow);
* scores are ``sm_scale * q.k`` (fp32), then the bias, then the mask; fully
  masked query rows produce zeros (like the kernel);
* dropout is applied to the unnormalised probabilities after the softmax
  normaliser is accumulated (like the kernel).  ``dropout_impl="counter"``
  reproduces the kernel's counter-hash mask bit for bit (global batch index;
  under ``custom_partitioning`` the kernel hashes the *local* batch index, so
  sharded kernel runs with batch sharding differ);
  ``dropout_impl="materialize"`` reproduces the kernel's
  ``jax.random.bernoulli`` mask exactly as well;
* GQA: ``k``/``v`` may have fewer heads than ``q`` (``Hq % Hkv == 0``, head
  ``h`` uses KV head ``h // (Hq // Hkv)``, i.e. the ``jnp.repeat`` convention)
  without materialising repeated K/V.

Two strategies, picked by shape (see :func:`_auto_strategy`):

* **dense** for small problems: full ``[B, H, Q, K]`` scores, plain JAX
  autodiff in both modes;
* **tiled** for larger ones: online softmax over ``block_q x block_k`` tiles
  (``lax.map`` over query blocks, ``lax.scan`` over KV blocks, tiles that the
  mask's static ``block_mask`` rules out are skipped with ``lax.cond``), so the
  working set is ``O(B * H * block_q * block_k)``.  Reverse mode uses a
  ``custom_vjp`` whose backward *recomputes* the probabilities per tile from
  the saved log-sum-exp (flash-attention-2 style) instead of storing them; the
  residuals are only ``q, k, v, out, lse``.  Forward mode (``jax.jvp``)
  differentiates the tiled forward directly, with the same bounded working
  set.  Mid-sized problems use a single tile (no loop), which is the fastest
  option there because of the cheaper backward.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..attention_mask_bias import AttentionBias, AttentionMask
from ..kernel_utils import get_dropout_mask

__all__ = ["xla_attention", "use_xla_attention"]

# Strategy thresholds, on the per-head score count Q * K.  From CPU timings
# (docs/scripts/benchmark_kernel_attn-cpu.py, 24-core x86, B*H = 8, D = 64):
# * up to 256 x 256 the dense strategy is as fast as anything else;
# * up to 512 x 512 a single tile (flash-style custom_vjp backward over one
#   tile) beats dense by ~2x in fwd+bwd and 256x256 tiles by ~1.5x;
# * beyond that 256 x 256 tiles win (and skip fully masked tiles, ~2x for
#   causal), and keep the working set bounded.
DENSE_MAX_SCORES = 256 * 256
SINGLE_TILE_MAX_SCORES = 512 * 512
# Tile of the tiled strategy (query rows x KV columns).  256x256 was best or
# within noise of the best of {Q x 256, 512x256, 256x256, 512x128, 1024x512}
# for T in {1024, 2048}, causal and not, fwd and fwd+bwd.
DEFAULT_BLOCK_Q = 256
DEFAULT_BLOCK_K = 256


def use_xla_attention(x: Any, interpret: bool) -> bool:
    """Whether ``mha``/``mla`` should take the XLA path for array ``x``.

    True on a CPU default backend (or for concrete arrays committed to CPU),
    unless the caller explicitly asked for Pallas interpret mode.
    """
    if interpret:
        return False
    if jax.default_backend() == "cpu":
        return True
    try:  # concrete arrays committed to CPU (tracers raise here)
        return all(d.platform == "cpu" for d in x.devices())
    except Exception:  # noqa: BLE001 - best effort device introspection
        return False


# ---------------------------------------------------------------------------
# Mask / bias evaluation (mirrors mha_forward_kernel)
# ---------------------------------------------------------------------------


def _is_squeezed(dim) -> bool:
    if dim is None:
        return True
    return type(dim).__name__ == "Squeezed"


def _spec_block(arr, spec, grid_idx, window=None):
    """Slice the block that ``spec`` maps grid point ``grid_idx`` to.

    Squeezed (``None``) block dims are dropped, as in the kernel's ref view.
    ``window`` optionally maps an axis to ``(offset, size)`` to slice only that
    sub-range of the block along it (in one ``dynamic_slice``; the caller makes
    sure the array is long enough along those axes).
    """
    window = window or {}
    idx = spec.index_map(*grid_idx)
    if not isinstance(idx, (tuple, list)):
        idx = (idx,)
    starts, sizes, squeeze, pads = [], [], [], []
    for d, (bs, ix) in enumerate(zip(spec.block_shape, idx, strict=True)):
        dim = arr.shape[d]
        if _is_squeezed(bs):
            starts.append(ix)
            sizes.append(1)
            squeeze.append(d)
            pads.append((0, 0))
            continue
        bs = int(getattr(bs, "block_size", bs))
        if d in window:
            off, size = window[d]
            starts.append(ix * bs + off)
            sizes.append(size)
            pads.append((0, 0))
        else:
            starts.append(ix * bs)
            sizes.append(bs)
            pads.append((0, -dim % bs))
    if any(p for _, p in pads):
        # Pallas reads past the end of a ragged block (the values only reach
        # padded positions, which are masked); zero-fill here.
        arr = jnp.pad(arr, pads)
    starts = [jnp.asarray(s, jnp.int32) for s in starts]
    out = lax.dynamic_slice(arr, starts, sizes)
    return out.reshape([s for d, s in enumerate(sizes) if d not in squeeze])


def _pad_last(x, total: int):
    """Zero-pad the last axis of ``x`` up to ``total``."""
    if x.shape[-1] >= total:
        return x
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, total - x.shape[-1])])


def _mask_data_window(blk, start, size: int, total: int):
    """Equivalent of ``attention._load_mask_data`` on an (unpaged) array block.

    Scalar-like data is returned as a scalar; per-position data is windowed to
    ``[start, start + size)`` (padded up to ``total`` positions first; padded
    keys are masked out by the length mask).
    """
    if blk is None:
        return None
    shape = blk.shape
    trimmed = shape
    while len(trimmed) > 1 and trimmed[-1] == 1:
        trimmed = trimmed[:-1]
    n_leading = 0
    while n_leading < len(trimmed) - 1 and trimmed[n_leading] == 1:
        n_leading += 1
    core = trimmed[n_leading:]
    if core == () or shape == ():
        return blk.reshape(())
    if core == (1,):
        return blk.reshape(-1)[0]
    vec = blk[(0,) * n_leading + (slice(None),) + (0,) * (blk.ndim - n_leading - 1)]
    return lax.dynamic_slice_in_dim(_pad_last(vec, total), start, size)


class _Mods:
    """Mask/bias/dropout evaluation on a ``[B, Hq, bq, kc]`` score tile."""

    def __init__(
        self,
        mask: AttentionMask | None,
        bias: AttentionBias | None,
        *,
        q_len: int,
        kv_len: int,
        q_len_padded: int,
        kv_len_padded: int,
        dropout_rate: float,
        dropout_impl: str,
        rng_seed,
        dropout_mask,
    ):
        self.mask, self.bias = mask, bias
        self.q_len, self.kv_len = q_len, kv_len
        self.q_total, self.kv_total = q_len_padded, kv_len_padded
        self.kv_padded = kv_len_padded != kv_len
        self.dropout_rate = dropout_rate
        self.dropout_impl = dropout_impl
        self.rng_seed = rng_seed
        self.dropout_mask = (
            None
            if dropout_mask is None
            else _pad_last2(dropout_mask, q_len_padded, kv_len_padded)
        )
        self.q_id = self.k_id = self.q_spec = self.k_spec = None
        if mask is not None:
            self.q_id, self.k_id = mask.get_data(q_seq_len=q_len, kv_seq_len=kv_len)
            if self.q_id is not None or self.k_id is not None:
                self.q_spec, self.k_spec = mask.get_data_block_spec(
                    q_len, kv_len, q_len, kv_len
                )
        self.b_data = self.b_spec = None
        if bias is not None:
            self.b_data = bias.get_data()
            if self.b_data is not None:
                # One tile spans the whole [Q, K] plane, so the trailing block
                # indices are 0 and tiles are windows into it; zero-pad it to
                # whole tiles once (padded rows/keys are discarded/masked).
                self.b_spec = bias.get_block_spec(
                    q_len=q_len, kv_len=kv_len, block_q=q_len, block_kv=kv_len
                )
                self.b_data = _pad_last2(
                    self.b_data,
                    max(q_len_padded, q_len),
                    max(kv_len_padded, kv_len),
                )

    @property
    def has_scores_mods(self) -> bool:
        return self.mask is not None or self.bias is not None or self.kv_padded

    def apply(self, s, q_start, bq: int, kv_start, kc: int):
        """Biased + masked scores of a tile; masked entries become ``-inf``."""
        if not self.has_scores_mods:
            return s
        span_q = q_start + jnp.arange(bq, dtype=jnp.int32)
        span_k = kv_start + jnp.arange(kc, dtype=jnp.int32)

        def per_bh(s_bh, b, h):
            grid = (0, b, h)
            if self.bias is not None:
                chunk = None
                if self.b_data is not None:
                    nd = self.b_data.ndim
                    chunk = _spec_block(
                        self.b_data,
                        self.b_spec,
                        grid,
                        window={nd - 2: (q_start, bq), nd - 1: (kv_start, kc)},
                    )
                s_bh = self.bias(s_bh, h, span_q, span_k, data=chunk)
            keep = None
            if self.mask is not None:
                id_q = id_k = None
                q_blk = (
                    _spec_block(self.q_id, self.q_spec, grid)
                    if self.q_id is not None and self.q_spec is not None
                    else self.q_id
                )
                if q_blk is not None:
                    id_q = _mask_data_window(q_blk, q_start, bq, self.q_total)
                if self.k_id is not None:
                    k_blk = (
                        _spec_block(self.k_id, self.k_spec, grid)
                        if self.k_spec is not None
                        else self.k_id
                    )
                    id_k = _mask_data_window(k_blk, kv_start, kc, self.kv_total)
                elif q_blk is not None:
                    id_k = _mask_data_window(q_blk, kv_start, kc, self.kv_total)
                keep = jnp.broadcast_to(
                    self.mask(span_q, span_k, id_q, id_k), s_bh.shape
                )
            if self.kv_padded:
                valid = jnp.broadcast_to(span_k[None, :] < self.kv_len, s_bh.shape)
                keep = valid if keep is None else keep & valid
            if keep is not None:
                s_bh = jnp.where(keep, s_bh, -jnp.inf)
            return s_bh

        b_ax = jnp.arange(s.shape[0], dtype=jnp.int32)
        h_ax = jnp.arange(s.shape[1], dtype=jnp.int32)
        per_b = jax.vmap(per_bh, in_axes=(0, None, 0))
        return jax.vmap(per_b, in_axes=(0, 0, None))(s, b_ax, h_ax)

    def drop_scale(self, shape, q_start, bq: int, kv_start, kc: int):
        """Multiplier ``0`` (dropped) or ``1 / (1 - rate)`` per score, or None."""
        if self.dropout_rate <= 0:
            return None
        if self.dropout_impl == "counter":
            from .attention import _dropout_mask_counter

            span_q = q_start + jnp.arange(bq, dtype=jnp.int32)
            span_k = kv_start + jnp.arange(kc, dtype=jnp.int32)

            def per_bh(b, h):
                return _dropout_mask_counter(
                    self.rng_seed, b, h, span_q, span_k, self.dropout_rate
                )

            b_ax = jnp.arange(shape[0], dtype=jnp.uint32)
            h_ax = jnp.arange(shape[1], dtype=jnp.uint32)
            dropped = jax.vmap(jax.vmap(per_bh, in_axes=(None, 0)), in_axes=(0, None))(
                b_ax, h_ax
            )
        else:
            dropped = lax.dynamic_slice_in_dim(self.dropout_mask, kv_start, kc, axis=3)
            dropped = lax.dynamic_slice_in_dim(dropped, q_start, bq, axis=2)
        keep_scale = jnp.float32(1.0 / (1.0 - self.dropout_rate))
        return jnp.where(dropped, jnp.float32(0.0), keep_scale)


def _pad_last2(x, rows: int, cols: int):
    """Zero-pad the last two axes of ``x`` up to ``(rows, cols)``."""
    if x.ndim < 2:
        return x
    pr, pc = max(rows - x.shape[-2], 0), max(cols - x.shape[-1], 0)
    if not (pr or pc):
        return x
    return jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, pr), (0, pc)])


def _tile_block_mask(mask, q_total, kv_total, bq, kc):
    """Static ``[nQ, nK]`` map of tiles that are not fully masked, or None."""
    if mask is None:
        return None
    try:
        bm = mask.block_mask(q_total, kv_total, bq, kc)
        if bm is None:
            return None
        bm = np.asarray(bm, dtype=bool)
    except Exception:  # noqa: BLE001 - data-dependent masks: no static pruning
        return None
    if bm.shape != (q_total // bq, kv_total // kc) or bm.all():
        return None
    return bm


# ---------------------------------------------------------------------------
# Core math.  Layout inside: q5 [B, Hkv, G, Q, D], k/v [B, Hkv, K, D].
# ---------------------------------------------------------------------------


def _scores(q5, kc, sm_scale):
    s = jnp.einsum("bhgqd,bhkd->bhgqk", q5, kc, preferred_element_type=jnp.float32)
    if sm_scale != 1.0:
        s = s * jnp.float32(sm_scale)
    b, hkv, g, q, k = s.shape
    return s.reshape(b, hkv * g, q, k)


def _pv(p, vc, hkv):
    b, hq, q, k = p.shape
    p5 = p.reshape(b, hkv, hq // hkv, q, k).astype(vc.dtype)
    return jnp.einsum("bhgqk,bhkd->bhgqd", p5, vc, preferred_element_type=jnp.float32)


def _dense_forward(q5, k, v, mods: _Mods, sm_scale):
    hkv = k.shape[1]
    q_len, kv_len = q5.shape[3], k.shape[2]
    s = mods.apply(_scores(q5, k, sm_scale), 0, q_len, 0, kv_len)
    m = jnp.max(s, axis=-1)
    m = lax.stop_gradient(jnp.where(jnp.isfinite(m), m, jnp.float32(0.0)))
    p = jnp.exp(s - m[..., None])
    l_ = jnp.sum(p, axis=-1)
    l_ = jnp.where(l_ == 0.0, jnp.float32(1.0), l_)
    p = p / l_[..., None]
    ds = mods.drop_scale(p.shape, 0, q_len, 0, kv_len)
    if ds is not None:
        p = p * ds
    return _pv(p, v, hkv)


def _slice(x, start, size, axis):
    return lax.dynamic_slice_in_dim(x, start, size, axis=axis)


def _maybe_cond(bm_const, i, j, compute, carry):
    """``compute(carry)`` unless tile ``(i, j)`` is statically fully masked."""
    if bm_const is None:
        return compute(carry)
    return lax.cond(bm_const[i, j], compute, lambda c: c, carry)


def _tiled_forward(q5, k, v, mods: _Mods, sm_scale, bq: int, kc: int, bm):
    """Online softmax over ``bq x kc`` tiles; returns (o5 fp32, lse [B,Hq,Q])."""
    b, hkv, g, q_total, _ = q5.shape
    dv = v.shape[-1]
    hq = hkv * g
    nq, nk = q_total // bq, k.shape[2] // kc
    bm_const = None if bm is None else jnp.asarray(bm)

    def q_block(i):
        q_start = i * bq
        qi = _slice(q5, q_start, bq, 3)

        def body(carry, j):
            def compute(carry):
                o, m, l_ = carry
                kv_start = j * kc
                kj, vj = _slice(k, kv_start, kc, 2), _slice(v, kv_start, kc, 2)
                s = mods.apply(_scores(qi, kj, sm_scale), q_start, bq, kv_start, kc)
                m_new = jnp.maximum(m, jnp.max(s, axis=-1))
                m_safe = jnp.where(jnp.isfinite(m_new), m_new, jnp.float32(0.0))
                corr = jnp.exp(m - m_safe)
                p = jnp.exp(s - m_safe[..., None])
                l_ = l_ * corr + jnp.sum(p, axis=-1)
                ds = mods.drop_scale(p.shape, q_start, bq, kv_start, kc)
                if ds is not None:
                    p = p * ds
                corr5 = corr.reshape(b, hkv, g, bq)[..., None]
                return o * corr5 + _pv(p, vj, hkv), m_new, l_

            return _maybe_cond(bm_const, i, j, compute, carry), None

        init = (
            jnp.zeros((b, hkv, g, bq, dv), jnp.float32),
            jnp.full((b, hq, bq), -jnp.inf, jnp.float32),
            jnp.zeros((b, hq, bq), jnp.float32),
        )
        (o, m, l_), _ = lax.scan(body, init, jnp.arange(nk, dtype=jnp.int32))
        nonempty = l_ > 0.0
        l_safe = jnp.where(nonempty, l_, jnp.float32(1.0))
        o = o / l_safe.reshape(b, hkv, g, bq)[..., None]
        lse = jnp.where(nonempty, m + jnp.log(l_safe), jnp.inf)
        return o, lse

    if nq == 1:
        return q_block(0)
    o, lse = lax.map(q_block, jnp.arange(nq, dtype=jnp.int32))
    # [nq, B, Hkv, G, bq, Dv] -> [B, Hkv, G, Q, Dv]; [nq, B, Hq, bq] -> [B, Hq, Q]
    o = jnp.moveaxis(o, 0, 3).reshape(b, hkv, g, q_total, dv)
    lse = jnp.moveaxis(lse, 0, 2).reshape(b, hq, q_total)
    return o, lse


def _split_inexact(leaves):
    return tuple(
        i
        for i, x in enumerate(leaves)
        if jnp.issubdtype(jnp.result_type(x), jnp.inexact)
    )


def _make_tiled_vjp(
    *,
    mask_treedef,
    bias_treedef,
    bias_diff_idx,
    sm_scale,
    bq,
    kc,
    bm,
    mods_kwargs,
    out_dtype,
):
    """``custom_vjp`` whose backward recomputes probabilities per tile.

    Residuals are ``q, k, v, out, lse`` (plus mask/bias leaves); no ``[Q, K]``
    sized array is ever saved.  Bias-parameter gradients come from a VJP of the
    tile's score function, so any differentiable bias works.
    """
    from .attention import _unflatten_optional_pytree

    def build_mods(mask_leaves, bias_leaves, rng_seed, dropout_mask):
        mask = _unflatten_optional_pytree(mask_treedef, tuple(mask_leaves))
        bias = _unflatten_optional_pytree(bias_treedef, tuple(bias_leaves))
        return _Mods(
            mask, bias, rng_seed=rng_seed, dropout_mask=dropout_mask, **mods_kwargs
        )

    def merge_bias(bias_static, bias_diff):
        leaves = list(bias_static)
        for i, x in zip(bias_diff_idx, bias_diff, strict=True):
            leaves[i] = x
        return leaves

    @jax.custom_vjp
    def fn(q5, k, v, rng_seed, dropout_mask, mask_leaves, bias_leaves):
        mods = build_mods(mask_leaves, bias_leaves, rng_seed, dropout_mask)
        o, _ = _tiled_forward(q5, k, v, mods, sm_scale, bq, kc, bm)
        return o.astype(out_dtype)

    def fwd(q5, k, v, rng_seed, dropout_mask, mask_leaves, bias_leaves):
        mods = build_mods(mask_leaves, bias_leaves, rng_seed, dropout_mask)
        o, lse = _tiled_forward(q5, k, v, mods, sm_scale, bq, kc, bm)
        o = o.astype(out_dtype)
        return o, (q5, k, v, rng_seed, dropout_mask, mask_leaves, bias_leaves, o, lse)

    def bwd(res, do):
        q5, k, v, rng_seed, dropout_mask, mask_leaves, bias_leaves, o, lse = res
        b, hkv, g, q_total, d = q5.shape
        hq = hkv * g
        nq, nk = q_total // bq, k.shape[2] // kc
        bm_const = None if bm is None else jnp.asarray(bm)
        mm_dtype = v.dtype
        delta = jnp.sum(
            do.astype(jnp.float32) * o.astype(jnp.float32), axis=-1
        ).reshape(b, hq, q_total)
        do = do.astype(mm_dtype)
        bias_diff0 = tuple(bias_leaves[i] for i in bias_diff_idx)

        def kv_body(carry, j):
            dq, dbias = carry
            kv_start = j * kc
            kj, vj = _slice(k, kv_start, kc, 2), _slice(v, kv_start, kc, 2)

            def q_body(c, i):
                def compute(c):
                    dkj, dvj, dq, dbias = c
                    q_start = i * bq
                    qi = _slice(q5, q_start, bq, 3)
                    doi = _slice(do, q_start, bq, 3)
                    lse_i = _slice(lse, q_start, bq, 2)
                    delta_i = _slice(delta, q_start, bq, 2)

                    def logits(q_, k_, bdiff):
                        mods = build_mods(
                            mask_leaves,
                            merge_bias(bias_leaves, bdiff),
                            rng_seed,
                            dropout_mask,
                        )
                        s = _scores(q_, k_, sm_scale)
                        return mods.apply(s, q_start, bq, kv_start, kc), mods

                    s, vjp_fn, mods = jax.vjp(logits, qi, kj, bias_diff0, has_aux=True)
                    p = jnp.exp(s - lse_i[..., None])
                    dscale = mods.drop_scale(p.shape, q_start, bq, kv_start, kc)
                    pd = p if dscale is None else p * dscale
                    # dV_j += P^T dO (summed over the query heads of a KV group).
                    dvj = dvj + jnp.einsum(
                        "bhgqk,bhgqd->bhkd",
                        pd.reshape(b, hkv, g, bq, kc).astype(mm_dtype),
                        doi,
                        preferred_element_type=jnp.float32,
                    )
                    dp = jnp.einsum(
                        "bhgqd,bhkd->bhgqk", doi, vj, preferred_element_type=jnp.float32
                    ).reshape(b, hq, bq, kc)
                    if dscale is not None:
                        dp = dp * dscale
                    dsc = p * (dp - delta_i[..., None])
                    dq_i, dk_i, dbias_i = vjp_fn(dsc)
                    dq_blk = _slice(dq, q_start, bq, 3) + dq_i.astype(jnp.float32)
                    dq = lax.dynamic_update_slice_in_dim(dq, dq_blk, q_start, axis=3)
                    dbias = jax.tree_util.tree_map(jnp.add, dbias, dbias_i)
                    return dkj + dk_i.astype(jnp.float32), dvj, dq, dbias

                return _maybe_cond(bm_const, i, j, compute, c), None

            init = (
                jnp.zeros(kj.shape, jnp.float32),
                jnp.zeros(vj.shape, jnp.float32),
                dq,
                dbias,
            )
            (dkj, dvj, dq, dbias), _ = lax.scan(
                q_body, init, jnp.arange(nq, dtype=jnp.int32)
            )
            return (dq, dbias), (dkj, dvj)

        dbias0 = jax.tree_util.tree_map(
            lambda x: jnp.zeros(jnp.shape(x), jnp.result_type(x)), bias_diff0
        )
        (dq, dbias), (dk, dv) = lax.scan(
            kv_body,
            (jnp.zeros(q5.shape, jnp.float32), dbias0),
            jnp.arange(nk, dtype=jnp.int32),
        )
        # [nk, B, Hkv, kc, D] -> [B, Hkv, nk * kc, D]
        dk = jnp.moveaxis(dk, 0, 2).reshape(k.shape)
        dv = jnp.moveaxis(dv, 0, 2).reshape(v.shape)
        dbias_leaves = [None] * len(bias_leaves)
        for i, x in zip(bias_diff_idx, dbias, strict=True):
            dbias_leaves[i] = x
        return (
            dq.astype(q5.dtype),
            dk.astype(k.dtype),
            dv.astype(v.dtype),
            None,
            None,
            tuple(None for _ in mask_leaves),
            tuple(dbias_leaves),
        )

    fn.defvjp(fwd, bwd)
    return fn


def _auto_strategy(q_len: int, kv_len: int):
    """``(strategy, block_q, block_k)`` for the ``"auto"`` strategy."""
    if q_len * kv_len <= DENSE_MAX_SCORES:
        return "dense", None, None
    if q_len * kv_len <= SINGLE_TILE_MAX_SCORES:
        return "tiled", q_len, kv_len
    return "tiled", DEFAULT_BLOCK_Q, DEFAULT_BLOCK_K


def _balanced_block(length: int, target: int) -> int:
    """Block size <= ``target`` that splits ``length`` into equal-ish blocks."""
    n = max(1, math.ceil(length / target))
    return math.ceil(length / n)


def xla_attention(
    q,
    k,
    v,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    rng: jax.Array | None = None,
    sm_scale: float = 1.0,
    dropout_rate: float = 0.0,
    dropout_impl: str = "counter",
    diff_mode: str = "reverse",
    strategy: str = "auto",
    block_q: int | None = None,
    block_k: int | None = None,
):
    """XLA attention with :func:`mha` semantics; ``q,k,v`` are ``[B, T, H, D]``.

    ``v`` may have a different head dim than ``q``/``k``; ``k``/``v`` may have
    fewer heads than ``q`` (GQA).  ``strategy`` is ``"auto"``, ``"dense"`` or
    ``"tiled"``; ``block_q``/``block_k`` set the tile of the tiled strategy.
    """
    from .attention import (
        _flatten_optional_pytree,
        _is_direct_jvp_trace,
        _rng_seed_from_key,
    )

    batch, q_len, hq, _ = q.shape
    kv_len, hkv = k.shape[1], k.shape[2]
    if hq % hkv != 0:
        raise ValueError(f"query heads {hq} must be a multiple of kv heads {hkv}")
    if v.shape[:3] != k.shape[:3]:
        raise ValueError(f"k {k.shape} and v {v.shape} must agree on [B, T, H]")
    g = hq // hkv
    out_dtype = q.dtype

    if dropout_rate > 0:
        if rng is None:
            raise ValueError("prng_key must be provided when dropout_rate>0")
        if dropout_impl not in ("counter", "materialize"):
            raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")
    rng_seed = (
        _rng_seed_from_key(rng)
        if (rng is not None and dropout_rate > 0)
        else jnp.uint32(0)
    )
    dropout_mask = None
    if dropout_rate > 0 and dropout_impl == "materialize":
        dropout_mask = get_dropout_mask(
            (batch, hq, q_len, kv_len), prng_key=rng, rate=dropout_rate
        )

    if strategy == "auto":
        strategy, auto_bq, auto_bk = _auto_strategy(q_len, kv_len)
        block_q, block_k = block_q or auto_bq, block_k or auto_bk
    if strategy == "dense":
        bq, kc = q_len, kv_len
    elif strategy == "tiled":
        bq = _balanced_block(q_len, block_q or DEFAULT_BLOCK_Q)
        kc = _balanced_block(kv_len, block_k or DEFAULT_BLOCK_K)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    q_pad = math.ceil(q_len / bq) * bq
    kv_pad = math.ceil(kv_len / kc) * kc

    # [B, T, H, D] -> [B, H, T, D] (zero-padded to whole tiles); q is further
    # split into KV-head groups.
    def to_bhtd(x, total):
        x = jnp.transpose(x, (0, 2, 1, 3))
        if x.shape[2] != total:
            x = jnp.pad(x, [(0, 0), (0, 0), (0, total - x.shape[2]), (0, 0)])
        return x

    q5 = to_bhtd(q, q_pad).reshape(batch, hkv, g, q_pad, q.shape[-1])
    kt, vt = to_bhtd(k, kv_pad), to_bhtd(v, kv_pad)

    def finish(o5):
        o = o5.reshape(batch, hq, q_pad, v.shape[-1])[:, :, :q_len]
        return jnp.transpose(o, (0, 2, 1, 3)).astype(out_dtype)

    mods_kwargs = dict(
        q_len=q_len,
        kv_len=kv_len,
        q_len_padded=q_pad,
        kv_len_padded=kv_pad,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
    )

    if strategy == "dense":
        mods = _Mods(
            mask, bias, rng_seed=rng_seed, dropout_mask=dropout_mask, **mods_kwargs
        )
        return finish(_dense_forward(q5, kt, vt, mods, sm_scale))

    bm = _tile_block_mask(mask, q_pad, kv_pad, bq, kc)
    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)
    forward_mode = diff_mode == "forward" or _is_direct_jvp_trace(
        q, k, v, mask_leaves, bias_leaves
    )
    if forward_mode:
        mods = _Mods(
            mask, bias, rng_seed=rng_seed, dropout_mask=dropout_mask, **mods_kwargs
        )
        o5, _ = _tiled_forward(q5, kt, vt, mods, sm_scale, bq, kc, bm)
        return finish(o5)

    fn = _make_tiled_vjp(
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
        bias_diff_idx=_split_inexact(bias_leaves),
        sm_scale=sm_scale,
        bq=bq,
        kc=kc,
        bm=bm,
        mods_kwargs=mods_kwargs,
        out_dtype=out_dtype,
    )
    o5 = fn(q5, kt, vt, rng_seed, dropout_mask, tuple(mask_leaves), tuple(bias_leaves))
    return finish(o5)
