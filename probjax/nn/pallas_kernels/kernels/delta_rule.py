"""Gated delta rule (Gated DeltaNet / DeltaNet) sequence mixer.

Per head, with state ``S_t`` of shape ``[Dk, Dv]`` (same orientation as
:func:`probjax.nn.pallas_kernels.ssd`):

.. math::

    S_t = \\alpha_t (I - \\beta_t k_t k_t^\\top) S_{t-1} + \\beta_t k_t v_t^\\top,
    \\qquad o_t = S_t^\\top q_t,

with ``alpha_t = exp(log_alpha_t)`` in ``(0, 1]`` and ``beta_t`` in ``(0, 1)``.
``log_alpha == 0`` gives plain DeltaNet (Schlag et al. 2021; Yang et al. 2024,
"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"),
a learned ``log_alpha < 0`` gives Gated DeltaNet (Yang et al. 2024, "Gated Delta
Networks: Improving Mamba2 with Delta Rule").

Implementations
---------------
* :func:`gated_delta_rule_recurrent` -- token-by-token ``lax.scan`` reference
  (fp32 state, ``O(L)`` sequential steps).
* :func:`gated_delta_rule_chunked` -- chunkwise-parallel form. Inside a chunk of
  ``C`` tokens the product of the ``C`` generalized Householder transitions is
  written in WY / UT form: with ``u_t = beta_t (v_t - alpha_t S_{t-1}^T k_t)`` the
  recurrence becomes ``S_t = alpha_t S_{t-1} + k_t u_t^T`` and the ``u`` of a
  chunk solve a unit lower-triangular system
  ``(I + A) U = beta V - diag(beta gamma) K S_0`` where
  ``A_ij = beta_i (k_i . k_j) gamma_i / gamma_j`` (``j < i``) and ``gamma`` is the
  within-chunk cumulative gate. Solving once per chunk (UT transform) gives
  ``U = W_v - W_k S_0`` with chunk-local ``W_v, W_k``; only the ``L / C``
  chunk-boundary states are then propagated sequentially. Every decay factor is
  ``exp`` of a non-positive number, so no division by the cumulative gate occurs.
  ``T = (I + A)^{-1}`` is formed explicitly by a block forward substitution
  (16 x 16 row substitution + pairwise block merges -- only small matmuls;
  XLA:CPU's batched ``triangular_solve`` is ~1000x slower).
  The whole op is one ``custom_vjp`` with a hand-written backward: residuals
  are only the inputs and the ``L / C`` chunk-boundary states
  (``L / C * Dk * Dv`` per head); the chunk-local WY quantities are recomputed
  (behind an optimization barrier, so XLA cannot CSE them with the forward),
  the state cotangent is propagated in one reverse scan and the chunk-local
  VJP is closed-form (``dA = -T^T dT T^T``). Nothing of size ``L * Dk * Dv``
  is ever materialized. This is the CPU path.
* :func:`gated_delta_rule_pallas` -- the same algorithm on GPU with
  Pallas-Triton kernels for ``(I + A)^{-1}`` and for the sequential
  forward / reverse state passes (optionally fused with the WY rebuild).
* :func:`gated_delta_rule` -- public entry point dispatching by backend.

Layout follows ``ssd``: ``q, k: [B, H, L, Dk]``, ``v: [B, H, L, Dv]``,
``log_alpha, beta: [B, H, L]``, ``h0: [B, H, Dk, Dv]``. All functions return
``(o, final_state)`` with ``o`` in the input dtype and ``final_state`` in fp32.
"""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "gated_delta_rule",
    "gated_delta_rule_chunked",
    "gated_delta_rule_pallas",
    "gated_delta_rule_recurrent",
    "l2_normalize",
]

Implementation = Literal["auto", "recurrent", "chunked", "pallas"]


def l2_normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalizes the last axis (computed in fp32, returned in ``x.dtype``)."""
    xf = x.astype(jnp.float32)
    inv = lax.rsqrt(jnp.sum(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv).astype(x.dtype)


def _validate(q, k, v, log_alpha, beta, h0):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"q, k, v must be rank-4 [B, H, L, D]; got {q.shape}, {k.shape}, {v.shape}."
        )
    if q.shape != k.shape:
        raise ValueError(f"q and k must have equal shapes, got {q.shape}, {k.shape}.")
    b, h, seq, _ = q.shape
    if v.shape[:3] != (b, h, seq):
        raise ValueError(
            f"v must be [B, H, L, Dv] = {(b, h, seq)} + (Dv,), got {v.shape}."
        )
    for name, x in (("log_alpha", log_alpha), ("beta", beta)):
        if x is not None and x.shape != (b, h, seq):
            raise ValueError(f"{name} must have shape {(b, h, seq)}, got {x.shape}.")
    if h0 is not None and h0.shape != (b, h, q.shape[-1], v.shape[-1]):
        raise ValueError(
            f"h0 must have shape {(b, h, q.shape[-1], v.shape[-1])}, got {h0.shape}."
        )


def _prepare(q, k, v, log_alpha, beta, h0, qk_l2norm):
    _validate(q, k, v, log_alpha, beta, h0)
    if qk_l2norm:
        q = l2_normalize(q)
        k = l2_normalize(k)
    b, h, seq, dk = q.shape
    dv = v.shape[-1]
    if log_alpha is None:
        log_alpha = jnp.zeros((b, h, seq), jnp.float32)
    if h0 is None:
        h0 = jnp.zeros((b, h, dk, dv), jnp.float32)
    return q, k, v, log_alpha, beta, h0


# --------------------------------------------------------------------------
# Recurrent reference
# --------------------------------------------------------------------------


def gated_delta_rule_recurrent(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array | None,
    beta: jax.Array,
    h0: jax.Array | None = None,
    *,
    qk_l2norm: bool = False,
    precision=lax.Precision.HIGHEST,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token ``lax.scan`` reference (fp32 state and arithmetic).

    Args:
        q, k: ``[B, H, L, Dk]``.
        v: ``[B, H, L, Dv]``.
        log_alpha: ``[B, H, L]`` log of the gate (``<= 0``); ``None`` means
            ``alpha == 1`` (plain DeltaNet).
        beta: ``[B, H, L]`` write strength in ``(0, 1)``.
        h0: optional initial state ``[B, H, Dk, Dv]``.
        qk_l2norm: L2-normalize ``q`` and ``k`` along the feature axis first.

    Returns:
        ``(o, final_state)`` with ``o: [B, H, L, Dv]`` (input dtype) and
        ``final_state: [B, H, Dk, Dv]`` (fp32).
    """
    dtype = q.dtype
    q, k, v, log_alpha, beta, h0 = _prepare(q, k, v, log_alpha, beta, h0, qk_l2norm)
    f32 = lambda x: x.astype(jnp.float32)  # noqa: E731
    q, k, v, log_alpha, beta, h0 = map(f32, (q, k, v, log_alpha, beta, h0))

    def step(s, xs):
        q_t, k_t, v_t, la_t, b_t = xs
        s = jnp.exp(la_t)[:, :, None, None] * s
        pred = jnp.einsum("bhkv,bhk->bhv", s, k_t, precision=precision)
        s = s + jnp.einsum(
            "bhk,bhv->bhkv", k_t, b_t[..., None] * (v_t - pred), precision=precision
        )
        o_t = jnp.einsum("bhkv,bhk->bhv", s, q_t, precision=precision)
        return s, o_t

    xs = (
        jnp.moveaxis(q, 2, 0),
        jnp.moveaxis(k, 2, 0),
        jnp.moveaxis(v, 2, 0),
        jnp.moveaxis(log_alpha, 2, 0),
        jnp.moveaxis(beta, 2, 0),
    )
    s, o = lax.scan(step, h0, xs)
    return jnp.moveaxis(o, 0, 2).astype(dtype), s


# --------------------------------------------------------------------------
# Chunked (WY / UT transform) form
# --------------------------------------------------------------------------


def _ein(spec, *ops, precision=None):
    return jnp.einsum(
        spec, *ops, precision=precision, preferred_element_type=jnp.float32
    )


def _diag_blocks(x: jax.Array, s: int) -> jax.Array:
    """``[..., C, C] -> [..., C // s, s, s]`` diagonal blocks."""
    *lead, c, _ = x.shape
    n = c // s
    xr = x.reshape(*lead, n, s, n, s)
    return jnp.moveaxis(jnp.diagonal(xr, axis1=-4, axis2=-2), -1, -3)


def _unit_lower_inverse(a: jax.Array, precision=None, base: int = 16) -> jax.Array:
    """``(I + a)^{-1}`` for strictly lower-triangular ``a: [..., C, C]``.

    Block forward substitution: ``base x base`` diagonal blocks are inverted by
    (vectorized, unrolled) row substitution, then merged pairwise with
    ``[[X, 0], [B, Y]]^{-1} = [[X^-1, 0], [-Y^-1 B X^-1, Y^-1]]``. Only small
    batched matmuls -- XLA:CPU's batched ``triangular_solve`` is ~1000x slower.
    ``C`` must be ``base * 2^j`` or at most ``base``.
    """
    c = a.shape[-1]
    s = min(base, c)
    ad = _diag_blocks(a, s)
    eye = jnp.eye(s, dtype=a.dtype)
    rows = [jnp.broadcast_to(eye[0], ad.shape[:-1])]
    for i in range(1, s):
        prev = jnp.stack(rows, axis=-2)  # [..., i, s]
        rows.append(
            eye[i] - _ein("...j,...jk->...k", ad[..., i, :i], prev, precision=precision)
        )
    inv = jnp.stack(rows, axis=-2)  # [..., C // s, s, s]
    while s < c:
        lower_left = _diag_blocks(a, 2 * s)[..., s:, :s]
        top, bot = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        off = -_ein(
            "...ij,...jk->...ik",
            _ein("...ij,...jk->...ik", bot, lower_left, precision=precision),
            top,
            precision=precision,
        )
        zeros = jnp.zeros_like(top)
        inv = jnp.concatenate(
            [jnp.concatenate([top, zeros], -1), jnp.concatenate([off, bot], -1)], -2
        )
        s *= 2
    return inv[..., 0, :, :]


def _unit_lower_inv(a, precision, use_pallas=False):
    """``(I + a)^{-1}`` for strictly lower-triangular ``a`` (explicit)."""
    c = a.shape[-1]
    if use_pallas and c % 16 == 0 and c <= 64:
        from . import delta_rule_pallas as _dp

        return _dp.unit_lower_inverse(a, precision=precision)
    if jax.default_backend() == "gpu" or (c > 16 and c & (c - 1)):
        # cuBLAS batched trsm is fast; also the fallback for odd chunk sizes.
        eye = jnp.eye(c, dtype=a.dtype)
        return lax.linalg.triangular_solve(
            a + eye,
            jnp.broadcast_to(eye, a.shape),
            left_side=True,
            lower=True,
            unit_diagonal=True,
        )
    return _unit_lower_inverse(a, precision)


def _chunk_gate(log_alpha, chunk_size):
    """Within-chunk cumulative log gate ``g: [B, H, N, C]`` (fp32, <= 0)."""
    b, h, seq = log_alpha.shape
    la = log_alpha.astype(jnp.float32).reshape(b, h, seq // chunk_size, chunk_size)
    return jnp.cumsum(la, axis=-1)


def _causal_decay(g):
    """``M_ij = exp(g_i - g_j)`` for ``j <= i``, else 0. ``g: [..., C]``."""
    c = g.shape[-1]
    idx = jnp.arange(c)
    causal = idx[:, None] >= idx[None, :]
    return jnp.exp(jnp.where(causal, g[..., :, None] - g[..., None, :], -jnp.inf))


def _chunks(x, c):
    b, h, seq = x.shape[:3]
    return x.astype(jnp.float32).reshape(b, h, seq // c, c, *x.shape[3:])


def _wy_inverse(k, g, beta, chunk_size, precision, use_pallas=False):
    """``T = (I + A)^{-1}`` with ``A_ij = beta_i (k_i . k_j) exp(g_i - g_j)``
    (``j < i``) for every chunk: ``[B, H, N, C, C]`` fp32."""
    c = chunk_size
    kc, bc = _chunks(k, c), _chunks(beta, c)
    idx = jnp.arange(c)
    strict = idx[:, None] > idx[None, :]
    kk = _ein("bhncd,bhnsd->bhncs", kc, kc, precision=precision)
    a_mat = jnp.where(strict, bc[..., :, None] * kk * _causal_decay(g), 0.0)
    return _unit_lower_inv(a_mat, precision, use_pallas)


def _wy_w(k, v, g, beta, t, chunk_size, precision):
    """``Wk = T (beta e^g K)``, ``Wv = T (beta V)`` per chunk (fp32)."""
    c = chunk_size
    kc, vc, bc = _chunks(k, c), _chunks(v, c), _chunks(beta, c)
    rk = kc * (bc * jnp.exp(g))[..., None]
    rv = vc * bc[..., None]
    wk = _ein("bhncs,bhnsd->bhncd", t, rk, precision=precision)
    wv = _ein("bhncs,bhnsd->bhncd", t, rv, precision=precision)
    return wk, wv


def _chunk_prep_xla(q, k, v, g, beta, chunk_size, precision, use_pallas=False):
    """Chunk-local (parallel) quantities of the WY representation.

    ``q, k: [B, H, L, Dk]``, ``v: [B, H, L, Dv]``, ``beta: [B, H, L]``,
    ``g: [B, H, N, C]``. Returns fp32 ``Wk, Wv, Kd, Qg: [B, H, N, C, D]`` and
    ``P, T: [B, H, N, C, C]`` with ``T = (I + A)^{-1}``.
    """
    c = chunk_size
    qc, kc = _chunks(q, c), _chunks(k, c)
    t = _wy_inverse(k, g, beta, c, precision, use_pallas)
    wk, wv = _wy_w(k, v, g, beta, t, c, precision)
    kd = kc * jnp.exp(g[..., -1:] - g)[..., None]
    qg = qc * jnp.exp(g)[..., None]
    p = _ein("bhncd,bhnsd->bhncs", qc, kc, precision=precision) * _causal_decay(g)
    return wk, wv, kd, qg, p, t


def _chunk_prep(q, k, v, g, beta, chunk_size, precision, use_pallas):
    return _chunk_prep_xla(q, k, v, g, beta, chunk_size, precision, use_pallas)


def _chunk_prep_vjp(q, k, v, g, beta, t, cts, chunk_size, precision):
    """Hand-written VJP of :func:`_chunk_prep_xla` given ``T = (I + A)^{-1}``.

    Returns cotangents of ``(q, k, v, g, beta)`` (``g`` chunked, fp32).
    """
    d_wk, d_wv, d_kd, d_qg, d_p, d_a = cts
    c = chunk_size
    qc, kc, vc, bc = (_chunks(x, c) for x in (q, k, v, beta))
    e = jnp.exp(g)
    m = _causal_decay(g)
    idx = jnp.arange(c)
    strict = idx[:, None] > idx[None, :]
    ein = functools.partial(_ein, precision=precision)
    rk = kc * (bc * e)[..., None]
    rv = vc * bc[..., None]
    # W = T R
    d_rk = ein("bhnsc,bhnsd->bhncd", t, d_wk)
    d_rv = ein("bhnsc,bhnsd->bhncd", t, d_wv)
    d_t = ein("bhncd,bhnsd->bhncs", d_wk, rk) + ein("bhncd,bhnsd->bhncs", d_wv, rv)
    # T = (I + A)^-1  =>  dA = -T^T dT T^T (strictly lower part).
    d_a_mat = -ein("bhnjc,bhnjs->bhncs", t, ein("bhncj,bhnsj->bhncs", d_t, t))
    d_a_mat = jnp.where(strict, d_a_mat, 0.0)
    kk = ein("bhncd,bhnsd->bhncs", kc, kc)
    qk = ein("bhncd,bhnsd->bhncs", qc, kc)
    d_kk = d_a_mat * bc[..., :, None] * m
    d_pm = d_p * m
    d_m = d_a_mat * bc[..., :, None] * kk + d_p * qk
    dm_m = d_m * m
    d_q = ein("bhncs,bhnsd->bhncd", d_pm, kc) + d_qg * e[..., None]
    w_kd = jnp.sum(d_kd * kc, axis=-1) * jnp.exp(g[..., -1:] - g)
    d_k = (
        ein("bhncs,bhnsd->bhncd", d_kk + jnp.swapaxes(d_kk, -1, -2), kc)
        + ein("bhncs,bhncd->bhnsd", d_pm, qc)
        + d_rk * (bc * e)[..., None]
        + d_kd * jnp.exp(g[..., -1:] - g)[..., None]
    )
    d_v = d_rv * bc[..., None]
    rk_dot = jnp.sum(d_rk * kc, axis=-1)
    d_beta = (
        jnp.sum(d_a_mat * kk * m, axis=-1) + rk_dot * e + jnp.sum(d_rv * vc, axis=-1)
    )
    d_g = (
        jnp.sum(dm_m, axis=-1)
        - jnp.sum(dm_m, axis=-2)
        + rk_dot * bc * e
        + jnp.sum(d_qg * qc, axis=-1) * e
        - w_kd
    )
    d_glast = jnp.sum(w_kd, axis=-1) + d_a * jnp.exp(g[..., -1])
    d_g = d_g.at[..., -1].add(d_glast)

    def unchunk(x, like):
        return x.reshape(like.shape).astype(like.dtype)

    return (
        unchunk(d_q, q),
        unchunk(d_k, k),
        unchunk(d_v, v),
        d_g,
        unchunk(d_beta, beta),
    )


# ---- inter-chunk state pass: pure-JAX implementation ----------------------


def _state_pass_fwd_scan(wk, wv, kd, qg, p, a, s0, precision):
    """Sequential state pass. Returns (o, states_before_each_chunk, final)."""

    def step(s, xs):
        wk_n, wv_n, kd_n, qg_n, p_n, a_n = xs
        u = wv_n - _ein("bhck,bhkv->bhcv", wk_n, s, precision=precision)
        o = _ein("bhck,bhkv->bhcv", qg_n, s, precision=precision) + _ein(
            "bhcs,bhsv->bhcv", p_n, u, precision=precision
        )
        s_new = a_n[..., None, None] * s + _ein(
            "bhck,bhcv->bhkv", kd_n, u, precision=precision
        )
        return s_new, (o, s)

    xs = tuple(jnp.moveaxis(x, 2, 0) for x in (wk, wv, kd, qg, p, a))
    s_fin, (o, states) = lax.scan(step, s0, xs)
    return jnp.moveaxis(o, 0, 2), jnp.moveaxis(states, 0, 2), s_fin


def _state_pass_bwd_scan(wk, kd, qg, p, a, do, ds_fin, precision):
    """Reverse sequential pass. Returns ``(du, ds_after, ds0)`` where
    ``ds_after[n]`` is the cotangent of the state leaving chunk ``n``."""

    def step(ds, xs):
        wk_n, kd_n, qg_n, p_n, a_n, do_n = xs
        du = _ein("bhcs,bhcv->bhsv", p_n, do_n, precision=precision) + _ein(
            "bhck,bhkv->bhcv", kd_n, ds, precision=precision
        )
        ds_prev = (
            a_n[..., None, None] * ds
            + _ein("bhck,bhcv->bhkv", qg_n, do_n, precision=precision)
            - _ein("bhck,bhcv->bhkv", wk_n, du, precision=precision)
        )
        return ds_prev, (du, ds)

    xs = tuple(jnp.moveaxis(x, 2, 0) for x in (wk, kd, qg, p, a, do))
    ds0, (du, ds_after) = lax.scan(step, ds_fin, xs, reverse=True)
    return jnp.moveaxis(du, 0, 2), jnp.moveaxis(ds_after, 0, 2), ds0


def _state_pass_param_grads(wk, wv, states, do, du, ds_after, precision):
    """Parallel (all chunks at once) parameter cotangents of the state pass."""
    # U is cheap to recompute from the stored chunk-boundary states.
    u = wv - _ein("bhnck,bhnkv->bhncv", wk, states, precision=precision)
    dwk = -_ein("bhncv,bhnkv->bhnck", du, states, precision=precision)
    dkd = _ein("bhncv,bhnkv->bhnck", u, ds_after, precision=precision)
    dqg = _ein("bhncv,bhnkv->bhnck", do, states, precision=precision)
    dp = _ein("bhncv,bhnsv->bhncs", do, u, precision=precision)
    da = jnp.sum(states * ds_after, axis=(-2, -1))
    return dwk, du, dkd, dqg, dp, da


def _state_pass_fwd(wk, wv, kd, qg, p, a, s0, precision, use_pallas):
    if use_pallas:
        from . import delta_rule_pallas as _dp

        return _dp.state_pass_fwd(wk, wv, kd, qg, p, a, s0, precision=precision)
    return _state_pass_fwd_scan(wk, wv, kd, qg, p, a, s0, precision)


def _state_pass_bwd(wk, kd, qg, p, a, do, ds_fin, precision, use_pallas):
    if use_pallas:
        from . import delta_rule_pallas as _dp

        return _dp.state_pass_bwd_seq(wk, kd, qg, p, a, do, ds_fin, precision=precision)
    return _state_pass_bwd_scan(wk, kd, qg, p, a, do, ds_fin, precision)


# ---- end-to-end chunked op with a hand-written, memory-lean VJP ------------


def _fused_ok(mode, chunk_size, dk, dv, precision):
    """Whether to use the fused Pallas state kernels (WY rebuilt on chip)."""
    if mode not in ("fused", "auto"):
        return False
    from . import delta_rule_pallas as _dp

    # fp32 (HIGHEST) dots need more shared memory than a 3090 block has at Dk=128.
    highest = precision in ("highest", lax.Precision.HIGHEST)
    if not _dp.prep_supported(chunk_size, dk, dv) or (highest and dk > 64):
        return False
    # Measured on a 3090: at Dk = 128 the fused kernels spill (bf16 fwd 21 ms vs
    # 5.7 ms unfused at B4 H8 L8192), at Dk = 64 they win (1.9 vs 2.6 ms).
    return mode == "fused" or dk <= 64


def _forward(q, k, v, log_alpha, beta, h0, chunk_size, precision, mode):
    g = _chunk_gate(log_alpha, chunk_size)
    if _fused_ok(mode, chunk_size, q.shape[-1], v.shape[-1], precision):
        from . import delta_rule_pallas as _dp

        t = _wy_inverse(k, g, beta, chunk_size, precision, use_pallas=True)
        ch = functools.partial(_chunk_view, c=chunk_size)
        o, states, s_fin = _dp.fused_state_pass_fwd(
            ch(q), ch(k), ch(v), g, ch(beta), t, h0, precision=precision
        )
        return o, s_fin, states, g, (t,)
    use_pallas = mode in ("pallas", "fused", "auto")
    prep = _chunk_prep(q, k, v, g, beta, chunk_size, precision, use_pallas)
    wk, wv, kd, qg, p, t = prep
    a = jnp.exp(g[..., -1])
    o, states, s_fin = _state_pass_fwd(wk, wv, kd, qg, p, a, h0, precision, use_pallas)
    return o, s_fin, states, g, prep


def _chunk_view(x, c):
    """``[B, H, L, ...] -> [B, H, N, C, ...]`` without changing the dtype."""
    b, h, seq = x.shape[:3]
    return x.reshape(b, h, seq // c, c, *x.shape[3:])


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _delta_core(q, k, v, log_alpha, beta, h0, chunk_size, precision, mode, store_prep):
    o, s_fin, *_ = _forward(q, k, v, log_alpha, beta, h0, chunk_size, precision, mode)
    return o, s_fin


def _delta_core_fwd(
    q, k, v, log_alpha, beta, h0, chunk_size, precision, mode, store_prep
):
    o, s_fin, states, g, prep = _forward(
        q, k, v, log_alpha, beta, h0, chunk_size, precision, mode
    )
    # Residuals: the inputs plus the L / C chunk-boundary states; the
    # chunk-local WY quantities are recomputed in the backward pass unless
    # ``store_prep`` (benchmark baseline that keeps them).
    return (o, s_fin), (q, k, v, log_alpha, beta, states, prep if store_prep else None)


def _delta_core_bwd(chunk_size, precision, mode, store_prep, res, cts):
    q, k, v, log_alpha, beta, states, prep = res
    do, ds_fin = cts
    do = jnp.asarray(do, jnp.float32)
    ds_fin = jnp.asarray(ds_fin, jnp.float32)
    g = _chunk_gate(log_alpha, chunk_size)
    fused = _fused_ok(mode, chunk_size, q.shape[-1], v.shape[-1], precision)
    use_pallas = mode in ("pallas", "fused", "auto")
    if prep is None:
        # The barrier stops XLA from CSE-ing this recompute with the forward
        # prep (which would keep every chunk-local intermediate alive).
        qb, kb, vb, gb, bb = lax.optimization_barrier((q, k, v, g, beta))
        if fused:
            prep = (_wy_inverse(kb, gb, bb, chunk_size, precision, use_pallas=True),)
        else:
            prep = _chunk_prep(qb, kb, vb, gb, bb, chunk_size, precision, use_pallas)
    a = jnp.exp(g[..., -1])
    if fused:
        from . import delta_rule_pallas as _dp

        (t,) = prep
        ch = functools.partial(_chunk_view, c=chunk_size)
        du, ds_after, ds0 = _dp.fused_state_pass_bwd(
            ch(q), ch(k), g, ch(beta), t, do, ds_fin, precision=precision
        )
        wk, wv = _wy_w(k, v, g, beta, t, chunk_size, precision)
    else:
        wk, wv, kd, qg, p, t = prep
        du, ds_after, ds0 = _state_pass_bwd(
            wk, kd, qg, p, a, do, ds_fin, precision, use_pallas
        )
    cts_prep = _state_pass_param_grads(wk, wv, states, do, du, ds_after, precision)
    d_q, d_k, d_v, d_g, d_beta = _chunk_prep_vjp(
        q, k, v, g, beta, t, cts_prep, chunk_size, precision
    )
    # g = cumsum(log_alpha) within chunks  =>  reverse cumsum.
    d_la = jnp.flip(jnp.cumsum(jnp.flip(d_g, -1), -1), -1)
    d_la = d_la.reshape(log_alpha.shape).astype(log_alpha.dtype)
    return d_q, d_k, d_v, d_la, d_beta, ds0


_delta_core.defvjp(_delta_core_fwd, _delta_core_bwd)


def _chunked_impl(
    q,
    k,
    v,
    log_alpha,
    beta,
    h0,
    *,
    chunk_size,
    qk_l2norm,
    precision,
    mode,
    store_prep=False,
):
    dtype = q.dtype
    q, k, v, log_alpha, beta, h0 = _prepare(q, k, v, log_alpha, beta, h0, qk_l2norm)
    b, h, seq, _ = q.shape
    dv = v.shape[-1]
    c = int(chunk_size)
    if c <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    pad = (-seq) % c
    if pad:
        # beta = 0 and log_alpha = 0 on padded steps leave the state unchanged.
        cfg = ((0, 0), (0, 0), (0, pad))
        q, k, v = (jnp.pad(x, cfg + ((0, 0),)) for x in (q, k, v))
        log_alpha = jnp.pad(log_alpha, cfg)
        beta = jnp.pad(beta, cfg)
    o, s_fin = _delta_core(
        q, k, v, log_alpha, beta, h0.astype(jnp.float32), c, precision, mode,
        store_prep,
    )  # fmt: skip
    o = o.reshape(b, h, seq + pad, dv)[:, :, :seq]
    return o.astype(dtype), s_fin


def gated_delta_rule_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array | None,
    beta: jax.Array,
    h0: jax.Array | None = None,
    *,
    chunk_size: int = 64,
    qk_l2norm: bool = False,
    precision=None,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise-parallel gated delta rule in pure JAX (the CPU path).

    Same contract as :func:`gated_delta_rule_recurrent`. ``L`` need not be a
    multiple of ``chunk_size`` (inputs are padded with ``beta = 0`` steps).
    ``precision`` is forwarded to the matmuls (``None`` = backend default,
    which is TF32 for fp32 inputs on Ampere GPUs).
    """
    return _chunked_impl(
        q,
        k,
        v,
        log_alpha,
        beta,
        h0,
        chunk_size=chunk_size,
        qk_l2norm=qk_l2norm,
        precision=precision,
        mode="xla",
    )


def gated_delta_rule_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array | None,
    beta: jax.Array,
    h0: jax.Array | None = None,
    *,
    chunk_size: int = 64,
    qk_l2norm: bool = False,
    precision=None,
    mode: Literal["auto", "fused", "pallas"] = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Chunked gated delta rule with Pallas-Triton kernels (GPU).

    ``(I + A)^{-1}`` always comes from a Pallas block forward substitution.
    ``mode="fused"``: sequential state kernels rebuild the WY quantities per
    chunk from ``q, k, v`` (nothing chunk-local is written to HBM in the
    forward pass). ``mode="pallas"``: WY quantities from XLA einsums, Pallas
    state kernels only. ``"auto"``: fused for ``Dk <= 64``, else ``"pallas"``.
    Shapes the kernels do not support fall back to the XLA pieces.
    """
    return _chunked_impl(
        q,
        k,
        v,
        log_alpha,
        beta,
        h0,
        chunk_size=chunk_size,
        qk_l2norm=qk_l2norm,
        precision=precision,
        mode=mode,
    )


@jax.named_call
def gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array | None,
    beta: jax.Array,
    h0: jax.Array | None = None,
    *,
    chunk_size: int = 64,
    qk_l2norm: bool = False,
    implementation: Implementation = "auto",
    precision=None,
) -> tuple[jax.Array, jax.Array]:
    """Gated delta rule ``S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T``.

    Args:
        q, k: ``[B, H, L, Dk]`` queries / keys (L2-normalize them, e.g. via
            ``qk_l2norm=True``, for a contractive transition).
        v: ``[B, H, L, Dv]`` values.
        log_alpha: ``[B, H, L]`` log-gate ``<= 0`` or ``None`` (DeltaNet).
        beta: ``[B, H, L]`` write strengths in ``(0, 1)``.
        h0: optional initial state ``[B, H, Dk, Dv]``.
        chunk_size: tokens per chunk of the chunkwise-parallel form.
        qk_l2norm: L2-normalize ``q`` and ``k`` inside the op.
        implementation: ``"recurrent"`` (token scan), ``"chunked"`` (pure
            JAX, default on CPU/TPU), ``"pallas"`` (Pallas-Triton kernels,
            GPU) or ``"auto"`` (``"pallas"`` on GPU when ``chunk_size`` is
            16/32/64, ``Dk`` a power of two in ``[16, 128]`` and ``Dv`` a power
            of two ``>= 16``, else
            ``"chunked"``).
        precision: matmul precision for the chunked forms.

    Returns:
        ``(o, final_state)``: ``[B, H, L, Dv]`` in ``q.dtype`` and
        ``[B, H, Dk, Dv]`` in fp32.
    """
    kwargs = dict(qk_l2norm=qk_l2norm)
    if implementation == "auto":
        implementation = _auto_implementation(q, v, chunk_size)
    if implementation == "recurrent":
        return gated_delta_rule_recurrent(q, k, v, log_alpha, beta, h0, **kwargs)
    if implementation == "chunked":
        return gated_delta_rule_chunked(
            q,
            k,
            v,
            log_alpha,
            beta,
            h0,
            chunk_size=chunk_size,
            precision=precision,
            **kwargs,
        )
    if implementation == "pallas":
        return gated_delta_rule_pallas(
            q,
            k,
            v,
            log_alpha,
            beta,
            h0,
            chunk_size=chunk_size,
            precision=precision,
            **kwargs,
        )
    raise ValueError(f"Unknown implementation {implementation!r}.")


def _auto_implementation(q, v, chunk_size) -> str:
    """Pallas on GPU for supported shapes, else the pure-JAX chunked form."""
    if jax.default_backend() == "gpu":
        from . import delta_rule_pallas as _dp

        if _dp.prep_supported(chunk_size, q.shape[-1], v.shape[-1]):
            return "pallas"
    return "chunked"
