"""Pallas-Triton kernels for the gated delta rule (GPU).

See :mod:`probjax.nn.pallas_kernels.kernels.delta_rule` for the algorithm.

* :func:`unit_lower_inverse` -- ``(I + A)^-1`` for the per-chunk strictly
  lower-triangular ``A`` by 16-row block forward substitution (fully parallel;
  ~10x faster than cuBLAS batched trsm on a 3090).
* :func:`state_pass_fwd` / :func:`state_pass_bwd_seq` -- the only sequential
  part, the recurrence over chunk-boundary states, given the WY quantities.
* :func:`fused_state_pass_fwd` / :func:`fused_state_pass_bwd` -- the same, but
  rebuilding ``Wk, Wv, Kd, Qg, P`` per chunk on chip from ``q, k, v, g, beta``
  and ``T`` (no chunk-local HBM round trip).

The ``[Dk, bv]`` state tile stays resident for the whole sequence. Columns of
the state (the ``Dv`` axis) evolve independently, so the grid of the state
kernels is ``(B * H, Dv / bv)``.

Forward, per chunk ``n`` (``S`` = state entering the chunk)::

    U = Wv - Wk S;   O = Qg S + P U;   S <- a S + Kd^T U

Backward (reverse over chunks, ``dS`` = cotangent of the state leaving the
chunk)::

    dU = P^T dO + Kd dS;   dS <- a dS + Qg^T dO - Wk^T dU
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental import pallas as pl

from ..kernel_utils import pallas_call_compat, use_interpret_mode

_load = pallas_primitives.load
_store = pallas_primitives.store


def _is_pow2(x: int) -> bool:
    return x >= 16 and (x & (x - 1)) == 0


def supports_shape(chunk_size: int, dk: int, dv: int) -> bool:
    """Whether the Triton kernels support this (chunk, Dk, Dv) configuration."""
    return (
        _is_pow2(chunk_size)
        and _is_pow2(dk)
        and _is_pow2(dv)
        and chunk_size <= 128
        and dk <= 128
    )


def _block_v(dk: int, dv: int, chunk: int) -> int:
    del dk, chunk
    # Tuned on an RTX 3090 (B4 H8 L8192, Dk = Dv in {64, 128}, 4 warps): 32-wide
    # value tiles beat 64 (D128: fwd 3.35 -> 2.55 ms, bwd 3.52 -> 2.29 ms) by
    # doubling the number of programs and halving the resident state tile.
    return min(dv, 32)


def _dot(a, b, precision):
    return jnp.dot(a, b, precision=precision, preferred_element_type=jnp.float32)


def _fwd_kernel(
    wk_ref, wv_ref, kd_ref, qg_ref, p_ref, a_ref, s0_ref,
    o_ref, states_ref, sfin_ref, *, num_chunks, precision,
):  # fmt: skip
    def body(n, s):
        idx = (n, slice(None), slice(None))
        _store(states_ref, idx, s)
        wk = _load(wk_ref, idx)
        wv = _load(wv_ref, idx)
        u = wv - _dot(wk, s, precision)
        qg = _load(qg_ref, idx)
        p = _load(p_ref, idx)
        o = _dot(qg, s, precision) + _dot(p, u, precision)
        _store(o_ref, idx, o.astype(o_ref.dtype))
        kd = _load(kd_ref, idx)
        a = _load(a_ref, (n, slice(None)))  # [Dk], all entries equal a_n
        return a[:, None] * s + _dot(kd.T, u, precision)

    s = jax.lax.fori_loop(0, num_chunks, body, s0_ref[...].astype(jnp.float32))
    sfin_ref[...] = s


def _bwd_kernel(
    wk_ref, kd_ref, qg_ref, p_ref, a_ref, do_ref, dsfin_ref,
    du_ref, dsafter_ref, ds0_ref, *, num_chunks, precision,
):  # fmt: skip
    def body(i, ds):
        n = num_chunks - 1 - i
        idx = (n, slice(None), slice(None))
        _store(dsafter_ref, idx, ds)
        do = _load(do_ref, idx).astype(jnp.float32)
        p = _load(p_ref, idx)
        kd = _load(kd_ref, idx)
        du = _dot(p.T, do, precision) + _dot(kd, ds, precision)
        _store(du_ref, idx, du)
        qg = _load(qg_ref, idx)
        wk = _load(wk_ref, idx)
        a = _load(a_ref, (n, slice(None)))
        return a[:, None] * ds + _dot(qg.T, do, precision) - _dot(wk.T, du, precision)

    ds = jax.lax.fori_loop(0, num_chunks, body, dsfin_ref[...].astype(jnp.float32))
    ds0_ref[...] = ds


def _flat(x):
    return x.reshape((-1,) + x.shape[2:])


def _kernel_precision(precision):
    if precision in ("highest", jax.lax.Precision.HIGHEST):
        return jax.lax.Precision.HIGHEST
    return None


def _call_kwargs(num_warps: int, num_stages: int):
    kwargs = dict(interpret=use_interpret_mode())
    if jax.default_backend() == "gpu" and not kwargs["interpret"]:
        from jax.experimental.pallas import triton as plgpu

        params_cls = (
            getattr(plgpu, "CompilerParams", None) or plgpu.TritonCompilerParams
        )
        kwargs["compiler_params"] = params_cls(
            num_warps=num_warps, num_stages=num_stages
        )
    return kwargs


@functools.partial(jax.jit, static_argnames=("precision", "num_warps", "block_v"))
def state_pass_fwd(
    wk, wv, kd, qg, p, a, s0, *, precision=None, num_warps=4, block_v=None
):
    """Forward inter-chunk pass. Shapes ``[B, H, N, C, D]``; returns
    ``(o [B,H,N,C,Dv], states [B,H,N,Dk,Dv], final [B,H,Dk,Dv])`` in fp32."""
    b, h, n, c, dk = wk.shape
    dv = wv.shape[-1]
    bv = block_v or _block_v(dk, dv, c)
    prec = _kernel_precision(precision)
    a_rows = jnp.broadcast_to(a[..., None], (b, h, n, dk))
    seq = lambda d, tiled: pl.BlockSpec(  # noqa: E731
        (None, n, c, bv if tiled else d),
        (lambda i, j: (i, 0, 0, j)) if tiled else (lambda i, j: (i, 0, 0, 0)),
    )
    in_specs = [
        seq(dk, False),  # wk
        seq(dv, True),  # wv
        seq(dk, False),  # kd
        seq(dk, False),  # qg
        seq(c, False),  # p
        pl.BlockSpec((None, n, dk), lambda i, j: (i, 0, 0)),  # a rows
        pl.BlockSpec((None, dk, bv), lambda i, j: (i, 0, j)),  # s0
    ]
    out_specs = [
        seq(dv, True),
        pl.BlockSpec((None, n, dk, bv), lambda i, j: (i, 0, 0, j)),
        pl.BlockSpec((None, dk, bv), lambda i, j: (i, 0, j)),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((b * h, n, c, dv), jnp.float32),
        jax.ShapeDtypeStruct((b * h, n, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((b * h, dk, dv), jnp.float32),
    ]
    kernel = functools.partial(_fwd_kernel, num_chunks=n, precision=prec)
    o, states, sfin = pallas_call_compat(
        kernel,
        grid=(b * h, dv // bv),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        **_call_kwargs(num_warps, 1),
    )(*map(_flat, (wk, wv, kd, qg, p, a_rows, s0)))
    return (
        o.reshape(b, h, n, c, dv),
        states.reshape(b, h, n, dk, dv),
        sfin.reshape(b, h, dk, dv),
    )


@functools.partial(jax.jit, static_argnames=("precision", "num_warps", "block_v"))
def state_pass_bwd_seq(
    wk, kd, qg, p, a, do, ds_fin, *, precision=None, num_warps=4, block_v=None
):
    """Reverse inter-chunk pass. Returns ``(du, ds_after, ds0)``."""
    b, h, n, c, dk = wk.shape
    dv = do.shape[-1]
    bv = block_v or _block_v(dk, dv, c)
    prec = _kernel_precision(precision)
    a_rows = jnp.broadcast_to(a[..., None], (b, h, n, dk))
    seq = lambda d, tiled: pl.BlockSpec(  # noqa: E731
        (None, n, c, bv if tiled else d),
        (lambda i, j: (i, 0, 0, j)) if tiled else (lambda i, j: (i, 0, 0, 0)),
    )
    in_specs = [
        seq(dk, False),  # wk
        seq(dk, False),  # kd
        seq(dk, False),  # qg
        seq(c, False),  # p
        pl.BlockSpec((None, n, dk), lambda i, j: (i, 0, 0)),
        seq(dv, True),  # do
        pl.BlockSpec((None, dk, bv), lambda i, j: (i, 0, j)),
    ]
    out_specs = [
        seq(dv, True),
        pl.BlockSpec((None, n, dk, bv), lambda i, j: (i, 0, 0, j)),
        pl.BlockSpec((None, dk, bv), lambda i, j: (i, 0, j)),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((b * h, n, c, dv), jnp.float32),
        jax.ShapeDtypeStruct((b * h, n, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((b * h, dk, dv), jnp.float32),
    ]
    kernel = functools.partial(_bwd_kernel, num_chunks=n, precision=prec)
    du, ds_after, ds0 = pallas_call_compat(
        kernel,
        grid=(b * h, dv // bv),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        **_call_kwargs(num_warps, 1),
    )(*map(_flat, (wk, kd, qg, p, a_rows, do, ds_fin)))
    return (
        du.reshape(b, h, n, c, dv),
        ds_after.reshape(b, h, n, dk, dv),
        ds0.reshape(b, h, dk, dv),
    )


# --------------------------------------------------------------------------
# Chunk-local prep (WY representation) -- fully parallel over (B*H, N)
# --------------------------------------------------------------------------

_SUB = 16  # row-block size of the block forward substitution


def prep_supported(chunk: int, dk: int, dv: int) -> bool:
    return chunk % _SUB == 0 and chunk <= 64 and supports_shape(chunk, dk, dv)


def _tril_inv_kernel(a_ref, t_ref, *, chunk, precision):
    """``T = (I + A)^-1`` for one strictly lower-triangular ``[C, C]`` block."""
    nb = chunk // _SUB
    f32 = jnp.float32
    rows = jax.lax.broadcasted_iota(jnp.int32, (_SUB, _SUB), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (_SUB, _SUB), 1)
    eye = (rows == cols).astype(f32)

    def a_blk(i, j):
        return _load(a_ref, (pl.ds(i * _SUB, _SUB), pl.ds(j * _SUB, _SUB)))

    t_blk = {}
    for i in range(nb):
        a = a_blk(i, i)
        t = eye
        for r in range(1, _SUB):
            t = t - _dot(jnp.where(rows == r, a, 0.0), t, precision)
        t_blk[i, i] = t
    for i in range(nb):
        for j in range(i - 1, -1, -1):
            acc = _dot(a_blk(i, j), t_blk[j, j], precision)
            for m in range(j + 1, i):
                acc = acc + _dot(a_blk(i, m), t_blk[m, j], precision)
            t_blk[i, j] = -_dot(t_blk[i, i], acc, precision)
    zero = jnp.zeros((_SUB, _SUB), f32)
    for i in range(nb):
        for j in range(nb):
            _store(
                t_ref,
                (pl.ds(i * _SUB, _SUB), pl.ds(j * _SUB, _SUB)),
                t_blk[i, j] if j <= i else zero,
            )


@functools.partial(jax.jit, static_argnames=("precision", "num_warps", "num_stages"))
def unit_lower_inverse(a, *, precision=None, num_warps=1, num_stages=1):
    """``(I + a)^-1`` for strictly lower-triangular fp32 ``a: [..., C, C]``
    (``C`` a multiple of 16, at most 64) by 16-row block forward substitution."""
    lead = a.shape[:-2]
    c = a.shape[-1]
    flat = a.reshape(-1, c, c).astype(jnp.float32)
    spec = pl.BlockSpec((None, c, c), lambda i: (i, 0, 0))
    kernel = functools.partial(
        _tril_inv_kernel, chunk=c, precision=_kernel_precision(precision)
    )
    t = pallas_call_compat(
        kernel,
        grid=(flat.shape[0],),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(flat.shape, jnp.float32),
        **_call_kwargs(num_warps, num_stages),
    )(flat)
    return t.reshape(*lead, c, c)


# --------------------------------------------------------------------------
# Fused state pass: WY quantities rebuilt per chunk from q, k, v, g, beta, T
# --------------------------------------------------------------------------


def _wy_chunk(q_ref, k_ref, g_ref, b_ref, t_ref, n, precision):
    """Chunk ``n`` quantities that do not depend on the value tile."""
    f32 = jnp.float32
    idx = (n, slice(None), slice(None))
    qn = _load(q_ref, idx)
    kn = _load(k_ref, idx)
    gn = _load(g_ref, (n, slice(None)))
    bn = _load(b_ref, (n, slice(None)))
    tn = _load(t_ref, idx)
    en = jnp.exp(gn)
    g_last = jnp.min(gn)  # g is non-increasing within a chunk
    kf = kn.astype(f32)
    wk = _dot(tn, kf * (bn * en)[:, None], precision)
    kd = kf * jnp.exp(g_last - gn)[:, None]
    qg = qn.astype(f32) * en[:, None]
    c = gn.shape[0]
    rows = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
    m = jnp.exp(jnp.where(rows >= cols, gn[:, None] - gn[None, :], -jnp.inf))
    p = _dot(qn, kn.T, precision) * m
    return wk, kd, qg, p, tn, bn, jnp.exp(g_last)


def _fused_fwd_kernel(
    q_ref, k_ref, v_ref, g_ref, b_ref, t_ref, s0_ref,
    o_ref, states_ref, sfin_ref, *, num_chunks, precision,
):  # fmt: skip
    def body(n, s):
        idx = (n, slice(None), slice(None))
        _store(states_ref, idx, s)
        wk, kd, qg, p, tn, bn, a = _wy_chunk(
            q_ref, k_ref, g_ref, b_ref, t_ref, n, precision
        )
        vn = _load(v_ref, idx).astype(jnp.float32)
        wv = _dot(tn, vn * bn[:, None], precision)
        u = wv - _dot(wk, s, precision)
        o = _dot(qg, s, precision) + _dot(p, u, precision)
        _store(o_ref, idx, o.astype(o_ref.dtype))
        return a * s + _dot(kd.T, u, precision)

    s = jax.lax.fori_loop(0, num_chunks, body, s0_ref[...].astype(jnp.float32))
    sfin_ref[...] = s


def _fused_bwd_kernel(
    q_ref, k_ref, g_ref, b_ref, t_ref, do_ref, dsfin_ref,
    du_ref, dsafter_ref, ds0_ref, *, num_chunks, precision,
):  # fmt: skip
    def body(i, ds):
        n = num_chunks - 1 - i
        idx = (n, slice(None), slice(None))
        _store(dsafter_ref, idx, ds)
        wk, kd, qg, p, _, _, a = _wy_chunk(
            q_ref, k_ref, g_ref, b_ref, t_ref, n, precision
        )
        do = _load(do_ref, idx).astype(jnp.float32)
        du = _dot(p.T, do, precision) + _dot(kd, ds, precision)
        _store(du_ref, idx, du)
        return a * ds + _dot(qg.T, do, precision) - _dot(wk.T, du, precision)

    ds = jax.lax.fori_loop(0, num_chunks, body, dsfin_ref[...].astype(jnp.float32))
    ds0_ref[...] = ds


def _fused_specs(n, c, dk, dv, bv):
    full = lambda d: pl.BlockSpec((None, n, c, d), lambda i, j: (i, 0, 0, 0))  # noqa: E731
    tiled = pl.BlockSpec((None, n, c, bv), lambda i, j: (i, 0, 0, j))
    row = pl.BlockSpec((None, n, c), lambda i, j: (i, 0, 0))
    state = pl.BlockSpec((None, dk, bv), lambda i, j: (i, 0, j))
    states = pl.BlockSpec((None, n, dk, bv), lambda i, j: (i, 0, 0, j))
    return full, tiled, row, state, states


@functools.partial(jax.jit, static_argnames=("precision", "num_warps", "block_v"))
def fused_state_pass_fwd(
    q, k, v, g, beta, t, s0, *, precision=None, num_warps=4, block_v=None
):
    """Forward state pass from raw inputs. ``q, k: [B,H,N,C,Dk]``,
    ``v: [B,H,N,C,Dv]``, ``g, beta: [B,H,N,C]`` (fp32), ``t: [B,H,N,C,C]``.
    Returns ``(o fp32 [B,H,N,C,Dv], states [B,H,N,Dk,Dv], final [B,H,Dk,Dv])``."""
    b, h, n, c, dk = q.shape
    dv = v.shape[-1]
    bv = block_v or _block_v(dk, dv, c)
    full, tiled, row, state, states = _fused_specs(n, c, dk, dv, bv)
    kernel = functools.partial(
        _fused_fwd_kernel, num_chunks=n, precision=_kernel_precision(precision)
    )
    o, st, sfin = pallas_call_compat(
        kernel,
        grid=(b * h, dv // bv),
        in_specs=[full(dk), full(dk), tiled, row, row, full(c), state],
        out_specs=[tiled, states, state],
        out_shape=[
            jax.ShapeDtypeStruct((b * h, n, c, dv), jnp.float32),
            jax.ShapeDtypeStruct((b * h, n, dk, dv), jnp.float32),
            jax.ShapeDtypeStruct((b * h, dk, dv), jnp.float32),
        ],
        **_call_kwargs(num_warps, 1),
    )(*map(_flat, (q, k, v, g, beta, t, s0)))
    return (
        o.reshape(b, h, n, c, dv),
        st.reshape(b, h, n, dk, dv),
        sfin.reshape(b, h, dk, dv),
    )


@functools.partial(jax.jit, static_argnames=("precision", "num_warps", "block_v"))
def fused_state_pass_bwd(
    q, k, g, beta, t, do, ds_fin, *, precision=None, num_warps=4, block_v=None
):
    """Reverse state pass from raw inputs. Returns ``(du, ds_after, ds0)``."""
    b, h, n, c, dk = q.shape
    dv = do.shape[-1]
    bv = block_v or _block_v(dk, dv, c)
    full, tiled, row, state, states = _fused_specs(n, c, dk, dv, bv)
    kernel = functools.partial(
        _fused_bwd_kernel, num_chunks=n, precision=_kernel_precision(precision)
    )
    du, ds_after, ds0 = pallas_call_compat(
        kernel,
        grid=(b * h, dv // bv),
        in_specs=[full(dk), full(dk), row, row, full(c), tiled, state],
        out_specs=[tiled, states, state],
        out_shape=[
            jax.ShapeDtypeStruct((b * h, n, c, dv), jnp.float32),
            jax.ShapeDtypeStruct((b * h, n, dk, dv), jnp.float32),
            jax.ShapeDtypeStruct((b * h, dk, dv), jnp.float32),
        ],
        **_call_kwargs(num_warps, 1),
    )(*map(_flat, (q, k, g, beta, t, do, ds_fin)))
    return (
        du.reshape(b, h, n, c, dv),
        ds_after.reshape(b, h, n, dk, dv),
        ds0.reshape(b, h, dk, dv),
    )
