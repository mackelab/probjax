"""Pallas-Triton kernels for the inter-chunk pass of the gated delta rule.

See :mod:`probjax.nn.pallas_kernels.kernels.delta_rule` for the algorithm. The
chunk-local WY quantities are computed by XLA (batched, fully parallel); these
kernels run the only sequential part -- the recurrence over chunk-boundary
states -- with the ``[Dk, bv]`` state tile resident on chip for the whole
sequence. Columns of the state (the ``Dv`` axis) evolve independently, so the
grid is ``(B * H, Dv / bv)``.

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
    del chunk
    # Keep the fp32 state tile [Dk, bv] at <= 128 x 64.
    return min(dv, 64 if dk >= 128 else 128)


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


@functools.partial(jax.jit, static_argnames=("precision", "num_warps"))
def state_pass_fwd(wk, wv, kd, qg, p, a, s0, *, precision=None, num_warps=4):
    """Forward inter-chunk pass. Shapes ``[B, H, N, C, D]``; returns
    ``(o [B,H,N,C,Dv], states [B,H,N,Dk,Dv], final [B,H,Dk,Dv])`` in fp32."""
    b, h, n, c, dk = wk.shape
    dv = wv.shape[-1]
    bv = _block_v(dk, dv, c)
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


@functools.partial(jax.jit, static_argnames=("precision", "num_warps"))
def state_pass_bwd_seq(wk, kd, qg, p, a, do, ds_fin, *, precision=None, num_warps=4):
    """Reverse inter-chunk pass. Returns ``(du, ds_after, ds0)``."""
    b, h, n, c, dk = wk.shape
    dv = do.shape[-1]
    bv = _block_v(dk, dv, c)
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


def _prep_kernel(
    q_ref, k_ref, v_ref, g_ref, b_ref,
    wk_ref, wv_ref, kd_ref, qg_ref, p_ref, t_ref, *, chunk, precision,
):  # fmt: skip
    """One chunk: ``T = (I + A)^-1`` by 16-row block forward substitution, then
    ``Wk = T (beta e^g K)``, ``Wv = T (beta V)``, ``Kd``, ``Qg`` and ``P``."""
    nb = chunk // _SUB
    f32 = jnp.float32
    rows = jax.lax.broadcasted_iota(jnp.int32, (_SUB, _SUB), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (_SUB, _SUB), 1)
    eye = (rows == cols).astype(f32)
    g_last = _load(g_ref, (pl.ds(chunk - _SUB, _SUB),))
    g_last = jnp.min(g_last)  # g is non-increasing within a chunk

    def blk(ref, i):
        return _load(ref, (pl.ds(i * _SUB, _SUB), slice(None)))

    def vec(ref, i):
        return _load(ref, (pl.ds(i * _SUB, _SUB),))

    ks = [blk(k_ref, i) for i in range(nb)]
    gs = [vec(g_ref, i) for i in range(nb)]
    bs = [vec(b_ref, i) for i in range(nb)]

    def decay(i, j):
        d = gs[i][:, None] - gs[j][None, :]
        if i == j:
            d = jnp.where(rows >= cols, d, -jnp.inf)
        return jnp.exp(d)

    # A_ij = beta_i (k_i . k_j) exp(g_i - g_j), strictly lower-triangular.
    a_blk = {}
    for i in range(nb):
        for j in range(i + 1):
            kk = _dot(ks[i], ks[j].T, precision)
            a = bs[i][:, None] * kk * decay(i, j)
            if i == j:
                a = jnp.where(rows > cols, a, 0.0)
            a_blk[i, j] = a
    # Diagonal blocks: row-by-row forward substitution (row r only reads rows < r).
    t_blk = {}
    for i in range(nb):
        t = eye
        a = a_blk[i, i]
        for r in range(1, _SUB):
            a_r = jnp.where(rows == r, a, 0.0)
            t = t - _dot(a_r, t, precision)
        t_blk[i, i] = t
    # Off-diagonal blocks: T_ij = -T_ii sum_{j<=m<i} A_im T_mj.
    for i in range(nb):
        for j in range(i - 1, -1, -1):
            acc = _dot(a_blk[i, j], t_blk[j, j], precision)
            for m in range(j + 1, i):
                acc = acc + _dot(a_blk[i, m], t_blk[m, j], precision)
            t_blk[i, j] = -_dot(t_blk[i, i], acc, precision)
    zero = jnp.zeros((_SUB, _SUB), f32)
    for i in range(nb):
        for j in range(nb):
            _store(
                t_ref,
                (pl.ds(i * _SUB, _SUB), pl.ds(j * _SUB, _SUB)),
                t_blk[i, j] if j <= i else zero,
            )
    # W = T R with R_k = beta e^g K, R_v = beta V.
    rk = [ks[j].astype(f32) * (bs[j] * jnp.exp(gs[j]))[:, None] for j in range(nb)]
    rv = [blk(v_ref, j).astype(f32) * bs[j][:, None] for j in range(nb)]
    for i in range(nb):
        wk = _dot(t_blk[i, 0], rk[0], precision)
        wv = _dot(t_blk[i, 0], rv[0], precision)
        for j in range(1, i + 1):
            wk = wk + _dot(t_blk[i, j], rk[j], precision)
            wv = wv + _dot(t_blk[i, j], rv[j], precision)
        rows_i = (pl.ds(i * _SUB, _SUB), slice(None))
        _store(wk_ref, rows_i, wk)
        _store(wv_ref, rows_i, wv)
        _store(kd_ref, rows_i, ks[i].astype(f32) * jnp.exp(g_last - gs[i])[:, None])
        qi = blk(q_ref, i)
        _store(qg_ref, rows_i, qi.astype(f32) * jnp.exp(gs[i])[:, None])
        for j in range(nb):
            pij = _dot(qi, ks[j].T, precision) * decay(i, j) if j <= i else zero
            _store(p_ref, (pl.ds(i * _SUB, _SUB), pl.ds(j * _SUB, _SUB)), pij)


def prep_supported(chunk: int, dk: int, dv: int) -> bool:
    return chunk % _SUB == 0 and chunk <= 64 and supports_shape(chunk, dk, dv)


@functools.partial(
    jax.jit, static_argnames=("chunk", "precision", "num_warps", "num_stages")
)
def chunk_prep(q, k, v, g, beta, *, chunk, precision=None, num_warps=4, num_stages=1):
    """Pallas chunk prep. ``q, k: [B,H,L,Dk]``, ``v: [B,H,L,Dv]``, ``g`` (the
    within-chunk cumulative log gate) and ``beta: [B,H,L]`` (fp32).

    Returns fp32 ``wk, wv, kd, qg: [B,H,N,C,D]``, ``p, t: [B,H,N,C,C]``.
    """
    b, h, seq, dk = q.shape
    dv = v.shape[-1]
    n = seq // chunk
    prec = _kernel_precision(precision)
    r4 = lambda x: x.reshape(b * h, n, chunk, x.shape[-1])  # noqa: E731
    r3 = lambda x: x.reshape(b * h, n, chunk).astype(jnp.float32)  # noqa: E731
    mat = lambda d: pl.BlockSpec((None, None, chunk, d), lambda i, j: (i, j, 0, 0))  # noqa: E731
    row = pl.BlockSpec((None, None, chunk), lambda i, j: (i, j, 0))
    out_shape = [
        jax.ShapeDtypeStruct((b * h, n, chunk, d), jnp.float32)
        for d in (dk, dv, dk, dk, chunk, chunk)
    ]
    kernel = functools.partial(_prep_kernel, chunk=chunk, precision=prec)
    outs = pallas_call_compat(
        kernel,
        grid=(b * h, n),
        in_specs=[mat(dk), mat(dk), mat(dv), row, row],
        out_specs=[mat(d) for d in (dk, dv, dk, dk, chunk, chunk)],
        out_shape=out_shape,
        **_call_kwargs(num_warps, num_stages),
    )(r4(q), r4(k), r4(v), r3(g), r3(beta))
    return tuple(o.reshape(b, h, n, chunk, o.shape[-1]) for o in outs)
