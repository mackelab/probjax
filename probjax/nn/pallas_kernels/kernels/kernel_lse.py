"""Memory-scalable log-domain kernel reductions.

For point clouds ``x`` [N, D] and ``y`` [M, D], log-weights ``log_w`` [M] and a
temperature ``epsilon`` this module computes, for every query row ``i``::

    lse_i = logsumexp_j( log_w_j - c(x_i, y_j) / epsilon )

and optionally the soft-assignment average of per-key values ``v`` [M, O]::

    out_i = sum_j softmax_j( log_w_j - c(x_i, y_j) / epsilon ) * v_j

without materializing the [N, M] cost matrix. Like flash attention the
forward streams key blocks with an online max-rescaled accumulator, so the
working set is O(block_q * block_k) and the residuals saved for the backward
are only the inputs plus ``lse``/``out`` (O((N + M) * (D + O))). The backward
(custom VJP) recomputes the logits block by block.

These reductions are the building block of log-domain Sinkhorn
(:mod:`probjax.nn.ot`), entropic barycentric maps, Gaussian KDE log-densities
and log-domain MMD estimates.

Backends:

* ``impl="pallas"`` - a Pallas-Triton kernel (default on GPU). The forward is one
  program per query block; the backward runs one pass over query blocks
  (``dx``, ``d epsilon``) and one over key blocks (``dy``, ``d log_w``, ``dv``).
* ``impl="xla"`` - the same blocked algorithm as plain XLA loops (default on
  CPU): ``lax.map`` over query blocks with an inner ``fori_loop`` over key
  blocks; a single backward sweep that accumulates the key-side cotangents.
* ``impl="dense"`` - the materializing reference (small problems / testing).
"""

from __future__ import annotations

import functools
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from probjax.utils.typing import Array

__all__ = [
    "kernel_logsumexp",
    "kernel_logsumexp_dense",
    "sqeuclidean_cost",
]

CostFn = Callable[[Array, Array], Array]
Cost = Literal["sqeuclidean"] | CostFn
Impl = Literal["pallas", "xla", "dense"]

# Finite "minus infinity" for the running max: keeps exp(m_old - m_new) finite
# even when every logit of a block is -inf (zero weights / padded keys).
_NEG_BIG = -1e30


# ---------------------------------------------------------------------------
# Block math shared by the Pallas kernels and the blocked XLA implementation.
# ---------------------------------------------------------------------------


def _dot(a: Array, b: Array, precision) -> Array:
    """a @ b with fp32 accumulation."""
    return lax.dot_general(
        a,
        b,
        (((1,), (0,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def _dot_nt(a: Array, b: Array, precision) -> Array:
    """a @ b.T with fp32 accumulation."""
    return lax.dot_general(
        a,
        b,
        (((1,), (1,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def _dot_tn(a: Array, b: Array, precision) -> Array:
    """a.T @ b with fp32 accumulation."""
    return lax.dot_general(
        a,
        b,
        (((0,), (0,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def sqeuclidean_cost(x: Array, y: Array, precision=None) -> Array:
    """Squared Euclidean cost block ``|x_i - y_j|^2`` (fp32) via one matmul."""
    xf = x.astype(jnp.float32)
    yf = y.astype(jnp.float32)
    xx = jnp.sum(xf * xf, axis=-1)[:, None]
    yy = jnp.sum(yf * yf, axis=-1)[None, :]
    return xx + yy - 2.0 * _dot_nt(x, y, precision)


def _cost_block(cost: Cost, x: Array, y: Array, precision) -> Array:
    if cost == "sqeuclidean":
        return sqeuclidean_cost(x, y, precision)
    return cost(x, y).astype(jnp.float32)


def _cost_block_vjp(
    cost: Cost, x: Array, y: Array, dc: Array, precision
) -> tuple[Array, Array]:
    """Cotangents (fp32) of x and y for a cost block with cotangent ``dc``."""
    if cost == "sqeuclidean":
        xf = x.astype(jnp.float32)
        yf = y.astype(jnp.float32)
        dc_in = dc.astype(x.dtype)
        row = jnp.sum(dc, axis=1)[:, None]
        col = jnp.sum(dc, axis=0)[:, None]
        dx = 2.0 * (row * xf - _dot(dc_in, y, precision))
        dy = 2.0 * (col * yf - _dot_tn(dc_in, x, precision))
        return dx, dy
    _, vjp = jax.vjp(lambda a, b: cost(a, b).astype(jnp.float32), x, y)
    dx, dy = vjp(dc)
    return dx.astype(jnp.float32), dy.astype(jnp.float32)


def _fwd_block_update(carry, s, v, precision):
    """Online (max-rescaled) logsumexp / softmax-average update for one block."""
    m, denom, acc = carry
    m_new = jnp.maximum(m, jnp.max(s, axis=1))
    alpha = jnp.exp(m - m_new)
    p = jnp.exp(s - m_new[:, None])
    denom = denom * alpha + jnp.sum(p, axis=1)
    if v is not None:
        acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v, precision)
    return m_new, denom, acc


def _bwd_block(s, c, lse, g, do, dl, v, inv_eps, precision):
    """Logit cotangents for one block.

    Returns ``(p, ds)`` where ``p = softmax`` probabilities and ``ds`` the
    cotangent of the logits ``s = log_w - c * inv_eps``.
    """
    p = jnp.exp(s - lse[:, None])
    ds = p * g[:, None]
    if v is not None:
        dp = _dot_nt(do.astype(v.dtype), v, precision)
        ds = ds + p * (dp - dl[:, None])
    return p, ds


# ---------------------------------------------------------------------------
# Dense reference.
# ---------------------------------------------------------------------------


def kernel_logsumexp_dense(
    x: Array,
    y: Array,
    *,
    values: Array | None = None,
    log_weights: Array | None = None,
    cost: Cost = "sqeuclidean",
    epsilon: Array | float = 1.0,
    precision=lax.Precision.HIGHEST,
):
    """Materializing reference of :func:`kernel_logsumexp` (O(N*M) memory)."""
    c = _cost_block(cost, x, y, precision)
    s = -c / jnp.asarray(epsilon, jnp.float32)
    if log_weights is not None:
        s = s + log_weights.astype(jnp.float32)[None, :]
    lse = jax.nn.logsumexp(s, axis=1)
    if values is None:
        return lse
    p = jnp.exp(s - lse[:, None])
    out = jnp.dot(p, values.astype(jnp.float32), precision=precision)
    return lse, out.astype(values.dtype)


# ---------------------------------------------------------------------------
# Padding helpers.
# ---------------------------------------------------------------------------


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _next_pow2(n: int, minimum: int = 16) -> int:
    p = minimum
    while p < n:
        p *= 2
    return p


def _pad_rows(a: Array, rows: int, value=0.0) -> Array:
    if a.shape[0] == rows:
        return a
    pad = [(0, rows - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
    return jnp.pad(a, pad, constant_values=value)


def _pad_cols(a: Array, cols: int) -> Array:
    if a.shape[1] == cols:
        return a
    return jnp.pad(a, [(0, 0), (0, cols - a.shape[1])])


# ---------------------------------------------------------------------------
# Pallas (Triton) implementation.
# ---------------------------------------------------------------------------


def _pl_fwd_kernel(
    x_ref,
    y_ref,
    lw_ref,
    *refs,
    cost,
    block_k,
    num_k_blocks,
    has_values,
    precision,
):
    if has_values:
        v_ref, ie_ref, lse_ref, o_ref = refs
    else:
        v_ref = o_ref = None
        ie_ref, lse_ref = refs
    x = x_ref[...]
    inv_eps = ie_ref[0]
    bq = x.shape[0]
    m = jnp.full((bq,), _NEG_BIG, jnp.float32)
    denom = jnp.zeros((bq,), jnp.float32)
    acc = (
        jnp.zeros(o_ref.shape, jnp.float32)
        if has_values
        else jnp.zeros((), jnp.float32)
    )

    def body(kb, carry):
        sl = pl.dslice(kb * block_k, block_k)
        y = y_ref[sl, :]
        lw = lw_ref[sl]
        v = v_ref[sl, :] if has_values else None
        s = lw[None, :] - _cost_block(cost, x, y, precision) * inv_eps
        return _fwd_block_update(carry, s, v, precision)

    m, denom, acc = lax.fori_loop(0, num_k_blocks, body, (m, denom, acc))
    lse_ref[...] = m + jnp.log(denom)
    if has_values:
        o_ref[...] = (acc / denom[:, None]).astype(o_ref.dtype)


def _pl_bwd_q_kernel(
    x_ref,
    y_ref,
    lw_ref,
    *refs,
    cost,
    block_k,
    num_k_blocks,
    has_values,
    precision,
):
    """Query-block pass: dx, per-row d(epsilon), log Z and the softmax term D.

    The logits recomputed here can differ from the forward's by an ulp of the
    cost divided by epsilon (Triton contracts FMAs differently per kernel), so
    ``Z_i = sum_j exp(s_ij - lse_i)`` is not exactly 1 and, with values,
    ``D_i = dO_i . O_i`` no longer matches ``sum_j p_ij dO_i . v_j``. At small
    epsilon these mismatches dominate the gradient error, so both are
    re-accumulated here and the gradient is formed self-consistently:
    ``dx = (A - D' B) / Z`` with ``A = sum_j p (g + dp) ds/dx``,
    ``B = sum_j p ds/dx`` and ``D' = sum_j p dp / Z``. The key pass then reuses
    ``lse + log Z`` and ``D'``.
    """
    if has_values:
        (v_ref, ie_ref, lse_ref, g_ref, do_ref, dx_ref, de_ref, lz_ref, dl_ref) = refs
    else:
        ie_ref, lse_ref, g_ref, dx_ref, de_ref, lz_ref = refs
        v_ref = do_ref = dl_ref = None
    x = x_ref[...]
    inv_eps = ie_ref[0]
    lse = lse_ref[...]
    g = g_ref[...]
    do = do_ref[...] if has_values else None
    bq = x.shape[0]
    zeros_x = jnp.zeros(x.shape, jnp.float32)
    zeros_r = jnp.zeros((bq,), jnp.float32)
    # carry: A, B, deA, deB, Z, sum p*dp   (B/deB/pd unused without values)
    init = (zeros_x, zeros_x, zeros_r, zeros_r, zeros_r, zeros_r)

    def body(kb, carry):
        a_acc, b_acc, de_a, de_b, z, pd = carry
        sl = pl.dslice(kb * block_k, block_k)
        y = y_ref[sl, :]
        lw = lw_ref[sl]
        c = _cost_block(cost, x, y, precision)
        s = lw[None, :] - c * inv_eps
        p = jnp.exp(s - lse[:, None])
        w = p * g[:, None]
        if has_values:
            v = v_ref[sl, :]
            dp = _dot_nt(do.astype(v.dtype), v, precision)
            pdp = p * dp
            w = w + pdp
            pd = pd + jnp.sum(pdp, axis=1)
            db, _ = _cost_block_vjp(cost, x, y, -p * inv_eps, precision)
            b_acc = b_acc + db
            de_b = de_b + jnp.sum(p * c, axis=1)
        da, _ = _cost_block_vjp(cost, x, y, -w * inv_eps, precision)
        return (
            a_acc + da,
            b_acc,
            de_a + jnp.sum(w * c, axis=1),
            de_b,
            z + jnp.sum(p, axis=1),
            pd,
        )

    a_acc, b_acc, de_a, de_b, z, pd = lax.fori_loop(0, num_k_blocks, body, init)
    inv_z = 1.0 / z
    if has_values:
        dl = pd * inv_z
        a_acc = a_acc - dl[:, None] * b_acc
        de_a = de_a - dl * de_b
        dl_ref[...] = dl
    dx_ref[...] = (a_acc * inv_z[:, None]).astype(dx_ref.dtype)
    de_ref[...] = de_a * inv_z * inv_eps * inv_eps
    lz_ref[...] = jnp.log(z)


def _pl_bwd_k_kernel(
    x_ref,
    y_ref,
    lw_ref,
    *refs,
    cost,
    block_q,
    num_q_blocks,
    has_values,
    precision,
):
    """Key-block pass: dy, d(log_w) and dv."""
    if has_values:
        (v_ref, ie_ref, lse_ref, g_ref, do_ref, dl_ref, dy_ref, dlw_ref, dv_ref) = refs
    else:
        ie_ref, lse_ref, g_ref, dy_ref, dlw_ref = refs
        v_ref = do_ref = dl_ref = dv_ref = None
    y = y_ref[...]
    lw = lw_ref[...]
    v = v_ref[...] if has_values else None
    inv_eps = ie_ref[0]
    dy = jnp.zeros(y.shape, jnp.float32)
    dlw = jnp.zeros(lw.shape, jnp.float32)
    dv = jnp.zeros(v.shape, jnp.float32) if has_values else jnp.zeros((), jnp.float32)

    def body(qb, carry):
        dy, dlw, dv = carry
        sl = pl.dslice(qb * block_q, block_q)
        x = x_ref[sl, :]
        lse = lse_ref[sl]
        g = g_ref[sl]
        do = do_ref[sl, :] if has_values else None
        dl = dl_ref[sl] if has_values else None
        c = _cost_block(cost, x, y, precision)
        s = lw[None, :] - c * inv_eps
        p, ds = _bwd_block(s, c, lse, g, do, dl, v, inv_eps, precision)
        _, dyb = _cost_block_vjp(cost, x, y, -ds * inv_eps, precision)
        if has_values:
            dv = dv + _dot_tn(p.astype(do.dtype), do, precision)
        return dy + dyb, dlw + jnp.sum(ds, axis=0), dv

    dy, dlw, dv = lax.fori_loop(0, num_q_blocks, body, (dy, dlw, dv))
    dy_ref[...] = dy.astype(dy_ref.dtype)
    dlw_ref[...] = dlw
    if has_values:
        dv_ref[...] = dv.astype(dv_ref.dtype)


def _full_spec(shape):
    return pl.BlockSpec(shape, lambda i: (0,) * len(shape))


def _row_spec(block, *rest):
    return pl.BlockSpec((block, *rest), lambda i: (i,) + (0,) * len(rest))


def _pallas_prepare(x, y, lw, v, cost, block_q, block_k):
    n, d = x.shape
    m = y.shape[0]
    if cost == "sqeuclidean":
        # Zero feature padding leaves the squared distance unchanged and gives
        # Triton the >= 16 contraction dim its dot requires.
        dp = _next_pow2(d)
        x = _pad_cols(x, dp)
        y = _pad_cols(y, dp)
    n_pad = _round_up(n, block_q)
    m_pad = _round_up(m, block_k)
    x = _pad_rows(x, n_pad)
    y = _pad_rows(y, m_pad)
    lw = _pad_rows(lw.astype(jnp.float32), m_pad, -jnp.inf)
    if v is not None:
        v = _pad_rows(_pad_cols(v, _next_pow2(v.shape[1])), m_pad)
    return x, y, lw, v


def _compiler_params(num_warps=4, num_stages=2):
    return plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages)


def _pallas_fwd(x, y, lw, v, inv_eps, *, cost, block_q, block_k, precision, interpret):
    n = x.shape[0]
    has_values = v is not None
    o = v.shape[1] if has_values else 0
    xp, yp, lwp, vp = _pallas_prepare(x, y, lw, v, cost, block_q, block_k)
    n_pad, dp = xp.shape
    m_pad = yp.shape[0]
    kernel = functools.partial(
        _pl_fwd_kernel,
        cost=cost,
        block_k=block_k,
        num_k_blocks=m_pad // block_k,
        has_values=has_values,
        precision=precision,
    )
    in_specs = [_row_spec(block_q, dp), _full_spec(yp.shape), _full_spec(lwp.shape)]
    args = [xp, yp, lwp]
    out_specs = [_row_spec(block_q)]
    out_shape = [jax.ShapeDtypeStruct((n_pad,), jnp.float32)]
    if has_values:
        in_specs.append(_full_spec(vp.shape))
        args.append(vp)
        out_specs.append(_row_spec(block_q, vp.shape[1]))
        out_shape.append(jax.ShapeDtypeStruct((n_pad, vp.shape[1]), v.dtype))
    in_specs.append(_full_spec((1,)))
    args.append(inv_eps)
    outs = pl.pallas_call(
        kernel,
        grid=(n_pad // block_q,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=_compiler_params(),
        name="kernel_lse_fwd",
    )(*args)
    lse = outs[0][:n]
    out = outs[1][:n, :o] if has_values else None
    return lse, out


def _pallas_bwd(
    x,
    y,
    lw,
    v,
    inv_eps,
    lse,
    g,
    do,
    dl,
    *,
    cost,
    block_q,
    block_k,
    precision,
    interpret,
):
    n, d = x.shape
    m = y.shape[0]
    has_values = v is not None
    xp, yp, lwp, vp = _pallas_prepare(x, y, lw, v, cost, block_q, block_k)
    n_pad, dp = xp.shape
    m_pad = yp.shape[0]
    # Padded query rows: lse=+inf makes p = 0 there, so they contribute nothing.
    lse_p = _pad_rows(lse, n_pad, jnp.inf)
    g_p = _pad_rows(g.astype(jnp.float32), n_pad)
    common = dict(cost=cost, has_values=has_values, precision=precision)

    # ---- pass over query blocks: dx, d eps, log Z, D ----
    in_specs = [_row_spec(block_q, dp), _full_spec(yp.shape), _full_spec(lwp.shape)]
    args = [xp, yp, lwp]
    if has_values:
        op = vp.shape[1]
        do_p = _pad_rows(_pad_cols(do, op), n_pad)
        in_specs.append(_full_spec(vp.shape))
        args.append(vp)
    in_specs += [_full_spec((1,)), _row_spec(block_q), _row_spec(block_q)]
    args += [inv_eps, lse_p, g_p]
    out_specs = [_row_spec(block_q, dp), _row_spec(block_q), _row_spec(block_q)]
    out_shape = [
        jax.ShapeDtypeStruct((n_pad, dp), jnp.float32),
        jax.ShapeDtypeStruct((n_pad,), jnp.float32),
        jax.ShapeDtypeStruct((n_pad,), jnp.float32),
    ]
    if has_values:
        in_specs.append(_row_spec(block_q, op))
        args.append(do_p)
        out_specs.append(_row_spec(block_q))
        out_shape.append(jax.ShapeDtypeStruct((n_pad,), jnp.float32))
    outs = pl.pallas_call(
        functools.partial(
            _pl_bwd_q_kernel, block_k=block_k, num_k_blocks=m_pad // block_k, **common
        ),
        grid=(n_pad // block_q,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=_compiler_params(),
        name="kernel_lse_bwd_q",
    )(*args)
    dx, de, log_z = outs[:3]
    if has_values:
        # Padded rows: do = 0 so D' = 0 there.
        dl_p = jnp.where(jnp.isfinite(outs[3]), outs[3], 0.0)

    # ---- pass over key blocks: dy, d log_w, dv ----
    in_specs = [_full_spec(xp.shape), _row_spec(block_k, dp), _row_spec(block_k)]
    args = [xp, yp, lwp]
    if has_values:
        in_specs.append(_row_spec(block_k, op))
        args.append(vp)
    in_specs += [_full_spec((1,)), _full_spec((n_pad,)), _full_spec((n_pad,))]
    # Padded rows have lse=+inf and log Z=-inf: keep them at +inf (p = 0).
    lse_k = jnp.where(jnp.isinf(lse_p), lse_p, lse_p + log_z)
    args += [inv_eps, lse_k, g_p]
    out_specs = [_row_spec(block_k, dp), _row_spec(block_k)]
    out_shape = [
        jax.ShapeDtypeStruct((m_pad, dp), jnp.float32),
        jax.ShapeDtypeStruct((m_pad,), jnp.float32),
    ]
    if has_values:
        in_specs += [_full_spec((n_pad, op)), _full_spec((n_pad,))]
        args += [do_p, dl_p]
        out_specs.append(_row_spec(block_k, op))
        out_shape.append(jax.ShapeDtypeStruct((m_pad, op), jnp.float32))
    outs = pl.pallas_call(
        functools.partial(
            _pl_bwd_k_kernel, block_q=block_q, num_q_blocks=n_pad // block_q, **common
        ),
        grid=(m_pad // block_k,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=_compiler_params(),
        name="kernel_lse_bwd_k",
    )(*args)
    dy, dlw = outs[0], outs[1]
    dv = outs[2][:m, : v.shape[1]] if has_values else None
    return dx[:n, :d], dy[:m, :d], dlw[:m], dv, jnp.sum(de[:n])


# ---------------------------------------------------------------------------
# Blocked XLA implementation (CPU default).
# ---------------------------------------------------------------------------


def _xla_fwd(x, y, lw, v, inv_eps, *, cost, block_q, block_k, precision):
    n = x.shape[0]
    m = y.shape[0]
    has_values = v is not None
    block_q = min(block_q, n)
    block_k = min(block_k, m)
    n_pad = _round_up(n, block_q)
    m_pad = _round_up(m, block_k)
    xp = _pad_rows(x, n_pad)
    yp = _pad_rows(y, m_pad)
    lwp = _pad_rows(lw.astype(jnp.float32), m_pad, -jnp.inf)
    vp = _pad_rows(v, m_pad) if has_values else None
    nkb = m_pad // block_k
    ie = inv_eps[0]

    def q_block(xb):
        bq = xb.shape[0]
        init = (
            jnp.full((bq,), _NEG_BIG, jnp.float32),
            jnp.zeros((bq,), jnp.float32),
            jnp.zeros((bq, v.shape[1]), jnp.float32) if has_values else None,
        )

        def body(kb, carry):
            start = kb * block_k
            yb = lax.dynamic_slice_in_dim(yp, start, block_k)
            lwb = lax.dynamic_slice_in_dim(lwp, start, block_k)
            vb = lax.dynamic_slice_in_dim(vp, start, block_k) if has_values else None
            s = lwb[None, :] - _cost_block(cost, xb, yb, precision) * ie
            return _fwd_block_update(carry, s, vb, precision)

        if nkb == 1:
            mm, ll, acc = body(0, init)
        else:
            mm, ll, acc = lax.fori_loop(0, nkb, body, init)
        lse = mm + jnp.log(ll)
        out = (acc / ll[:, None]).astype(v.dtype) if has_values else None
        return lse, out

    xs = xp.reshape(n_pad // block_q, block_q, x.shape[1])
    if n_pad == block_q:
        lse, out = q_block(xp)
    else:
        lse, out = lax.map(q_block, xs)
        lse = lse.reshape(n_pad)
        out = out.reshape(n_pad, v.shape[1]) if has_values else None
    return lse[:n], (out[:n] if has_values else None)


def _xla_bwd(
    x, y, lw, v, inv_eps, lse, g, do, dl, *, cost, block_q, block_k, precision
):
    n, d = x.shape
    m = y.shape[0]
    has_values = v is not None
    block_q = min(block_q, n)
    block_k = min(block_k, m)
    n_pad = _round_up(n, block_q)
    m_pad = _round_up(m, block_k)
    xp = _pad_rows(x, n_pad)
    yp = _pad_rows(y, m_pad)
    lwp = _pad_rows(lw.astype(jnp.float32), m_pad, -jnp.inf)
    vp = _pad_rows(v, m_pad) if has_values else None
    lse_p = _pad_rows(lse, n_pad, jnp.inf)
    g_p = _pad_rows(g.astype(jnp.float32), n_pad)
    do_p = _pad_rows(do, n_pad) if has_values else None
    dl_p = _pad_rows(dl, n_pad) if has_values else None
    nqb = n_pad // block_q
    nkb = m_pad // block_k
    ie = inv_eps[0]

    def blk(a, i, size):
        return lax.dynamic_slice_in_dim(a, i * size, size) if a is not None else None

    def q_step(carry, qb):
        dy, dlw, dv = carry
        xb = blk(xp, qb, block_q)
        lseb = blk(lse_p, qb, block_q)
        gb = blk(g_p, qb, block_q)
        dob = blk(do_p, qb, block_q)
        dlb = blk(dl_p, qb, block_q)

        def k_step(kb, inner):
            dx, de, dy, dlw, dv = inner
            yb = blk(yp, kb, block_k)
            lwb = blk(lwp, kb, block_k)
            vb = blk(vp, kb, block_k)
            c = _cost_block(cost, xb, yb, precision)
            s = lwb[None, :] - c * ie
            p, ds = _bwd_block(s, c, lseb, gb, dob, dlb, vb, ie, precision)
            dxb, dyb = _cost_block_vjp(cost, xb, yb, -ds * ie, precision)
            start = kb * block_k
            dy = lax.dynamic_update_slice_in_dim(
                dy, lax.dynamic_slice_in_dim(dy, start, block_k) + dyb, start, 0
            )
            dlw = lax.dynamic_update_slice_in_dim(
                dlw,
                lax.dynamic_slice_in_dim(dlw, start, block_k) + jnp.sum(ds, axis=0),
                start,
                0,
            )
            if has_values:
                dvb = _dot_tn(p.astype(dob.dtype), dob, precision)
                dv = lax.dynamic_update_slice_in_dim(
                    dv, lax.dynamic_slice_in_dim(dv, start, block_k) + dvb, start, 0
                )
            return dx + dxb, de + jnp.sum(ds * c), dy, dlw, dv

        init = (jnp.zeros((block_q, d), jnp.float32), jnp.zeros((), jnp.float32))
        if nkb == 1:
            dx, de, dy, dlw, dv = k_step(0, init + (dy, dlw, dv))
        else:
            dx, de, dy, dlw, dv = lax.fori_loop(0, nkb, k_step, init + (dy, dlw, dv))
        return (dy, dlw, dv), (dx, de)

    init = (
        jnp.zeros((m_pad, d), jnp.float32),
        jnp.zeros((m_pad,), jnp.float32),
        jnp.zeros((m_pad, v.shape[1]), jnp.float32) if has_values else None,
    )
    if nqb == 1:
        (dy, dlw, dv), (dx, de) = q_step(init, 0)
    else:
        (dy, dlw, dv), (dx, de) = lax.scan(q_step, init, jnp.arange(nqb))
        dx = dx.reshape(n_pad, d)
    de = jnp.sum(de) * ie * ie
    return dx[:n], dy[:m], dlw[:m], (dv[:m] if has_values else None), de


# ---------------------------------------------------------------------------
# custom_vjp core and public API.
# ---------------------------------------------------------------------------


def _run_fwd(x, y, lw, v, eps, cfg):
    cost, impl, block_q, block_k, precision, interpret = cfg
    inv_eps = (1.0 / jnp.asarray(eps, jnp.float32)).reshape(1)
    if impl == "pallas":
        return _pallas_fwd(
            x,
            y,
            lw,
            v,
            inv_eps,
            cost=cost,
            block_q=block_q,
            block_k=block_k,
            precision=precision,
            interpret=interpret,
        )
    return _xla_fwd(
        x,
        y,
        lw,
        v,
        inv_eps,
        cost=cost,
        block_q=block_q,
        block_k=block_k,
        precision=precision,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _klse(x, y, lw, v, eps, cfg):
    return _run_fwd(x, y, lw, v, eps, cfg)


def _klse_fwd(x, y, lw, v, eps, cfg):
    lse, out = _run_fwd(x, y, lw, v, eps, cfg)
    return (lse, out), (x, y, lw, v, eps, lse, out)


def _klse_bwd(cfg, res, cts):
    cost, impl, block_q, block_k, precision, interpret = cfg
    x, y, lw, v, eps, lse, out = res
    g, do = cts
    g = jnp.zeros_like(lse) if g is None else g
    has_values = v is not None
    if has_values:
        do = jnp.zeros_like(out) if do is None else do
        dl = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=1)
    else:
        dl = None
    inv_eps = (1.0 / jnp.asarray(eps, jnp.float32)).reshape(1)
    kw = dict(cost=cost, block_q=block_q, block_k=block_k, precision=precision)
    if impl == "pallas":
        dx, dy, dlw, dv, de = _pallas_bwd(
            x, y, lw, v, inv_eps, lse, g, do, dl, interpret=interpret, **kw
        )
    else:
        dx, dy, dlw, dv, de = _xla_bwd(x, y, lw, v, inv_eps, lse, g, do, dl, **kw)
    return (
        dx.astype(x.dtype),
        dy.astype(y.dtype),
        dlw.astype(lw.dtype),
        dv.astype(v.dtype) if has_values else None,
        jnp.reshape(de, jnp.shape(eps)).astype(jnp.result_type(eps)),
    )


_klse.defvjp(_klse_fwd, _klse_bwd)


# Automatic dispatch on GPU uses the dense path up to this many (query, key) pairs.
_GPU_DENSE_MAX_PAIRS = 2**21

_DEFAULT_BLOCKS = {
    "pallas": (64, 64),
    "xla": (2048, 2048),
}


def _default_precision(dtype, impl):
    # fp32 needs ~fp32-exact products: plain TF32 rounding of the x.y cross
    # term costs ~1e-3 * |x||y| / epsilon in the logits, which wrecks
    # small-epsilon OT. In the Triton kernel 3xTF32 is as accurate as IEEE fp32
    # here and 1.2-1.7x faster (RTX 3090, N=16k, D in {2, 8, 32}).
    if jnp.dtype(dtype) == jnp.float32:
        if impl == "pallas":
            return lax.DotAlgorithmPreset.TF32_TF32_F32_X3
        return lax.DotAlgorithmPreset.F32_F32_F32
    if jnp.dtype(dtype) == jnp.float64:
        return lax.Precision.HIGHEST
    return None


def kernel_logsumexp(
    x: Array,
    y: Array,
    *,
    values: Array | None = None,
    log_weights: Array | None = None,
    cost: Cost = "sqeuclidean",
    epsilon: Array | float = 1.0,
    impl: Impl | None = None,
    block_q: int | None = None,
    block_k: int | None = None,
    precision=None,
    interpret: bool = False,
):
    """Streaming ``logsumexp_j(log_w_j - c(x_i, y_j) / epsilon)`` over key blocks.

    Args:
        x: Query points ``[N, D]``.
        y: Key points ``[M, D]``.
        values: Optional per-key values ``[M, O]``; if given, also returns the
            softmax-weighted average ``sum_j softmax_ij v_j`` (e.g. the entropic
            barycentric map when ``values = y``).
        log_weights: Optional per-key log-weights ``[M]`` (``-inf`` allowed).
            ``None`` means all zeros (a plain sum, not a mean).
        cost: ``"sqeuclidean"`` (``|x - y|^2``, computed with one matmul) or a
            callable ``cost(x_block [bq, D], y_block [bk, D]) -> [bq, bk]``
            built from jnp ops (its gradient comes from ``jax.vjp`` per block).
            With ``impl="pallas"`` and ``D < 16`` a callable must not use
            matmuls (Triton requires contraction dims >= 16); the automatic
            dispatch therefore routes callables with ``D < 16`` to ``"xla"``.
        epsilon: Temperature (scalar, may be traced and differentiated).
        impl: ``"pallas"`` (GPU default), ``"xla"`` (CPU default) or ``"dense"``.
            Automatic dispatch on GPU uses ``"dense"`` for ``N * M <= 2**21``.
        block_q, block_k: Tile sizes (powers of two for ``"pallas"``).
        precision: Matmul precision for the cost cross term; defaults to exact
            fp32 products for fp32 inputs.
        interpret: Run the Pallas kernels in interpret mode (debugging).

    Returns:
        ``lse`` ``[N]`` (fp32), or ``(lse, out)`` with ``out`` ``[N, O]`` when
        ``values`` is given. Differentiable in ``x``, ``y``, ``values``,
        ``log_weights`` and ``epsilon``; the backward recomputes logits block-wise
        so memory stays O((N + M) * (D + O)).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"x and y must be 2D [N, D] / [M, D], got {x.shape}, {y.shape}"
        )
    if cost == "sqeuclidean" and x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dims differ: {x.shape[1]} vs {y.shape[1]}")
    m = y.shape[0]
    dtype = jnp.result_type(x.dtype, y.dtype)
    x = x.astype(dtype)
    y = y.astype(dtype)
    if log_weights is None:
        lw = jnp.zeros((m,), jnp.float32)
    else:
        lw = jnp.asarray(log_weights)
        if lw.shape != (m,):
            raise ValueError(f"log_weights must have shape ({m},), got {lw.shape}")
    if values is not None:
        values = jnp.asarray(values)
        if values.ndim != 2 or values.shape[0] != m:
            raise ValueError(f"values must have shape ({m}, O), got {values.shape}")
    if not (cost == "sqeuclidean" or callable(cost)):
        raise ValueError(f"Unknown cost {cost!r}")
    if impl is None:
        impl = "xla" if jax.default_backend() == "cpu" and not interpret else "pallas"
        if impl == "pallas" and x.shape[0] * m <= _GPU_DENSE_MAX_PAIRS:
            # Tiny problems: launch overhead dominates the 3 Pallas kernels; the
            # dense path is ~3x faster for fwd+bwd at N=M=1k (RTX 3090) and its
            # N*M buffer is only a few MB here.
            impl = "dense"
        if impl == "pallas" and callable(cost) and x.shape[1] < 16:
            # Triton dots need a contraction dim >= 16 and silently misbehave
            # below it; zero-padding features is only safe for sqeuclidean.
            impl = "xla"
    if precision is None:
        precision = _default_precision(dtype, impl)
    if impl == "dense":
        return kernel_logsumexp_dense(
            x,
            y,
            values=values,
            log_weights=lw,
            cost=cost,
            epsilon=epsilon,
            precision=lax.Precision.HIGHEST if precision is not None else None,
        )
    if impl not in ("pallas", "xla"):
        raise ValueError(f"Unknown impl {impl!r}")
    dq, dk = _DEFAULT_BLOCKS[impl]
    block_q = dq if block_q is None else int(block_q)
    block_k = dk if block_k is None else int(block_k)
    eps = jnp.asarray(epsilon)
    if not jnp.issubdtype(eps.dtype, jnp.floating):
        eps = eps.astype(jnp.float32)
    cfg = (cost, impl, block_q, block_k, precision, bool(interpret))
    lse, out = _klse(x, y, lw, values, eps, cfg)
    return lse if values is None else (lse, out)
