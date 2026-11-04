"""Chunked pure-JAX/XLA Mamba (S6) selective scan with a memory-bounded VJP.

The recurrence (per batch element, state ``s`` and channel ``d``)::

    h_t = exp(delta_t * a) * h_{t-1} + delta_t * b_t * x_t,    h_{-1} = 0
    y_t = sum_s c_t[s] * h_t[s] + d * x_t

is evaluated in chunks of ``chunk_size`` time steps: a ``lax.scan`` carries the
``[B, S, D]`` state across chunks. Within a chunk, ``inner="fused"`` (default,
best on CPU) runs one fused per-step update on the cache-resident state;
``inner="assoc"`` runs an associative scan over a ``[T, B, S, D]`` working
set (fewer sequential steps, more memory traffic). The per-step state
history ``[B, L, S, D]`` is never materialized for the whole sequence:

* forward: only the chunk-entry states ``[L / T, B, S, D]`` are kept as
  residuals (the ``lax.scan`` reference keeps all ``L`` states);
* backward (``custom_vjp``): a reverse ``lax.scan`` over chunks recomputes the
  in-chunk states from the saved entry state and back-propagates the state
  cotangent in reverse, so the backward working set is ``O(T * B * S * D)``.

This is the CPU path of :func:`compute_mamba_scan` and is a drop-in,
differentiable replacement for the sequential ``lax.scan`` reference.
"""

from __future__ import annotations

import functools

import jax
from jax import lax
from jax import numpy as jnp

_HI = lax.Precision.HIGHEST


def _combine(earlier, later):
    """Compose two affine maps ``h -> p*h + q`` (``earlier`` applied first)."""
    p_i, q_i = earlier
    p_j, q_j = later
    return p_j * p_i, p_j * q_i + q_j


def _linear_recurrence(p, q, h0):
    """All states of ``h_t = p_t * h_{t-1} + q_t`` along axis 0, given ``h0``."""
    p_cum, q_cum = lax.associative_scan(_combine, (p, q), axis=0)
    return p_cum * h0[None] + q_cum


def _reverse_linear_recurrence(p, q):
    """``g_t = p_t * g_{t+1} + q_t`` along axis 0 with ``g_T = 0``."""
    _, g = lax.associative_scan(_combine, (p, q), axis=0, reverse=True)
    return g


def _chunk_coeffs(a, xs, bs, ds):
    """Decay ``p = exp(delta*a)`` and input ``q = delta*b*x``: [T, B, S, D]."""
    p = jnp.exp(ds[:, :, None, :] * a[None, None])
    q = (ds * xs)[:, :, None, :] * bs[..., None]
    return p, q


def _to_chunks(arr, n_chunks: int, chunk: int):
    """[B, L, F] -> [N, T, B, F] (time-major chunks)."""
    bsz, _, feat = arr.shape
    return arr.reshape(bsz, n_chunks, chunk, feat).transpose(1, 2, 0, 3)


def _from_chunks(arr):
    """[N, T, B, F] -> [B, N*T, F]."""
    n, t, bsz, feat = arr.shape
    return arr.transpose(2, 0, 1, 3).reshape(bsz, n * t, feat)


def _step(h, a, x_t, b_t, delta_t):
    """One fused recurrence step on [B, S, D]; returns (h_t, p_t)."""
    p = jnp.exp(delta_t[:, None, :] * a[None])
    return p * h + (delta_t * x_t)[:, None, :] * b_t[:, :, None], p


def _fused_chunk_forward(h0, a, d, xc, bc, cc, dc):
    """Sequential per-step scan over one chunk: the whole step (decay, update,
    readout) is one fusion on a cache-resident [B, S, D] state."""

    def step(h, inp):
        x_t, b_t, c_t, delta_t = inp
        h, _ = _step(h, a, x_t, b_t, delta_t)
        y = jnp.sum(c_t[:, :, None] * h, axis=1) + x_t * d
        return h, y

    return lax.scan(step, h0, (xc, bc, cc, dc))


def _fused_chunk_backward(g_next, da_acc, h0, a, d, xc, bc, cc, dc, dyc):
    """Recompute the chunk's states, then a per-step reverse scan."""

    def fwd_step(h, inp):
        h, _ = _step(h, a, *inp)
        return h, h

    _, hs = lax.scan(fwd_step, h0, (xc, bc, dc))
    h_prev = jnp.concatenate([h0[None], hs[:-1]], axis=0)

    def bwd_step(carry, inp):
        g_next, da = carry
        x_t, b_t, c_t, delta_t, dy_t, h_t, hp_t = inp
        p = jnp.exp(delta_t[:, None, :] * a[None])
        g = c_t[:, :, None] * dy_t[:, None, :] + g_next  # dL/dh_t
        dlog = g * hp_t * p  # dL/d(delta_t * a)
        gb = jnp.sum(g * b_t[:, :, None], axis=1)
        ddelta = gb * x_t + jnp.sum(dlog * a[None], axis=1)
        dx = gb * delta_t + dy_t * d
        db = jnp.sum(g * (delta_t * x_t)[:, None, :], axis=2)
        dc_t = jnp.sum(dy_t[:, None, :] * h_t, axis=2)
        da = da + jnp.sum(dlog * delta_t[:, None, :], axis=0)
        return (p * g, da), (dx, db, dc_t, ddelta)

    (g0, da_acc), outs = lax.scan(
        bwd_step,
        (g_next, da_acc),
        (xc, bc, cc, dc, dyc, hs, h_prev),
        reverse=True,
    )
    return (g0, da_acc), outs


def _forward(x, a, b, c, delta, d, chunk, inner):
    """Returns (y [B, L, D] fp32, chunk-entry states [N, B, S, D])."""
    bsz, seq, dim = x.shape
    state = a.shape[0]
    n_chunks = seq // chunk
    xs, bs, cs, ds = (_to_chunks(v, n_chunks, chunk) for v in (x, b, c, delta))

    def body(h0, inp):
        xc, bc, cc, dc = inp
        if inner == "fused":
            h_last, y = _fused_chunk_forward(h0, a, d, xc, bc, cc, dc)
            return h_last, (y, h0)
        p, q = _chunk_coeffs(a, xc, bc, dc)
        hs = _linear_recurrence(p, q, h0)
        y = jnp.einsum("tbs,tbsd->tbd", cc, hs, precision=_HI) + xc * d
        return hs[-1], (y, h0)

    h_init = jnp.zeros((bsz, state, dim), jnp.float32)
    _, (ys, h_entry) = lax.scan(body, h_init, (xs, bs, cs, ds))
    return _from_chunks(ys), h_entry


def _backward(dy, x, a, b, c, delta, d, h_entry, chunk, inner):
    bsz, seq, dim = x.shape
    n_chunks = seq // chunk
    xs, bs, cs, ds, dys = (_to_chunks(v, n_chunks, chunk) for v in (x, b, c, delta, dy))

    def body(carry, inp):
        g_next, da_acc = carry  # g_next: cotangent of the chunk's last state
        xc, bc, cc, dc, dyc, h0 = inp
        if inner == "fused":
            return _fused_chunk_backward(g_next, da_acc, h0, a, d, xc, bc, cc, dc, dyc)
        p, q = _chunk_coeffs(a, xc, bc, dc)
        hs = _linear_recurrence(p, q, h0)
        # State cotangent: g_t = c_t dy_t + p_{t+1} g_{t+1}; the last step of
        # the chunk receives the cotangent flowing back from the next chunk.
        local = cc[..., None] * dyc[:, :, None, :]
        local = local.at[-1].add(g_next)
        p_next = jnp.concatenate([p[1:], jnp.zeros_like(p[:1])], axis=0)
        g = _reverse_linear_recurrence(p_next, local)
        h_prev = jnp.concatenate([h0[None], hs[:-1]], axis=0)
        dlog = g * h_prev * p  # cotangent of delta*a
        g_bx = jnp.einsum("tbsd,tbs->tbd", g, bc, precision=_HI)
        ddelta = g_bx * xc + jnp.einsum("tbsd,sd->tbd", dlog, a, precision=_HI)
        dx = g_bx * dc + dyc * d
        db = jnp.einsum("tbsd,tbd->tbs", g, dc * xc, precision=_HI)
        dc_out = jnp.einsum("tbd,tbsd->tbs", dyc, hs, precision=_HI)
        da_acc = da_acc + jnp.einsum("tbsd,tbd->sd", dlog, dc, precision=_HI)
        return (p[0] * g[0], da_acc), (dx, db, dc_out, ddelta)

    init = (
        jnp.zeros((bsz,) + a.shape, jnp.float32),
        jnp.zeros(a.shape, jnp.float32),
    )
    (_, da), (dx, db, dc, ddelta) = lax.scan(
        body, init, (xs, bs, cs, ds, dys, h_entry), reverse=True
    )
    dd = jnp.sum(dy * x, axis=(0, 1))[None]
    return (
        _from_chunks(dx),
        da,
        _from_chunks(db),
        _from_chunks(dc),
        _from_chunks(ddelta),
        dd,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _chunked_op(x, a, b, c, delta, d, chunk, inner):
    # Without differentiation no residuals are needed: the fused path runs
    # as a single flat scan (one chunk spanning the sequence).
    if inner == "fused":
        chunk = x.shape[1]
    return _forward(x, a, b, c, delta, d, chunk, inner)[0]


def _chunked_op_fwd(x, a, b, c, delta, d, chunk, inner):
    y, h_entry = _forward(x, a, b, c, delta, d, chunk, inner)
    return y, (x, a, b, c, delta, d, h_entry)


def _chunked_op_bwd(chunk, inner, residuals, dy):
    x, a, b, c, delta, d, h_entry = residuals
    return _backward(dy, x, a, b, c, delta, d, h_entry, chunk, inner)


_chunked_op.defvjp(_chunked_op_fwd, _chunked_op_bwd)


def mamba_scan_chunked(
    x: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    delta: jax.Array,
    d: jax.Array,
    *,
    chunk_size: int = 32,
    inner: str = "fused",
) -> jax.Array:
    """Chunked Mamba selective scan (same contract as ``compute_mamba_scan``).

    Args:
        x, delta: ``[B, L, D]``; a: ``[S, D]``; b, c: ``[B, L, S]``; d: ``[1, D]``.
        chunk_size: time steps per chunk; bounds the working set to
            ``chunk_size * B * S * D`` floats and the saved residual states to
            ``L / chunk_size * B * S * D``.
        inner: ``"fused"`` (fused per-step loop inside a chunk; fastest on
            CPU) or ``"assoc"`` (associative scan inside a chunk).

    Returns:
        ``y`` with shape ``[B, L, D]`` and dtype of ``x``. Computation is in
        float32; gradients are cast back to each input's dtype.
    """
    if inner not in ("fused", "assoc"):
        raise ValueError(f"inner must be 'fused' or 'assoc', got {inner!r}.")
    x32, a32, b32, c32, delta32, d32 = (
        v.astype(jnp.float32) for v in (x, a, b, c, delta, d)
    )
    seq = x.shape[1]
    chunk = max(1, min(int(chunk_size), seq))
    pad = (-seq) % chunk
    if pad:
        # delta = 0 makes padded steps the identity (p = 1, q = 0).
        widths = ((0, 0), (0, pad), (0, 0))
        x32, b32, c32, delta32 = (jnp.pad(v, widths) for v in (x32, b32, c32, delta32))
    y = _chunked_op(x32, a32, b32, c32, delta32, d32, chunk, inner)
    return y[:, :seq].astype(x.dtype)
