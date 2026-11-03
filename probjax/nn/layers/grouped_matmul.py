"""Grouped matmul for dropless MoE, on a Pallas (Triton) kernel for NVIDIA GPUs.

``jax.lax.ragged_dot`` is the natural primitive for "multiply each contiguous
group of rows by its own matrix", but on an A100 (jax 0.11) XLA does not lower
it to a grouped GEMM: an MoE layer's forward + backward at 65k tokens took
134 ms with 12 GiB of scratch -- slower than evaluating every expert densely
(74 ms) and 4.5x the capacity-dropping path (29 ms).  jax's own grouped-matmul
kernels are Mosaic-GPU (Hopper and newer) only.

Layout.  The routed rows arrive sorted by expert.  Each expert's group is
padded up to a multiple of the row block ``bm``, so every row block belongs to
exactly ONE expert; the kernel then reads that block's expert id and runs a
plain tiled matmul against that expert's weights.  Padding rows are zero, so
they produce zero output and add nothing to the weight gradient.  The padded
row count is static -- ``round_up(M + E * (bm - 1), bm)`` -- which bounds the
waste at E partial blocks rather than provisioning every expert for every
token.

Backward: ``dx`` is the same kernel against the transposed weights; ``dw[e]``
is ``x_e^T dy_e``, accumulated by a second kernel over expert e's row blocks.
The kernels live in :mod:`probjax.nn.pallas_kernels.kernels.grouped_matmul`
as sharding-aware primitives; this module owns the layout and the VJP.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from probjax.nn.pallas_kernels.kernel_utils import use_interpret_mode
from probjax.nn.pallas_kernels.kernels.grouped_matmul import (
    gather_gmm,
    gmm,
    gmm_scatter_add,
    is_hopper,
    mosaic_ragged_matmul,
    mosaic_transposed_matmul,
    tgmm,
    tgmm_gather,
)
from probjax.utils.typing import Array


@dataclass(frozen=True)
class GroupLayout:
    """Where each sorted row lands in the padded, block-aligned buffer."""

    padded_rows: Array  # [M] destination row of each sorted row
    block_expert: Array  # [n_blocks] expert owning each row block (clamped)
    block_start: Array  # [E] first row block of each expert
    block_end: Array  # [E] one past its last row block
    n_padded: int  # static padded row count


def make_group_layout(group_sizes: Array, n_rows: int, block_m: int) -> GroupLayout:
    """Block-aligned layout for ``n_rows`` rows sorted into ``len(group_sizes)`` groups."""
    num_groups = group_sizes.shape[0]
    group_sizes = group_sizes.astype(jnp.int32)
    padded = (group_sizes + block_m - 1) // block_m * block_m
    starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes)[:-1]])
    pad_starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)[:-1]])
    pad_ends = pad_starts + padded
    n_padded = -(-(n_rows + num_groups * (block_m - 1)) // block_m) * block_m

    row = jnp.arange(n_rows, dtype=jnp.int32)
    group_of_row = jnp.searchsorted(jnp.cumsum(group_sizes), row, side="right")
    group_of_row = jnp.minimum(group_of_row, num_groups - 1)
    padded_rows = pad_starts[group_of_row] + (row - starts[group_of_row])

    block_first_row = jnp.arange(n_padded // block_m, dtype=jnp.int32) * block_m
    block_expert = jnp.searchsorted(pad_ends, block_first_row, side="right")
    return GroupLayout(
        padded_rows=padded_rows,
        # Blocks past the last group hold only zero rows; any expert is fine.
        block_expert=jnp.minimum(block_expert, num_groups - 1).astype(jnp.int32),
        block_start=(pad_starts // block_m).astype(jnp.int32),
        block_end=(pad_ends // block_m).astype(jnp.int32),
        n_padded=int(n_padded),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _gmm(x, w, block_group, block_start, block_end, block_m, interpret):
    return gmm(block_group, x, w, block_m=block_m, interpret=interpret)


def _gmm_fwd(x, w, block_group, block_start, block_end, block_m, interpret):
    out = gmm(block_group, x, w, block_m=block_m, interpret=interpret)
    return out, (x, w, block_group, block_start, block_end)


def _gmm_bwd(block_m, interpret, res, dy):
    x, w, block_group, block_start, block_end = res
    dx = gmm(
        block_group,
        dy.astype(x.dtype),
        jnp.swapaxes(w, 1, 2),
        block_m=block_m,
        interpret=interpret,
    )
    dw = tgmm(block_start, block_end, x, dy, block_m=block_m, interpret=interpret)
    return dx, dw.astype(w.dtype), None, None, None


_gmm.defvjp(_gmm_fwd, _gmm_bwd)


def grouped_matmul(
    x: Array,
    w: Array,
    layout: GroupLayout,
    *,
    block_m: int = 64,
    interpret: bool | None = None,
) -> Array:
    """``[P, K] @ w[e] -> [P, N]`` on the padded block-aligned buffer.

    ``x`` must already be in the layout's padded order with zero padding rows
    (see :func:`scatter_to_layout`).  Differentiable in ``x`` and ``w``.
    """
    if interpret is None:
        interpret = use_interpret_mode()
    if x.shape[0] != layout.n_padded:
        raise ValueError(f"x has {x.shape[0]} rows, layout expects {layout.n_padded}")
    return _gmm(
        x,
        w.astype(x.dtype),
        layout.block_expert,
        layout.block_start,
        layout.block_end,
        block_m,
        interpret,
    )


def scatter_to_layout(x_sorted: Array, layout: GroupLayout) -> Array:
    """Sorted rows ``[M, D]`` -> padded buffer ``[P, D]`` (zeros elsewhere)."""
    out = jnp.zeros((layout.n_padded, x_sorted.shape[1]), x_sorted.dtype)
    return out.at[layout.padded_rows].set(x_sorted)


def gather_from_layout(y_padded: Array, layout: GroupLayout) -> Array:
    """Padded buffer ``[P, D]`` -> sorted rows ``[M, D]``."""
    return y_padded[layout.padded_rows]


# ---------------------------------------------------------------- Hopper


@jax.custom_vjp
def _ragged_mosaic(x, w, group_sizes):
    return mosaic_ragged_matmul(group_sizes, x, w)


def _ragged_mosaic_fwd(x, w, group_sizes):
    return mosaic_ragged_matmul(group_sizes, x, w), (x, w, group_sizes)


def _ragged_mosaic_bwd(res, dy):
    x, w, group_sizes = res
    dy = dy.astype(x.dtype)
    dx = mosaic_ragged_matmul(group_sizes, dy, w, transpose_rhs=True)
    dw = mosaic_transposed_matmul(group_sizes, x, dy)
    return dx, dw.astype(w.dtype), None


_ragged_mosaic.defvjp(_ragged_mosaic_fwd, _ragged_mosaic_bwd)


def ragged_matmul_hopper(x_sorted: Array, w: Array, group_sizes: Array) -> Array:
    """Sorted rows ``[M, K]`` times their group's ``w[g]`` on sm_90 (Mosaic GPU).

    No padded layout: jax's Mosaic kernels handle ragged groups themselves.
    Differentiable in ``x_sorted`` and ``w``.
    """
    return _ragged_mosaic(x_sorted, w, group_sizes)


def grouped_backend() -> Literal["mosaic", "triton"]:
    """ "mosaic" on Hopper, "triton" on other GPUs (and in interpret mode)."""
    return "mosaic" if is_hopper() else "triton"


# ---------------------------------------------------------- fused SwiGLU


def _swiglu_layout(group_sizes, n_rows, block_m):
    return make_group_layout(group_sizes, n_rows, block_m)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def grouped_swiglu(
    x: Array,
    w_gate: Array,
    w_up: Array,
    w_down: Array,
    group_sizes: Array,
    block_m: int = 64,
    interpret: bool | None = None,
) -> Array:
    """``(silu(x Wg_e) * (x Wu_e)) Wd_e`` for sorted rows ``x`` of group ``e``.

    One custom VJP for the whole expert block that saves only its INPUTS and
    recomputes gate/up in the backward.  With per-matmul VJPs, XLA kept every
    layer's ``[P, H]`` intermediates alive under the caller's remat: a stack of
    remat'd layers at 262k tokens grew 5.5 GiB per layer (23.8 GiB at 4 layers,
    against 15.8 for capacity routing), and the v8 decoder's update step asked
    for 33 GiB.  The backward recompute is tied to the incoming cotangent with
    an optimization barrier, so the scheduler cannot hoist it ahead of the
    backward pass and keep it alive.
    """
    y, _ = _grouped_swiglu_fwd(x, w_gate, w_up, w_down, group_sizes, block_m, interpret)
    return y


def _resolve_interpret(interpret):
    return use_interpret_mode() if interpret is None else interpret


# Row slabs of the padded layout: the expert block runs over slabs of about
# ``SLAB_ELEMS // H`` rows, one after another (a ``fori_loop``), so its
# ``[rows, H]`` hidden activations and their gradients exist for one slab at a
# time instead of for all ``N * top_k`` rows.  ``None``: one slab.
SLAB_ELEMS: int | None = 2**24


def _slabs(n_padded: int, block_m: int, hidden: int) -> tuple[int, int]:
    """(number of slabs, row blocks per slab) covering ``n_padded`` rows."""
    n_blocks = n_padded // block_m
    if SLAB_ELEMS is None:
        return 1, n_blocks
    target = max(1, SLAB_ELEMS // (hidden * block_m))  # blocks per slab
    n_slabs = -(-n_blocks // target)
    return n_slabs, -(-n_blocks // n_slabs)


def _slab_layout(layout: GroupLayout, n_slabs: int, rb: int, block_m: int, m: int):
    """Slab rows, the sorted row of every padded row (``m`` on padding, also
    past the layout's end) and the block experts, padded to whole slabs."""
    rows = rb * block_m
    total = n_slabs * rows
    src = (
        jnp
        .full((total,), m, jnp.int32)
        .at[layout.padded_rows]
        .set(jnp.arange(m, dtype=jnp.int32))
    )
    bexp = layout.block_expert
    bexp = jnp.pad(bexp, (0, n_slabs * rb - bexp.shape[0]), mode="edge")
    return rows, src, bexp


def _take_rows(a: Array, idx: Array) -> Array:
    """``a[idx]`` with zero rows where ``idx`` is out of range (padding)."""
    return a.at[idx].get(mode="fill", fill_value=0)


def _slab_fwd(xs, block_expert, w_gate, w_up, w_down, block_m, interpret):
    def mm(a, w):
        return gmm(block_expert, a, w, block_m=block_m, interpret=interpret)

    h = jax.nn.silu(mm(xs, w_gate)) * mm(xs, w_up)  # zero padding rows stay zero
    return mm(h, w_down)


def _slab_bwd(xs, dys, block_expert, bstart, bend, w_gate, w_up, w_down, bm, interp):
    """Gradients of one slab: ``(dx_slab, dw_gate, dw_up, dw_down)``."""

    def mm(a, w):
        return gmm(block_expert, a, w, block_m=bm, interpret=interp)

    def t(a, b):
        return tgmm(bstart, bend, a, b, block_m=bm, interpret=interp)

    dt = xs.dtype
    gate = mm(xs, w_gate).astype(jnp.float32)
    up = mm(xs, w_up).astype(jnp.float32)
    sig = jax.nn.sigmoid(gate)
    silu = gate * sig
    h = (silu * up).astype(dt)
    dw_down = t(h, dys)
    dh = mm(dys, jnp.swapaxes(w_down, 1, 2)).astype(jnp.float32)
    dgate = (dh * up * sig * (1.0 + gate * (1.0 - sig))).astype(dt)
    dup = (dh * silu).astype(dt)
    dw_gate = t(xs, dgate)
    dw_up = t(xs, dup)
    dxs = mm(dgate, jnp.swapaxes(w_gate, 1, 2)) + mm(dup, jnp.swapaxes(w_up, 1, 2))
    return dxs, dw_gate, dw_up, dw_down


def _grouped_swiglu_fwd(
    x: Array,
    w_gate: Array,
    w_up: Array,
    w_down: Array,
    group_sizes: Array,
    block_m: int,
    interpret: bool | None,
) -> tuple[Array, tuple[Array, ...]]:
    interpret = _resolve_interpret(interpret)
    layout = _swiglu_layout(group_sizes, x.shape[0], block_m)
    n_slabs, rb = _slabs(layout.n_padded, block_m, w_gate.shape[-1])
    ws = (w_gate, w_up, w_down, block_m, interpret)
    if n_slabs == 1:
        yp = _slab_fwd(scatter_to_layout(x, layout), layout.block_expert, *ws)
        return gather_from_layout(yp, layout), (x, w_gate, w_up, w_down, group_sizes)
    rows, src, bexp = _slab_layout(layout, n_slabs, rb, block_m, x.shape[0])

    def body(i, y):
        idx = jax.lax.dynamic_slice_in_dim(src, i * rows, rows)
        be = jax.lax.dynamic_slice_in_dim(bexp, i * rb, rb)
        ys = _slab_fwd(_take_rows(x, idx), be, *ws)
        return y.at[idx].set(ys, mode="drop")  # padding rows are dropped

    y = jax.lax.fori_loop(
        0, n_slabs, body, jnp.zeros((x.shape[0], w_down.shape[-1]), x.dtype)
    )
    return y, (x, w_gate, w_up, w_down, group_sizes)


def _grouped_swiglu_bwd(block_m, interpret, res, dy):
    interpret = _resolve_interpret(interpret)
    x, w_gate, w_up, w_down, group_sizes = res
    # Recompute only once the cotangent exists (see grouped_swiglu).
    x, dy = jax.lax.optimization_barrier((x, dy.astype(x.dtype)))
    layout = _swiglu_layout(group_sizes, x.shape[0], block_m)
    n_slabs, rb = _slabs(layout.n_padded, block_m, w_gate.shape[-1])
    ws = (w_gate, w_up, w_down, block_m, interpret)
    if n_slabs == 1:
        dxp, dw_gate, dw_up, dw_down = _slab_bwd(
            scatter_to_layout(x, layout),
            scatter_to_layout(dy, layout),
            layout.block_expert,
            layout.block_start,
            layout.block_end,
            *ws,
        )
        dx = gather_from_layout(dxp, layout)
    else:
        rows, src, bexp = _slab_layout(layout, n_slabs, rb, block_m, x.shape[0])

        def body(i, carry):
            dx, dwg, dwu, dwd = carry
            idx = jax.lax.dynamic_slice_in_dim(src, i * rows, rows)
            be = jax.lax.dynamic_slice_in_dim(bexp, i * rb, rb)
            # Each expert's blocks within this slab, in slab-local block ids.
            bs = jnp.clip(layout.block_start - i * rb, 0, rb)
            bn = jnp.clip(layout.block_end - i * rb, 0, rb)
            dxs, g, u, d = _slab_bwd(
                _take_rows(x, idx), _take_rows(dy, idx), be, bs, bn, *ws
            )
            dx = dx.at[idx].set(dxs, mode="drop")
            f32 = jnp.float32
            return dx, dwg + g.astype(f32), dwu + u.astype(f32), dwd + d.astype(f32)

        init = (
            jnp.zeros(x.shape, x.dtype),
            jnp.zeros(w_gate.shape, jnp.float32),
            jnp.zeros(w_up.shape, jnp.float32),
            jnp.zeros(w_down.shape, jnp.float32),
        )
        dx, dw_gate, dw_up, dw_down = jax.lax.fori_loop(0, n_slabs, body, init)
    return (
        dx,
        dw_gate.astype(w_gate.dtype),
        dw_up.astype(w_up.dtype),
        dw_down.astype(w_down.dtype),
        None,
    )


grouped_swiglu.defvjp(_grouped_swiglu_fwd, _grouped_swiglu_bwd)


# ------------------------------------------------- gather/scatter dispatch


@dataclass(frozen=True)
class _Routing:
    layout: GroupLayout
    order: Array  # [N*K] routed row (token * K + k) of each sorted row
    src: Array  # [P] token of each padded-layout row (0 on padding)
    valid: Array  # [P] False on padding rows
    gate: Array  # [P] float32 routing weight of each layout row (0 on padding)
    token_rows: Array  # [N, K] layout row of each (token, k) assignment


def _routing(expert_ids: Array, gates: Array, counts: Array, block_m: int) -> _Routing:
    n_tokens, top_k = expert_ids.shape
    m = n_tokens * top_k
    order = jnp.argsort(expert_ids.reshape(-1), stable=True).astype(jnp.int32)
    layout = make_group_layout(counts, m, block_m)
    p = layout.padded_rows
    src = jnp.zeros((layout.n_padded,), jnp.int32).at[p].set(order // top_k)
    valid = jnp.zeros((layout.n_padded,), bool).at[p].set(True)
    gate = (
        jnp
        .zeros((layout.n_padded,), jnp.float32)
        .at[p]
        .set(gates.reshape(-1).astype(jnp.float32)[order])
    )
    inverse = jnp.argsort(order).astype(jnp.int32)
    token_rows = p[inverse].reshape(n_tokens, top_k)
    return _Routing(layout, order, src, valid, gate, token_rows)


class _GatherOps:
    """Reference implementation: plain JAX gathers / scatter-adds around the
    padded-layout kernels.  Materialises ``[P, D]`` buffers, but only inside
    the custom VJP (nothing at model width is saved for the backward)."""

    def __init__(self, block_m, interpret):
        self.bm, self.interpret = block_m, interpret

    def gather_gmm(self, r, x, w):
        xp = jnp.where(r.valid[:, None], x[r.src], jnp.zeros((), x.dtype))
        return gmm(
            r.layout.block_expert, xp, w, block_m=self.bm, interpret=self.interpret
        )

    def gmm_scatter_add(self, r, scale, h, w, n_tokens):
        y = gmm(r.layout.block_expert, h, w, block_m=self.bm, interpret=self.interpret)
        y = y.astype(jnp.float32) * scale[:, None]
        return jnp.zeros((n_tokens, y.shape[1]), jnp.float32).at[r.src].add(y)

    def tgmm_gather_a(self, r, x, d):
        xp = jnp.where(r.valid[:, None], x[r.src], jnp.zeros((), x.dtype))
        return tgmm(
            r.layout.block_start,
            r.layout.block_end,
            xp,
            d,
            block_m=self.bm,
            interpret=self.interpret,
        )

    def tgmm_gather_b(self, r, h, dout, scale):
        d = jnp.where(
            r.valid[:, None], dout[r.src].astype(jnp.float32) * scale[:, None], 0.0
        )
        return tgmm(
            r.layout.block_start,
            r.layout.block_end,
            h,
            d.astype(h.dtype),
            block_m=self.bm,
            interpret=self.interpret,
        )


class _FusedOps(_GatherOps):
    """The same four operations with the gather / scatter inside the kernels:
    no ``[P, D]`` buffer is ever materialised."""

    def gather_gmm(self, r, x, w):
        return gather_gmm(
            r.layout.block_expert,
            r.src,
            r.valid,
            x,
            w,
            block_m=self.bm,
            interpret=self.interpret,
        )

    def gmm_scatter_add(self, r, scale, h, w, n_tokens):
        return gmm_scatter_add(
            r.layout.block_expert,
            r.src,
            r.valid,
            scale,
            h,
            w,
            n_tokens,
            block_m=self.bm,
            interpret=self.interpret,
        )

    def tgmm_gather_a(self, r, x, d):
        ones = jnp.ones_like(r.gate)
        return tgmm_gather(
            r.layout.block_start,
            r.layout.block_end,
            r.src,
            r.valid,
            ones,
            x,
            d,
            gather_a=True,
            gather_b=False,
            block_m=self.bm,
            interpret=self.interpret,
        )

    def tgmm_gather_b(self, r, h, dout, scale):
        return tgmm_gather(
            r.layout.block_start,
            r.layout.block_end,
            r.src,
            r.valid,
            scale,
            h,
            dout,
            gather_a=False,
            gather_b=True,
            block_m=self.bm,
            interpret=self.interpret,
        )


class _FusedGsumOps(_FusedOps):
    """Fused gathers, but a non-atomic combine: the down projection writes its
    rows in layout order ([P, D]) and each token sums its own top-k rows
    (an XLA gather-reduce).  Deterministic; one transient layout buffer."""

    def gmm_scatter_add(self, r, scale, h, w, n_tokens):
        y = gmm(r.layout.block_expert, h, w, block_m=self.bm, interpret=self.interpret)
        y = y.astype(jnp.float32) * scale[:, None]
        pos = r.token_rows  # [N, K] layout row of each (token, k)
        return jnp.sum(y[pos], axis=1)


def _ops(impl, block_m, interpret):
    cls = {"fused": _FusedOps, "fused_gsum": _FusedGsumOps}.get(impl, _GatherOps)
    return cls(block_m, _resolve_interpret(interpret))


def _dispatch_hidden(ops, r, x, w_gate, w_up):
    # One gather of x for both projections: [W_gate | W_up] along the hidden axis.
    gate_up = ops.gather_gmm(r, x, jnp.concatenate([w_gate, w_up], axis=2))
    return jnp.split(gate_up, 2, axis=1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def moe_swiglu_dispatch(
    x,
    gates,
    w_gate,
    w_up,
    w_down,
    expert_ids,
    counts,
    block_m=64,
    impl="fused",
    interpret=None,
):
    """Routed SwiGLU experts AND the combine, without top-k copies of ``x``.

    ``x`` [N, D] tokens, ``expert_ids`` / ``gates`` [N, K] (top-k choice and
    weight), ``counts`` [E] rows per expert.  Returns
    ``sum_k gates[n, k] * SwiGLU_{expert_ids[n, k]}(x[n])`` as [N, D].
    ``impl="fused"`` gathers / scatters inside the Pallas kernels;
    ``impl="fused_gsum"`` gathers inside them but combines without atomics
    (each token sums its top-k rows of the layout-order output: faster at
    large top-k, deterministic, one transient [N*K, D] float32 buffer);
    ``impl="gather"`` does the same math with JAX gathers (reference).
    Differentiable in ``x``, ``gates`` and the three weight stacks; saves only
    its inputs and recomputes once the cotangent exists.
    """
    y, _ = _moe_dispatch_fwd(
        x, gates, w_gate, w_up, w_down, expert_ids, counts, block_m, impl, interpret
    )
    return y


def _moe_dispatch_fwd(
    x, gates, w_gate, w_up, w_down, expert_ids, counts, block_m, impl, interpret
):
    ops = _ops(impl, block_m, interpret)
    r = _routing(expert_ids, gates, counts, block_m)
    gate, up = _dispatch_hidden(ops, r, x, w_gate, w_up)
    h = jax.nn.silu(gate) * up
    y = ops.gmm_scatter_add(r, r.gate, h, w_down, x.shape[0]).astype(x.dtype)
    return y, (x, gates, w_gate, w_up, w_down, expert_ids, counts)


def _moe_dispatch_bwd(block_m, impl, interpret, res, dy):
    ops = _ops(impl, block_m, interpret)
    x, gates, w_gate, w_up, w_down, expert_ids, counts = res
    x, dy = jax.lax.optimization_barrier((x, dy.astype(x.dtype)))
    r = _routing(expert_ids, gates, counts, block_m)
    gate, up = _dispatch_hidden(ops, r, x, w_gate, w_up)
    g32, u32 = gate.astype(jnp.float32), up.astype(jnp.float32)
    sig = jax.nn.sigmoid(g32)
    silu = g32 * sig
    h = (silu * u32).astype(x.dtype)
    # dy[src] @ Wd^T, unscaled: <y_p, dy[src_p]> = <h_p, (dy[src_p] Wd^T)> gives the
    # gate-weight gradient at hidden width; scaling by the gate gives dh.
    g_un = ops.gather_gmm(r, dy, jnp.swapaxes(w_down, 1, 2)).astype(jnp.float32)
    d_gate_row = jnp.sum(h.astype(jnp.float32) * g_un, axis=-1)
    dh = g_un * r.gate[:, None]
    dgate = (dh * u32 * sig * (1.0 + g32 * (1.0 - sig))).astype(x.dtype)
    dup = (dh * silu).astype(x.dtype)
    dw_down = ops.tgmm_gather_b(r, h, dy, r.gate)
    # Gate and up stacked again: one gather of x for both weight gradients and
    # one scatter-add for dx = dgate Wg^T + dup Wu^T.
    d_gate_up = jnp.concatenate([dgate, dup], axis=1)
    dw_gate, dw_up = jnp.split(ops.tgmm_gather_a(r, x, d_gate_up), 2, axis=2)
    w_gate_up_t = jnp.swapaxes(jnp.concatenate([w_gate, w_up], axis=2), 1, 2)
    ones = r.valid.astype(jnp.float32)
    dx = ops.gmm_scatter_add(r, ones, d_gate_up, w_gate_up_t, x.shape[0])
    n_tokens, top_k = expert_ids.shape
    d_gates = (
        jnp
        .zeros((n_tokens * top_k,), jnp.float32)
        .at[r.order]
        .set(d_gate_row[r.layout.padded_rows])
        .reshape(n_tokens, top_k)
    )
    return (
        dx.astype(x.dtype),
        d_gates.astype(gates.dtype),
        dw_gate.astype(w_gate.dtype),
        dw_up.astype(w_up.dtype),
        dw_down.astype(w_down.dtype),
        None,
        None,
    )


moe_swiglu_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)


# ------------------------------------------------ CPU: chunked XLA dispatch


def _chunk_table(group_sizes: Array, n_rows: int, chunk: int):
    """Static-count tiling of ``n_rows`` expert-sorted rows into row windows.

    Chunk ``i`` belongs to expert ``expert[i]`` and covers the sorted rows
    ``[lo[i], hi[i])`` (at most ``chunk`` of them, all of that expert); it
    reads the window ``[win[i], win[i] + chunk)``, clamped inside the array,
    and masks the rows outside ``[lo, hi)``.  ``sum_e ceil(c_e / chunk) <=
    ceil(n_rows / chunk) + E`` bounds the (static) number of chunks; the
    spare ones are empty (``lo == hi``).
    """
    num_groups = group_sizes.shape[0]
    gs = group_sizes.astype(jnp.int32)
    n_chunks = -(-n_rows // chunk) + num_groups
    ends = jnp.cumsum(gs)
    per = (gs + chunk - 1) // chunk
    cum = jnp.cumsum(per)
    i = jnp.arange(n_chunks, dtype=jnp.int32)
    expert = jnp.minimum(jnp.searchsorted(cum, i, side="right"), num_groups - 1)
    expert = expert.astype(jnp.int32)
    lo = ends[expert] - gs[expert] + (i - (cum[expert] - per[expert])) * chunk
    hi = jnp.where(i < cum[-1], jnp.minimum(lo + chunk, ends[expert]), lo)
    win = jnp.clip(lo, 0, n_rows - chunk)
    return expert, win, lo, hi, n_chunks


def default_chunk(n_rows: int, num_groups: int) -> int:
    """Rows per chunk of :func:`moe_swiglu_chunked`: about the mean group
    size, as a power of two in ``[128, 512]``.  Each group wastes up to one
    partial chunk, so chunks much larger than the groups do masked work, while
    small ones under-use the CPU matmuls (24-core CPU, 512 x 1024 experts:
    128 was best at 64 experts x 128 rows each, 512 at 8 x 2-4k rows)."""
    mean = max(1, n_rows // max(num_groups, 1))
    return int(min(512, max(128, 1 << (mean.bit_length() - 1))))


def _chunk_setup(expert_ids, gates, counts, chunk):
    n_tokens, top_k = expert_ids.shape
    m = n_tokens * top_k
    chunk = min(chunk, m)
    order = jnp.argsort(expert_ids.reshape(-1), stable=True).astype(jnp.int32)
    token = order // top_k
    gate = gates.reshape(-1).astype(jnp.float32)[order]
    table = _chunk_table(counts, m, chunk)
    return order, token, gate, table, chunk


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def moe_swiglu_chunked(x, gates, w_gate, w_up, w_down, expert_ids, counts, chunk=256):
    """:func:`moe_swiglu_dispatch` in plain XLA, for CPU.

    A loop over row chunks of the expert-sorted assignments: each chunk
    gathers its ``chunk`` tokens from ``x``, runs one expert's SwiGLU on them
    and scatter-adds the gate-weighted result into the ``[N, D]`` output.
    Neither the ``top_k`` copies of ``x`` nor any ``[N * K, H]`` hidden
    activation is ever materialised, and the weights are read in place --
    the ``"blocked"`` path instead gathers one weight copy per row block.
    The backward recomputes each chunk's hidden activations.  Computes in
    ``x.dtype`` with float32 accumulation; returns ``x.dtype``.
    """
    y, _ = _moe_chunked_fwd(x, gates, w_gate, w_up, w_down, expert_ids, counts, chunk)
    return y


def _moe_chunked_fwd(x, gates, w_gate, w_up, w_down, expert_ids, counts, chunk):
    _, token, gate, (expert, win, lo, hi, n_chunks), t = _chunk_setup(
        expert_ids, gates, counts, chunk
    )
    r = jnp.arange(t, dtype=jnp.int32)
    cdt = x.dtype

    def body(i, out):
        e, s = expert[i], win[i]
        rows = s + r
        tok = jax.lax.dynamic_slice_in_dim(token, s, t)
        g = jnp.where(
            (rows >= lo[i]) & (rows < hi[i]),
            jax.lax.dynamic_slice_in_dim(gate, s, t),
            0.0,
        )
        xc = x[tok]
        h = jax.nn.silu(_dot(xc, w_gate[e])) * _dot(xc, w_up[e])
        y = _dot(h.astype(cdt), w_down[e])
        return out.at[tok].add(y * g[:, None])

    out = jax.lax.fori_loop(
        0, n_chunks, body, jnp.zeros((x.shape[0], w_down.shape[-1]), jnp.float32)
    )
    return out.astype(cdt), (x, gates, w_gate, w_up, w_down, expert_ids, counts)


def _moe_chunked_bwd(chunk, res, dy):
    x, gates, w_gate, w_up, w_down, expert_ids, counts = res
    x, dy = jax.lax.optimization_barrier((x, dy))
    order, token, gate, (expert, win, lo, hi, n_chunks), t = _chunk_setup(
        expert_ids, gates, counts, chunk
    )
    r = jnp.arange(t, dtype=jnp.int32)
    cdt = x.dtype

    def body(i, carry):
        dx, dwg, dwu, dwd, dgate = carry
        e, s = expert[i], win[i]
        rows = s + r
        mask = (rows >= lo[i]) & (rows < hi[i])
        tok = jax.lax.dynamic_slice_in_dim(token, s, t)
        g = jnp.where(mask, jax.lax.dynamic_slice_in_dim(gate, s, t), 0.0)
        xc, dyc = x[tok], dy[tok]
        wg, wu, wd = w_gate[e], w_up[e], w_down[e]
        pre_g, u = _dot(xc, wg), _dot(xc, wu)
        sig = jax.nn.sigmoid(pre_g)
        silu = pre_g * sig
        h = silu * u
        g_un = _dot(dyc, wd.T)  # dy[tok] @ Wd^T
        d_row = jnp.sum(h * g_un, axis=-1)  # d gate of each row
        dh = g_un * g[:, None]
        dpre = (dh * u * sig * (1.0 + pre_g * (1.0 - sig))).astype(cdt)
        dup = (dh * silu).astype(cdt)
        hc = h.astype(cdt)
        dyg = (dyc.astype(jnp.float32) * g[:, None]).astype(cdt)
        dwd = dwd.at[e].add(_dot(hc.T, dyg))
        dwg = dwg.at[e].add(_dot(xc.T, dpre))
        dwu = dwu.at[e].add(_dot(xc.T, dup))
        dx = dx.at[tok].add(_dot(dpre, wg.T) + _dot(dup, wu.T))
        cur = jax.lax.dynamic_slice_in_dim(dgate, s, t)
        dgate = jax.lax.dynamic_update_slice_in_dim(
            dgate, jnp.where(mask, d_row, cur), s, 0
        )
        return dx, dwg, dwu, dwd, dgate

    f32 = jnp.float32
    init = (
        jnp.zeros(x.shape, f32),
        jnp.zeros(w_gate.shape, f32),
        jnp.zeros(w_up.shape, f32),
        jnp.zeros(w_down.shape, f32),
        jnp.zeros((token.shape[0],), f32),
    )
    dx, dwg, dwu, dwd, dgate = jax.lax.fori_loop(0, n_chunks, body, init)
    d_gates = jnp.zeros_like(dgate).at[order].set(dgate).reshape(gates.shape)
    return (
        dx.astype(x.dtype),
        d_gates.astype(gates.dtype),
        dwg.astype(w_gate.dtype),
        dwu.astype(w_up.dtype),
        dwd.astype(w_down.dtype),
        None,
        None,
    )


moe_swiglu_chunked.defvjp(_moe_chunked_fwd, _moe_chunked_bwd)
