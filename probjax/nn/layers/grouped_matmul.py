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


def _gmm_l(
    xp: Array, w: Array, layout: GroupLayout, block_m: int, interpret: bool
) -> Array:
    return gmm(layout.block_expert, xp, w, block_m=block_m, interpret=interpret)


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
    xp = scatter_to_layout(x, layout)
    gate = _gmm_l(xp, w_gate, layout, block_m, interpret)
    up = _gmm_l(xp, w_up, layout, block_m, interpret)
    h = jax.nn.silu(gate) * up  # zero padding rows stay zero
    y = gather_from_layout(_gmm_l(h, w_down, layout, block_m, interpret), layout)
    return y, (x, w_gate, w_up, w_down, group_sizes)


def _grouped_swiglu_bwd(block_m, interpret, res, dy):
    interpret = _resolve_interpret(interpret)
    x, w_gate, w_up, w_down, group_sizes = res
    # Recompute only once the cotangent exists (see grouped_swiglu).
    x, dy = jax.lax.optimization_barrier((x, dy.astype(x.dtype)))
    layout = _swiglu_layout(group_sizes, x.shape[0], block_m)
    xp = scatter_to_layout(x, layout)
    gate = _gmm_l(xp, w_gate, layout, block_m, interpret)
    up = _gmm_l(xp, w_up, layout, block_m, interpret)
    sig = jax.nn.sigmoid(gate.astype(jnp.float32))
    silu = gate.astype(jnp.float32) * sig
    h = (silu * up.astype(jnp.float32)).astype(x.dtype)

    dyp = scatter_to_layout(dy, layout)
    t = functools.partial(
        tgmm, layout.block_start, layout.block_end, block_m=block_m, interpret=interpret
    )
    dw_down = t(h, dyp)
    dh = _gmm_l(dyp, jnp.swapaxes(w_down, 1, 2), layout, block_m, interpret)
    dh = dh.astype(jnp.float32)
    dgate = (
        dh
        * up.astype(jnp.float32)
        * sig
        * (1.0 + gate.astype(jnp.float32) * (1.0 - sig))
    ).astype(x.dtype)
    dup = (dh * silu).astype(x.dtype)
    dw_gate = t(xp, dgate)
    dw_up = t(xp, dup)
    dxp = _gmm_l(
        dgate, jnp.swapaxes(w_gate, 1, 2), layout, block_m, interpret
    ) + _gmm_l(dup, jnp.swapaxes(w_up, 1, 2), layout, block_m, interpret)
    dx = gather_from_layout(dxp, layout)
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
    gate = jnp.zeros((layout.n_padded,), jnp.float32).at[p].set(
        gates.reshape(-1).astype(jnp.float32)[order]
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
        return gmm(r.layout.block_expert, xp, w, block_m=self.bm, interpret=self.interpret)

    def gmm_scatter_add(self, r, scale, h, w, n_tokens):
        y = gmm(r.layout.block_expert, h, w, block_m=self.bm, interpret=self.interpret)
        y = y.astype(jnp.float32) * scale[:, None]
        return jnp.zeros((n_tokens, y.shape[1]), jnp.float32).at[r.src].add(y)

    def tgmm_gather_a(self, r, x, d):
        xp = jnp.where(r.valid[:, None], x[r.src], jnp.zeros((), x.dtype))
        return tgmm(r.layout.block_start, r.layout.block_end, xp, d, block_m=self.bm, interpret=self.interpret)

    def tgmm_gather_b(self, r, h, dout, scale):
        d = jnp.where(r.valid[:, None], dout[r.src].astype(jnp.float32) * scale[:, None], 0.0)
        return tgmm(r.layout.block_start, r.layout.block_end, h, d.astype(h.dtype), block_m=self.bm, interpret=self.interpret)


class _FusedOps(_GatherOps):
    """The same four operations with the gather / scatter inside the kernels:
    no ``[P, D]`` buffer is ever materialised."""

    def gather_gmm(self, r, x, w):
        return gather_gmm(r.layout.block_expert, r.src, r.valid, x, w, block_m=self.bm, interpret=self.interpret)

    def gmm_scatter_add(self, r, scale, h, w, n_tokens):
        return gmm_scatter_add(r.layout.block_expert, r.src, r.valid, scale, h, w, n_tokens, block_m=self.bm, interpret=self.interpret)

    def tgmm_gather_a(self, r, x, d):
        ones = jnp.ones_like(r.gate)
        return tgmm_gather(r.layout.block_start, r.layout.block_end, r.src, r.valid, ones, x, d,
                           gather_a=True, gather_b=False, block_m=self.bm, interpret=self.interpret)

    def tgmm_gather_b(self, r, h, dout, scale):
        return tgmm_gather(r.layout.block_start, r.layout.block_end, r.src, r.valid, scale, h, dout,
                           gather_a=False, gather_b=True, block_m=self.bm, interpret=self.interpret)


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
    # One gathered GEMM against [W_gate | W_up]: x is read once, not twice.
    hidden = w_gate.shape[-1]
    gu = ops.gather_gmm(r, x, jnp.concatenate([w_gate, w_up], axis=-1))
    return gu[:, :hidden], gu[:, hidden:]


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def moe_swiglu_dispatch(x, gates, w_gate, w_up, w_down, expert_ids, counts,
                        block_m=64, impl="fused", interpret=None):
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
    y, _ = _moe_dispatch_fwd(x, gates, w_gate, w_up, w_down, expert_ids, counts, block_m, impl, interpret)
    return y


def _moe_dispatch_fwd(x, gates, w_gate, w_up, w_down, expert_ids, counts, block_m, impl, interpret):
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
    # gate and up halves in one kernel each: [dgate | dup] against
    # [W_gate | W_up] (K = 2 * hidden) for dx, and x^T [dgate | dup] for the
    # two weight gradients.
    hidden = w_gate.shape[-1]
    d_gu = jnp.concatenate([dgate, dup], axis=-1)
    dw_gu = ops.tgmm_gather_a(r, x, d_gu)
    dw_gate, dw_up = dw_gu[..., :hidden], dw_gu[..., hidden:]
    ones = r.valid.astype(jnp.float32)
    w_gu_t = jnp.swapaxes(jnp.concatenate([w_gate, w_up], axis=-1), 1, 2)
    dx = ops.gmm_scatter_add(r, ones, d_gu, w_gu_t, x.shape[0])
    n_tokens, top_k = expert_ids.shape
    d_gates = (jnp.zeros((n_tokens * top_k,), jnp.float32)
               .at[r.order].set(d_gate_row[r.layout.padded_rows])
               .reshape(n_tokens, top_k))
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
