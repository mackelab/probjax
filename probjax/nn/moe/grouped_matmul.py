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

import jax
import jax.numpy as jnp

from probjax.nn.pallas_kernels.kernel_utils import use_interpret_mode
from probjax.nn.pallas_kernels.kernels.grouped_matmul import (
    gmm,
    is_hopper,
    mosaic_ragged_matmul,
    mosaic_transposed_matmul,
    tgmm,
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
    dx = gmm(block_group, dy.astype(x.dtype), jnp.swapaxes(w, 1, 2),
             block_m=block_m, interpret=interpret)
    dw = tgmm(block_start, block_end, x, dy, block_m=block_m, interpret=interpret)
    return dx, dw.astype(w.dtype), None, None, None


_gmm.defvjp(_gmm_fwd, _gmm_bwd)


def grouped_matmul(x: Array, w: Array, layout: GroupLayout, *, block_m: int = 64,
                   interpret: bool | None = None) -> Array:
    """``[P, K] @ w[e] -> [P, N]`` on the padded block-aligned buffer.

    ``x`` must already be in the layout's padded order with zero padding rows
    (see :func:`scatter_to_layout`).  Differentiable in ``x`` and ``w``.
    """
    if interpret is None:
        interpret = use_interpret_mode()
    if x.shape[0] != layout.n_padded:
        raise ValueError(f"x has {x.shape[0]} rows, layout expects {layout.n_padded}")
    return _gmm(x, w.astype(x.dtype), layout.block_expert, layout.block_start,
                layout.block_end, block_m, interpret)


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


def grouped_backend() -> str:
    """"mosaic" on Hopper, "triton" on other GPUs (and in interpret mode)."""
    return "mosaic" if is_hopper() else "triton"
