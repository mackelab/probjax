"""Grouped matmul (GPU, Triton backend): each row block times its group's matrix.

Rows are laid out block-aligned by group (see
:mod:`probjax.nn.moe.grouped_matmul`), so every ``block_m`` rows belong to one
group and ``block_group[i]`` names it.  Two kernels:

* ``gmm``:  ``out[rows of block i] = x[rows of block i] @ w[block_group[i]]``
* ``tgmm``: ``dw[g] = sum over g's blocks of x_block^T @ dy_block``

Sharding (via :func:`shardable_kernel`): ``gmm`` is parallel over the output
columns ``n`` (tensor parallel: replicate ``x``, shard ``w`` and ``out`` on
``n``); ``tgmm`` over groups and both weight dims.  Rows are not shardable: a
row block's group comes from a global sort.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from ..kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    make_kernel_primitive,
    shardable_kernel,
)
from probjax.utils.typing import Array


_ALIGN = 16  # smallest Triton dot tile


def _tile(dim: int, pref: int) -> int:
    """Largest power of two <= pref dividing ``dim`` (a multiple of 16)."""
    t = 1 << (min(pref, dim).bit_length() - 1)
    while dim % t:
        t //= 2
    return t


def _pad_to(x: Array, axis: int, multiple: int = _ALIGN) -> Array:
    rem = -x.shape[axis] % multiple
    if not rem:
        return x
    pads = [(0, 0)] * x.ndim
    pads[axis] = (0, rem)
    return jnp.pad(x, pads)


def _gmm_kernel(bg_ref, x_ref, w_ref, o_ref, *, block_m, block_n, block_k):
    i, j = pl.program_id(0), pl.program_id(1)
    g = bg_ref[i]
    rows = pl.dslice(i * block_m, block_m)
    cols = pl.dslice(j * block_n, block_n)

    def body(kk, acc):
        ks = pl.dslice(kk * block_k, block_k)
        a = plgpu.load(x_ref.at[rows, ks])
        b = plgpu.load(w_ref.at[g, ks, cols])
        return acc + plgpu.dot(a, b)

    acc = lax.fori_loop(
        0, x_ref.shape[1] // block_k, body, jnp.zeros((block_m, block_n), jnp.float32)
    )
    plgpu.store(o_ref.at[rows, cols], acc.astype(o_ref.dtype))


def _tgmm_kernel(bs_ref, be_ref, x_ref, dy_ref, o_ref, *, block_m, block_k, block_n):
    g, ki, nj = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    ks = pl.dslice(ki * block_k, block_k)
    ns = pl.dslice(nj * block_n, block_n)

    def body(b, acc):
        rows = pl.dslice(b * block_m, block_m)
        a = plgpu.load(x_ref.at[rows, ks])
        d = plgpu.load(dy_ref.at[rows, ns])
        return acc + plgpu.dot(a.T, d)

    acc = lax.fori_loop(
        bs_ref[g], be_ref[g], body, jnp.zeros((block_k, block_n), jnp.float32)
    )
    plgpu.store(o_ref.at[g, ks, ns], acc.astype(o_ref.dtype))


def _gmm_impl(block_group, x, w, *, block_m, interpret):
    m, k = x.shape
    n = w.shape[-1]
    bn, bk = _tile(n, 128), _tile(k, 32)
    return pl.pallas_call(
        functools.partial(_gmm_kernel, block_m=block_m, block_n=bn, block_k=bk),
        grid=(m // block_m, n // bn),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name="grouped_matmul",
    )(block_group, x, w)


def _tgmm_impl(block_start, block_end, x, dy, *, block_m, interpret):
    k, n = x.shape[1], dy.shape[1]
    groups = block_start.shape[0]
    bk, bn = _tile(k, 64), _tile(n, 64)
    return pl.pallas_call(
        functools.partial(_tgmm_kernel, block_m=block_m, block_k=bk, block_n=bn),
        grid=(groups, k // bk, n // bn),
        out_shape=jax.ShapeDtypeStruct((groups, k, n), x.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name="grouped_matmul_tgmm",
    )(block_start, block_end, x, dy)


@functools.lru_cache(maxsize=None)
def _gmm_primitive():
    spec = KernelSpec(
        name="grouped_matmul",
        operands=(
            Operand("block_group", ("blocks",)),
            Operand("x", ("rows", "k")),
            Operand("w", ("groups", "k", "n")),
        ),
        outputs=(Output(("rows", "n"), dtype_like="x"),),
        shardable=frozenset({"n"}),
    )
    return make_kernel_primitive(spec, impl=_gmm_impl)


@functools.lru_cache(maxsize=None)
def _tgmm_primitive():
    spec = KernelSpec(
        name="grouped_matmul_tgmm",
        operands=(
            Operand("block_start", ("groups",)),
            Operand("block_end", ("groups",)),
            Operand("x", ("rows", "k")),
            Operand("dy", ("rows", "n")),
        ),
        outputs=(Output(("groups", "k", "n"), dtype_like="x"),),
        shardable=frozenset({"groups", "k", "n"}),
    )
    return make_kernel_primitive(spec, impl=_tgmm_impl)


def gmm(block_group: Array, x: Array, w: Array, *, block_m: int, interpret: bool) -> Array:
    """``[P, K] x [G, K, N] -> [P, N]``, row block ``i`` against ``w[block_group[i]]``.

    K and N are zero-padded to a multiple of 16 so every tile is a power of two.
    """
    n = w.shape[-1]
    out = shardable_kernel(
        _gmm_primitive(),
        {"block_group": block_group, "x": _pad_to(x, 1),
         "w": _pad_to(_pad_to(w.astype(x.dtype), 1), 2)},
        block_m=block_m,
        interpret=interpret,
    )
    return out[:, :n]


def tgmm(block_start: Array, block_end: Array, x: Array, dy: Array, *,
         block_m: int, interpret: bool) -> Array:
    """``dw[g] = x[g's blocks]^T @ dy[g's blocks]``: ``[G, K, N]``."""
    k, n = x.shape[1], dy.shape[1]
    out = shardable_kernel(
        _tgmm_primitive(),
        {"block_start": block_start, "block_end": block_end, "x": _pad_to(x, 1),
         "dy": _pad_to(dy.astype(x.dtype), 1)},
        block_m=block_m,
        interpret=interpret,
    )
    return out[:, :k, :n]


# --------------------------------------------------------------------------
# Hopper (sm_90): jax's Mosaic-GPU ragged-dot kernels (wgmma + TMA).  They
# take the sorted, UNPADDED rows and handle ragged groups themselves; the
# Triton kernels above run everywhere else (A100 included -- Mosaic GPU needs
# Hopper).  Blackwell has its own kernel in jax and is not wired here.
# --------------------------------------------------------------------------

_MOSAIC_BLOCK = dict(block_m=128, block_n=128, block_k=64,
                     max_concurrent_steps=2, grid_block_n=1)
_MOSAIC_ALIGN = 128


def is_hopper() -> bool:
    """Whether the default device is an sm_90 (H100/H200) GPU."""
    if jax.default_backend() != "gpu":
        return False
    cc = str(getattr(jax.devices()[0], "compute_capability", ""))
    return cc.split(".")[0] == "9"


def _mosaic_ragged_impl(group_sizes, x, w, *, transpose_rhs):
    from jax.experimental.pallas.ops.gpu.ragged_dot_mgpu import ragged_dot

    return ragged_dot(x, w, group_sizes=group_sizes, transpose_rhs=transpose_rhs,
                      **_MOSAIC_BLOCK)


def _mosaic_transposed_impl(group_sizes, x, dy):
    from jax.experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu import (
        transposed_ragged_dot,
    )

    return transposed_ragged_dot(x, dy, group_sizes=group_sizes, **_MOSAIC_BLOCK)


@functools.lru_cache(maxsize=None)
def _mosaic_ragged_primitive(transpose_rhs: bool):
    w_dims = ("groups", "n", "k") if transpose_rhs else ("groups", "k", "n")
    spec = KernelSpec(
        name=f"ragged_matmul_mosaic{'_t' if transpose_rhs else ''}",
        operands=(
            Operand("group_sizes", ("groups",)),
            Operand("x", ("rows", "k")),
            Operand("w", w_dims),
        ),
        outputs=(Output(("rows", "n"), dtype_like="x"),),
        shardable=frozenset({"n"}),
    )
    return make_kernel_primitive(spec, impl=_mosaic_ragged_impl)


@functools.lru_cache(maxsize=None)
def _mosaic_transposed_primitive():
    spec = KernelSpec(
        name="ragged_matmul_mosaic_tgmm",
        operands=(
            Operand("group_sizes", ("groups",)),
            Operand("x", ("rows", "k")),
            Operand("dy", ("rows", "n")),
        ),
        outputs=(Output(("groups", "k", "n"), dtype_like="x"),),
        shardable=frozenset({"groups", "k", "n"}),
    )
    return make_kernel_primitive(spec, impl=_mosaic_transposed_impl)


def mosaic_ragged_matmul(group_sizes: Array, x: Array, w: Array, *,
                         transpose_rhs: bool = False) -> Array:
    """Sorted rows ``[M, K]`` times ``w[g]`` (``[G, K, N]``, or ``[G, N, K]``
    with ``transpose_rhs``) -> ``[M, N]``.  Hopper only."""
    k_axis, n_axis = (2, 1) if transpose_rhs else (1, 2)
    n = w.shape[n_axis]
    w = _pad_to(_pad_to(w.astype(x.dtype), k_axis, _MOSAIC_ALIGN), n_axis, _MOSAIC_ALIGN)
    out = shardable_kernel(
        _mosaic_ragged_primitive(transpose_rhs),
        {"group_sizes": group_sizes.astype(jnp.int32),
         "x": _pad_to(x, 1, _MOSAIC_ALIGN), "w": w},
        transpose_rhs=transpose_rhs,
    )
    return out[:, :n]


def mosaic_transposed_matmul(group_sizes: Array, x: Array, dy: Array) -> Array:
    """``dw[g] = x_g^T dy_g`` over sorted rows: ``[G, K, N]``.  Hopper only."""
    k, n = x.shape[1], dy.shape[1]
    out = shardable_kernel(
        _mosaic_transposed_primitive(),
        {"group_sizes": group_sizes.astype(jnp.int32),
         "x": _pad_to(x, 1, _MOSAIC_ALIGN),
         "dy": _pad_to(dy.astype(x.dtype), 1, _MOSAIC_ALIGN)},
    )
    return out[:, :k, :n]
