from __future__ import annotations

import functools
from typing import Callable, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from probjax.utils.typing import Array

from ..kernel_utils import (
    dot,
    get_dot_precision,
    pad_to_multiple,
    use_interpret_mode,
)
from ..kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    make_kernel_primitive,
    shardable_kernel,
)

KernelFn = Callable[[Array, Array, Array], Array]
KernelVjpFn = Callable[[Array, Array, Array, Array], tuple[Array, Array, Array]]


def _broadcast_block_spec(shape: tuple[int, ...]) -> pl.BlockSpec:
    return pl.BlockSpec(shape, lambda *unused: (0,) * len(shape))


def _pad_axis_to_multiple(x: Array, axis: int, multiple: int) -> tuple[Array, int]:
    size = x.shape[axis]
    return pad_to_multiple(x, axis, multiple), size


def _validate_inputs(q: Array, k: Array, v: Array) -> None:
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, N, D], got {q.shape}.")
    if k.ndim != 3:
        raise ValueError(f"k must have shape [B, M, D], got {k.shape}.")
    if v.ndim != 3:
        raise ValueError(f"v must have shape [B, M, O], got {v.shape}.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}."
        )
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"Feature dims must match, got q={q.shape[2]}, k={k.shape[2]}."
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(
            f"Key/value lengths must match, got k={k.shape[1]}, v={v.shape[1]}."
        )


def _resolve_interpret_mode(interpret: bool) -> bool:
    return bool(interpret or use_interpret_mode())


def kernel_mv_naive(
    q: Array, k: Array, v: Array, params: Array, kernel_fn: KernelFn
) -> Array:
    """Naive dense kernel-matrix vector product.

    Args:
        q: Query array [B, N, D].
        k: Key array [B, M, D].
        v: Value array [B, M, O].
        params: Arbitrary-rank kernel parameters.
        kernel_fn: Callable returning kernel blocks [N, M] from ([N, D], [M, D], params).

    Returns:
        Output y = K(q, k) @ v with shape [B, N, O].
    """
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    params = jnp.asarray(params)
    _validate_inputs(q, k, v)

    k_mat = jax.vmap(lambda qb, kb: kernel_fn(qb, kb, params))(q, k)
    return jnp.einsum("bnm,bmo->bno", k_mat, v)


def _kernel_mv_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    params_ref,
    out_ref,
    *,
    kernel_fn: KernelFn,
    block_k: int,
):
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    q = q_ref[...]
    out = jnp.zeros_like(out_ref, dtype=jnp.float32)

    def body(start_k, carry):
        acc = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = k_ref[curr_k_slice, :]
        v = v_ref[curr_k_slice, :]
        w = kernel_fn(q, k, params_ref[...])
        acc = acc + dot(w.astype(v.dtype), v, precision=precision)
        return acc

    out = lax.fori_loop(0, pl.cdiv(k_ref.shape[0], block_k), body, out)
    out_ref[...] = out.astype(out_ref.dtype)


def _kernel_mv_dq_dparams_kernel(
    q_ref,
    k_ref,
    v_ref,
    dy_ref,
    params_ref,
    dq_ref,
    dparams_ref,
    *,
    kernel_vjp_fn: KernelVjpFn,
    block_k: int,
    param_size: int,
):
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    q = q_ref[...]
    dy = dy_ref[...]
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dparams = jnp.zeros((param_size,), dtype=jnp.float32)

    def body(start_k, carry):
        dq_acc, dparams_acc = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = k_ref[curr_k_slice, :]
        v = v_ref[curr_k_slice, :]

        coeff = dot(dy, v.T, precision=precision)
        dq_block, _, dparams_block = kernel_vjp_fn(q, k, params_ref[...], coeff)
        dq_acc = dq_acc + dq_block.astype(dq_acc.dtype)
        dparams_acc = dparams_acc + jnp.reshape(
            dparams_block.astype(dparams_acc.dtype), (param_size,)
        )
        return dq_acc, dparams_acc

    dq, dparams = lax.fori_loop(
        0, pl.cdiv(k_ref.shape[0], block_k), body, (dq, dparams)
    )
    dq_ref[...] = dq.astype(dq_ref.dtype)
    dparams_ref[...] = jnp.reshape(dparams.astype(dparams_ref.dtype), dparams_ref.shape)


def _kernel_mv_dk_kernel(
    q_ref,
    k_ref,
    v_ref,
    dy_ref,
    params_ref,
    dk_ref,
    *,
    kernel_vjp_fn: KernelVjpFn,
    block_q: int,
):
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    k = k_ref[...]
    v = v_ref[...]
    dk = jnp.zeros_like(k, dtype=jnp.float32)

    def body(start_q, carry):
        dk_acc = carry
        curr_q_slice = pl.dslice(start_q * block_q, block_q)
        q = q_ref[curr_q_slice, :]
        dy = dy_ref[curr_q_slice, :]

        coeff = dot(dy, v.T, precision=precision)
        _, dk_block, _ = kernel_vjp_fn(q, k, params_ref[...], coeff)
        dk_acc = dk_acc + dk_block.astype(dk_acc.dtype)
        return dk_acc

    dk = lax.fori_loop(0, pl.cdiv(q_ref.shape[0], block_q), body, dk)
    dk_ref[...] = dk.astype(dk_ref.dtype)


def _kernel_mv_pallas_impl(
    q: Array,
    k: Array,
    v: Array,
    params: Array,
    *,
    kernel_fn: KernelFn,
    block_q: int,
    block_k: int,
    block_o: int,
    interpret: bool,
) -> Array:
    _validate_inputs(q, k, v)

    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    params = jnp.asarray(params)

    dtype = jnp.result_type(q.dtype, k.dtype, v.dtype)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)

    bsz = q.shape[0]
    d = q.shape[2]

    q_pad, n_orig = _pad_axis_to_multiple(q, 1, block_q)
    k_pad, _ = _pad_axis_to_multiple(k, 1, block_k)
    v_pad, o_orig = _pad_axis_to_multiple(v, 2, block_o)
    v_pad, _ = _pad_axis_to_multiple(v_pad, 1, block_k)

    kernel = functools.partial(
        _kernel_mv_forward_kernel, kernel_fn=kernel_fn, block_k=block_k
    )
    out = pl.pallas_call(
        kernel,
        grid=(pl.cdiv(q_pad.shape[1], block_q), bsz, pl.cdiv(v_pad.shape[2], block_o)),
        in_specs=[
            pl.BlockSpec((None, block_q, d), lambda i, j, tile: (j, i, 0)),
            pl.BlockSpec((None, k_pad.shape[1], d), lambda i, j, tile: (j, 0, 0)),
            pl.BlockSpec(
                (None, v_pad.shape[1], block_o), lambda i, j, tile: (j, 0, tile)
            ),
            _broadcast_block_spec(params.shape),
        ],
        out_specs=pl.BlockSpec(
            (None, block_q, block_o), lambda i, j, tile: (j, i, tile)
        ),
        out_shape=jax.ShapeDtypeStruct((bsz, q_pad.shape[1], v_pad.shape[2]), dtype),
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        name="kernel_mv_forward",
    )(q_pad, k_pad, v_pad, params)
    return out[:, :n_orig, :o_orig]


@functools.lru_cache(maxsize=None)
def _kernel_mv_fwd_primitive(params_rank: int):
    """Forward primitive for kernel_mv; embarrassingly parallel over batch
    and query rows (each output row consumes its q row and all of k/v).

    Note: only the forward is sharding-aware. Gradients reduce over rows
    (dk/dv/dparams), so under a rows-sharded mesh the backward falls back to
    GSPMD's default handling (gather) rather than silently miscomputing.
    """
    spec = KernelSpec(
        name=f"kernel_mv_fwd_r{params_rank}",
        operands=(
            Operand("q", ("batch", "rows", "d")),
            Operand("k", ("batch", "cols", "d")),
            Operand("v", ("batch", "cols", "feat")),
            Operand("params", ("_",) * params_rank),
        ),
        outputs=(Output(("batch", "rows", "feat"), dtype_like=("q", "k", "v")),),
        shardable=frozenset({"batch", "rows"}),
    )

    def _impl(q, k, v, params, *, kernel_fn, block_q, block_k, block_o, interpret):
        return _kernel_mv_pallas_impl(
            q,
            k,
            v,
            params,
            kernel_fn=kernel_fn,
            block_q=block_q,
            block_k=block_k,
            block_o=block_o,
            interpret=interpret,
        )

    return make_kernel_primitive(spec, impl=_impl)


def _kernel_mv_forward(
    q, k, v, params, kernel_fn, block_q, block_k, block_o, interpret
):
    params = jnp.asarray(params)
    prim = _kernel_mv_fwd_primitive(params.ndim)
    return shardable_kernel(
        prim,
        {"q": q, "k": k, "v": v, "params": params},
        kernel_fn=kernel_fn,
        block_q=block_q,
        block_k=block_k,
        block_o=block_o,
        interpret=interpret,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _kernel_mv_pallas(
    q: Array,
    k: Array,
    v: Array,
    params: Array,
    kernel_fn: KernelFn,
    kernel_vjp_fn: KernelVjpFn,
    block_q: int,
    block_k: int,
    block_o: int,
    interpret: bool,
) -> Array:
    """Pallas tiled path of :func:`kernel_mv` (custom reverse-mode autodiff)."""
    return _kernel_mv_forward(
        q,
        k,
        v,
        params,
        kernel_fn,
        block_q,
        block_k,
        block_o,
        _resolve_interpret_mode(interpret),
    )


def _kernel_mv_fwd(
    q,
    k,
    v,
    params,
    kernel_fn,
    kernel_vjp_fn,
    block_q,
    block_k,
    block_o,
    interpret,
):
    del kernel_vjp_fn
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    params = jnp.asarray(params)
    use_interpret = _resolve_interpret_mode(interpret)
    out = _kernel_mv_forward(
        q, k, v, params, kernel_fn, block_q, block_k, block_o, use_interpret
    )
    return out, (q, k, v, params, use_interpret)


def _kernel_mv_bwd(
    kernel_fn,
    kernel_vjp_fn,
    block_q,
    block_k,
    block_o,
    interpret,
    res,
    dy,
):
    del interpret
    q, k, v, params, use_interpret = res
    dy = jnp.asarray(dy, dtype=jnp.result_type(q.dtype, k.dtype, v.dtype))

    bsz, d = q.shape[0], q.shape[2]
    param_shape = params.shape
    param_size = int(params.size)

    q_pad, n_orig = _pad_axis_to_multiple(q, 1, block_q)
    k_pad, m_orig = _pad_axis_to_multiple(k, 1, block_k)
    v_pad, _ = _pad_axis_to_multiple(v, 1, block_k)
    v_pad, o_orig = _pad_axis_to_multiple(v_pad, 2, block_o)
    dy_pad, _ = _pad_axis_to_multiple(dy, 1, block_q)
    dy_pad, _ = _pad_axis_to_multiple(dy_pad, 2, block_o)

    def _transpose_kernel_fn(kb, qb, p):
        return kernel_fn(qb, kb, p).T

    dv = _kernel_mv_pallas_impl(
        k,
        q,
        dy,
        params,
        kernel_fn=_transpose_kernel_fn,
        block_q=block_k,
        block_k=block_q,
        block_o=block_o,
        interpret=use_interpret,
    )

    dq_kernel = functools.partial(
        _kernel_mv_dq_dparams_kernel,
        kernel_vjp_fn=kernel_vjp_fn,
        block_k=block_k,
        param_size=param_size,
    )
    num_q_blocks = pl.cdiv(q_pad.shape[1], block_q)
    dq_pad, dparams_blocks = pl.pallas_call(
        dq_kernel,
        grid=(num_q_blocks, bsz),
        in_specs=[
            pl.BlockSpec((None, block_q, d), lambda i, j: (j, i, 0)),
            pl.BlockSpec((None, k_pad.shape[1], d), lambda i, j: (j, 0, 0)),
            pl.BlockSpec(
                (None, v_pad.shape[1], v_pad.shape[2]), lambda i, j: (j, 0, 0)
            ),
            pl.BlockSpec((None, block_q, v_pad.shape[2]), lambda i, j: (j, i, 0)),
            _broadcast_block_spec(params.shape),
        ],
        out_specs=[
            pl.BlockSpec((None, block_q, d), lambda i, j: (j, i, 0)),
            pl.BlockSpec((None, 1, param_size), lambda i, j: (j, i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, q_pad.shape[1], d), q.dtype),
            jax.ShapeDtypeStruct((bsz, num_q_blocks, param_size), jnp.float32),
        ],
        interpret=use_interpret,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        name="kernel_mv_dq_dparams",
    )(q_pad, k_pad, v_pad, dy_pad, params)

    dk_kernel = functools.partial(
        _kernel_mv_dk_kernel, kernel_vjp_fn=kernel_vjp_fn, block_q=block_q
    )
    dk_pad = pl.pallas_call(
        dk_kernel,
        grid=(pl.cdiv(k_pad.shape[1], block_k), bsz),
        in_specs=[
            pl.BlockSpec((None, q_pad.shape[1], d), lambda i, j: (j, 0, 0)),
            pl.BlockSpec((None, block_k, d), lambda i, j: (j, i, 0)),
            pl.BlockSpec((None, block_k, v_pad.shape[2]), lambda i, j: (j, i, 0)),
            pl.BlockSpec(
                (None, q_pad.shape[1], v_pad.shape[2]), lambda i, j: (j, 0, 0)
            ),
            _broadcast_block_spec(params.shape),
        ],
        out_specs=pl.BlockSpec((None, block_k, d), lambda i, j: (j, i, 0)),
        out_shape=jax.ShapeDtypeStruct((bsz, k_pad.shape[1], d), k.dtype),
        interpret=use_interpret,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        name="kernel_mv_dk",
    )(q_pad, k_pad, v_pad, dy_pad, params)

    dq = dq_pad[:, :n_orig, :]
    dk = dk_pad[:, :m_orig, :]
    dparams = (
        jnp.sum(dparams_blocks, axis=(0, 1)).reshape(param_shape).astype(params.dtype)
    )
    return dq, dk, dv, dparams


_kernel_mv_pallas.defvjp(_kernel_mv_fwd, _kernel_mv_bwd)


# ---------------------------------------------------------------------------
# Blocked XLA path (default on CPU).
#
# Queries are processed in row blocks so only a [B, block_rows, M] kernel tile is
# alive at a time; the backward recomputes the tiles instead of storing K, so the
# custom_vjp residuals are just (q, k, v, params): O((N + M) * D + M * O) memory.
# ---------------------------------------------------------------------------

#: Element budget of one materialized kernel tile ``[B, block_rows, M]``.
_XLA_TILE_ELEMENTS = 1 << 22


def _xla_block_rows(bsz: int, n: int, m: int, tile_elements: int | None) -> int:
    budget = _XLA_TILE_ELEMENTS if tile_elements is None else int(tile_elements)
    rows = max(budget // max(bsz * m, 1), 1)
    if rows >= n:
        return n
    rows = max(8, rows - rows % 8)
    num_blocks = -(-n // rows)
    # Balance the blocks so padding stays below 8 rows per block.
    rows = -(-n // num_blocks)
    return -(-rows // 8) * 8


def _split_rows(x: Array, block_rows: int) -> Array:
    """[B, N, F] -> [num_blocks, B, block_rows, F] (zero padded)."""
    bsz, n, f = x.shape
    x = pad_to_multiple(x, 1, block_rows)
    num_blocks = x.shape[1] // block_rows
    return jnp.swapaxes(x.reshape(bsz, num_blocks, block_rows, f), 0, 1)


def _merge_rows(x: Array, n: int) -> Array:
    """[num_blocks, B, block_rows, F] -> [B, N, F]."""
    num_blocks, bsz, block_rows, f = x.shape
    return jnp.swapaxes(x, 0, 1).reshape(bsz, num_blocks * block_rows, f)[:, :n]


def _batched_kernel(kernel_fn: KernelFn, q: Array, k: Array, params: Array) -> Array:
    return jax.vmap(lambda qb, kb: kernel_fn(qb, kb, params))(q, k)


def _kernel_mv_xla_forward(q, k, v, params, kernel_fn, block_rows):
    dtype = jnp.result_type(q.dtype, k.dtype, v.dtype)
    q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    n = q.shape[1]

    def block(q_blk):
        w = _batched_kernel(kernel_fn, q_blk, k, params).astype(dtype)
        out = jnp.einsum("bnm,bmo->bno", w, v, preferred_element_type=jnp.float32)
        return out.astype(dtype)

    if block_rows >= n:
        return block(q)
    return _merge_rows(lax.map(block, _split_rows(q, block_rows)), n)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _kernel_mv_xla(q, k, v, params, kernel_fn, kernel_vjp_fn, block_rows):
    del kernel_vjp_fn
    return _kernel_mv_xla_forward(q, k, v, params, kernel_fn, block_rows)


def _kernel_mv_xla_fwd(q, k, v, params, kernel_fn, kernel_vjp_fn, block_rows):
    del kernel_vjp_fn
    out = _kernel_mv_xla_forward(q, k, v, params, kernel_fn, block_rows)
    return out, (q, k, v, params)


def _kernel_mv_xla_bwd(kernel_fn, kernel_vjp_fn, block_rows, res, dy):
    q, k, v, params = res
    dtype = jnp.result_type(q.dtype, k.dtype, v.dtype)
    qc, kc, vc = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    dy = jnp.asarray(dy).astype(dtype)
    bsz, n, _ = q.shape
    m = k.shape[1]
    f32 = jnp.float32
    batched_vjp = jax.vmap(kernel_vjp_fn, in_axes=(0, 0, None, 0))

    def block(carry, xs):
        dk_acc, dv_acc, dp_acc = carry
        q_blk, dy_blk = xs
        w = _batched_kernel(kernel_fn, q_blk, kc, params).astype(dtype)
        dv_acc = dv_acc + jnp.einsum(
            "bnm,bno->bmo", w, dy_blk, preferred_element_type=f32
        )
        coeff = jnp.einsum("bno,bmo->bnm", dy_blk, vc, preferred_element_type=f32)
        dq_blk, dk_blk, dp_blk = batched_vjp(q_blk, kc, params, coeff.astype(dtype))
        dk_acc = dk_acc + dk_blk.astype(f32)
        dp_acc = dp_acc + jnp.sum(dp_blk.astype(f32), axis=0).reshape(dp_acc.shape)
        return (dk_acc, dv_acc, dp_acc), dq_blk.astype(q.dtype)

    init = (
        jnp.zeros((bsz, m, k.shape[2]), f32),
        jnp.zeros((bsz, m, v.shape[2]), f32),
        jnp.zeros(params.shape, f32),
    )
    if block_rows >= n:
        (dk, dv, dp), dq = block(init, (qc, dy))
    else:
        (dk, dv, dp), dq_blocks = lax.scan(
            block, init, (_split_rows(qc, block_rows), _split_rows(dy, block_rows))
        )
        dq = _merge_rows(dq_blocks, n)
    return (
        dq.astype(q.dtype),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        dp.astype(params.dtype),
    )


_kernel_mv_xla.defvjp(_kernel_mv_xla_fwd, _kernel_mv_xla_bwd)


_IMPLEMENTATIONS = ("xla", "pallas")


def _resolve_implementation(implementation: str | None, interpret: bool) -> str:
    if implementation is not None:
        if implementation not in _IMPLEMENTATIONS:
            raise ValueError(
                f"implementation must be one of {_IMPLEMENTATIONS} or None, "
                f"got {implementation!r}."
            )
        return implementation
    if interpret:
        return "pallas"
    return "xla" if jax.default_backend() == "cpu" else "pallas"


def kernel_mv(
    q: Array,
    k: Array,
    v: Array,
    params: Array,
    kernel_fn: KernelFn,
    kernel_vjp_fn: KernelVjpFn,
    block_q: int = 64,
    block_k: int = 64,
    block_o: int = 64,
    interpret: bool = False,
    *,
    implementation: str | None = None,
    xla_tile_elements: int | None = None,
) -> Array:
    """Kernel matrix-vector product ``y = K(q, k) @ v`` with custom autodiff.

    Never materializes the full ``[N, M]`` kernel matrix, in the forward or the
    backward pass: the backward recomputes kernel tiles, so the memory held for
    reverse mode is only ``(q, k, v, params)``.

    Args:
        q: Query array [B, N, D].
        k: Key array [B, M, D].
        v: Value array [B, M, O].
        params: Arbitrary-rank kernel parameters.
        kernel_fn: Callable kernel block function ``([n, D], [m, D], params) -> [n, m]``.
        kernel_vjp_fn: VJP of ``kernel_fn``:
            ``(q_blk, k_blk, params, ct[n, m]) -> (dq_blk, dk_blk, dparams)``.
        block_q: Query tile size (Pallas path, power of two).
        block_k: Key/value tile size (Pallas path, power of two).
        block_o: Output tile size (Pallas path); rounded to a power of two >= 8
            and capped at the padded output width.
        interpret: Force the Pallas path in interpret mode (debugging).
        implementation: ``"pallas"`` (tiled Pallas kernel), ``"xla"`` (row-blocked
            XLA: ``lax.map``/``lax.scan`` over query blocks with a dense
            ``[B, rows, M]`` tile inside) or ``None`` to choose automatically:
            ``"xla"`` on CPU, ``"pallas"`` elsewhere, ``"pallas"`` whenever
            ``interpret=True``.
        xla_tile_elements: Element budget of one kernel tile on the XLA path
            (default ``2**22``); smaller values lower peak memory.

    Note:
        On GPU the Pallas path zero-pads the feature dim ``D`` to a power of two
        >= 8 (a Triton requirement), so ``kernel_fn`` must be invariant to
        appended zero feature columns -- true for kernels of inner products or
        Euclidean distances such as the RBF. Use ``implementation="xla"`` for
        kernels where that does not hold (e.g. per-dimension parameters with a
        ``D`` that is not a power of two >= 8).

    Returns:
        Output y = K(q, k) @ v with shape [B, N, O].
    """
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    params = jnp.asarray(params)
    _validate_inputs(q, k, v)
    impl = _resolve_implementation(implementation, interpret)
    if impl == "xla":
        block_rows = _xla_block_rows(
            q.shape[0], q.shape[1], k.shape[1], xla_tile_elements
        )
        return _kernel_mv_xla(q, k, v, params, kernel_fn, kernel_vjp_fn, block_rows)
    use_interpret = _resolve_interpret_mode(interpret)
    if use_interpret:
        return _kernel_mv_pallas(
            q, k, v, params, kernel_fn, kernel_vjp_fn, block_q, block_k, block_o, True
        )
    # Triton needs power-of-two operand shapes and silently miscomputes dots whose
    # contraction dim is < 8 (D in {1, 2, 4} or block_o == 1 gave wrong results).
    # Zero-pad the feature and output dims outside the custom_vjp so autodiff
    # slices the gradients back.
    d, o = q.shape[2], v.shape[2]
    d_pad = _pallas_dim(d)
    block_o = min(_pallas_dim(block_o), _pallas_dim(o))
    if d_pad != d:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, d_pad - d)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, d_pad - d)))
    out = _kernel_mv_pallas(
        q, k, v, params, kernel_fn, kernel_vjp_fn, block_q, block_k, block_o, False
    )
    return out[..., :o]


_MIN_PALLAS_DIM = 8


def _pallas_dim(size: int) -> int:
    """Smallest power of two >= max(size, 8)."""
    return max(_MIN_PALLAS_DIM, 1 << (max(int(size), 1) - 1).bit_length())


def _rbf_kernel_fn(q_block: Array, k_block: Array, params: Array) -> Array:
    lengthscale = jnp.asarray(params)
    inv_l2 = 1.0 / (lengthscale * lengthscale)
    q_norm2 = jnp.sum(q_block * q_block, axis=-1, keepdims=True)
    k_norm2 = jnp.sum(k_block * k_block, axis=-1)[None, :]
    qk = q_block @ k_block.T
    r2 = jnp.maximum(q_norm2 + k_norm2 - 2.0 * qk, 0.0)
    return jnp.exp(-0.5 * inv_l2 * r2)


def _rbf_kernel_vjp_fn(
    q_block: Array,
    k_block: Array,
    params: Array,
    ct: Array,
) -> tuple[Array, Array, Array]:
    lengthscale = jnp.asarray(params)
    inv_l2 = 1.0 / (lengthscale * lengthscale)
    inv_l3 = inv_l2 / lengthscale

    q_norm2 = jnp.sum(q_block * q_block, axis=-1, keepdims=True)
    k_norm2 = jnp.sum(k_block * k_block, axis=-1)[None, :]
    qk = q_block @ k_block.T
    r2 = jnp.maximum(q_norm2 + k_norm2 - 2.0 * qk, 0.0)
    w = jnp.exp(-0.5 * inv_l2 * r2)
    coeff = ct * w

    row_sum = jnp.sum(coeff, axis=-1)
    col_sum = jnp.sum(coeff, axis=0)
    dq = inv_l2 * (coeff @ k_block - row_sum[:, None] * q_block)
    dk = inv_l2 * (coeff.T @ q_block - col_sum[:, None] * k_block)
    dlengthscale = inv_l3 * jnp.sum(coeff * r2)
    return dq, dk, jnp.asarray(dlengthscale, dtype=lengthscale.dtype)


def rbf_kernel_mv_naive(
    q: Array, k: Array, v: Array, lengthscale: Array | float
) -> Array:
    lengthscale = jnp.asarray(
        lengthscale, dtype=jnp.result_type(jnp.asarray(q).dtype, jnp.float32)
    )
    return kernel_mv_naive(q, k, v, lengthscale, _rbf_kernel_fn)


def rbf_kernel_mv(
    q: Array,
    k: Array,
    v: Array,
    lengthscale: Array | float,
    block_q: int = 64,
    block_k: int = 64,
    block_o: int = 64,
    interpret: bool = False,
    *,
    implementation: str | None = None,
) -> Array:
    """RBF kernel matrix-vector product; see :func:`kernel_mv` for the options."""
    lengthscale = jnp.asarray(
        lengthscale, dtype=jnp.result_type(jnp.asarray(q).dtype, jnp.float32)
    )
    return kernel_mv(
        q,
        k,
        v,
        lengthscale,
        _rbf_kernel_fn,
        _rbf_kernel_vjp_fn,
        block_q=block_q,
        block_k=block_k,
        block_o=block_o,
        interpret=interpret,
        implementation=implementation,
    )


def _prepare_kde_inputs(
    query: Array,
    data: Array,
    weights: Array | None,
) -> tuple[Array, Array, Array]:
    query = jnp.asarray(query)
    data = jnp.asarray(data)

    if query.ndim == 2:
        query = query[None, ...]
    if data.ndim == 2:
        data = data[None, ...]

    if query.ndim != 3 or data.ndim != 3:
        raise ValueError(
            f"Expected query/data with rank 2 or 3, got query={query.shape}, data={data.shape}."
        )
    if query.shape[0] != data.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got query={query.shape[0]}, data={data.shape[0]}."
        )
    if query.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"Feature dims must match, got query={query.shape[-1]}, data={data.shape[-1]}."
        )

    bsz, m = data.shape[0], data.shape[1]
    if weights is None:
        w = jnp.full((bsz, m), 1.0 / m, dtype=jnp.result_type(query.dtype, data.dtype))
    else:
        w = jnp.asarray(weights)
        if w.ndim == 1:
            w = jnp.broadcast_to(w[None, :], (bsz, w.shape[0]))
        if w.ndim != 2:
            raise ValueError(f"weights must have rank 1 or 2, got shape {w.shape}.")
        if w.shape != (bsz, m):
            raise ValueError(f"weights must have shape {(bsz, m)}, got {w.shape}.")
        w_sum = jnp.sum(w, axis=-1, keepdims=True)
        eps = jnp.finfo(jnp.result_type(w.dtype, jnp.float32)).tiny
        w = w / jnp.maximum(w_sum, eps)

    return query, data, w


def kde_density(
    query: Array,
    data: Array,
    params: Array,
    kernel_fn: KernelFn,
    kernel_vjp_fn: KernelVjpFn,
    *,
    weights: Array | None = None,
    normalizer: Array | float = 1.0,
    log_density: bool = False,
    block_q: int = 64,
    block_k: int = 64,
    interpret: bool = False,
    implementation: str | None = None,
) -> Array:
    """Kernel density estimate using `kernel_mv`.

    Computes density at `query` against `data` with optional sample weights.
    The kernel is provided by `kernel_fn`/`kernel_vjp_fn` and can be differentiated
    with respect to `query` and `params`.
    """
    query_b, data_b, w = _prepare_kde_inputs(query, data, weights)
    v = w[..., None]
    density = cast(
        Array,
        kernel_mv(
            query_b,
            data_b,
            v,
            params,
            kernel_fn,
            kernel_vjp_fn,
            block_q=block_q,
            block_k=block_k,
            block_o=1,
            interpret=interpret,
            implementation=implementation,
        ),
    )[..., 0]

    density = density * jnp.asarray(normalizer, dtype=density.dtype)
    if log_density:
        eps = jnp.finfo(density.dtype).tiny
        density = jnp.log(jnp.maximum(density, eps))

    return density if query.ndim == 3 else density[0]


def rbf_kde_density(
    query: Array,
    data: Array,
    lengthscale: Array | float,
    *,
    weights: Array | None = None,
    normalized: bool = True,
    log_density: bool = False,
    block_q: int = 64,
    block_k: int = 64,
    interpret: bool = False,
    implementation: str | None = None,
) -> Array:
    """Gaussian KDE using the RBF kernel backend."""
    query_b, _, _ = _prepare_kde_inputs(query, data, weights)
    d = query_b.shape[-1]
    ls = jnp.asarray(lengthscale, dtype=jnp.result_type(query_b.dtype, jnp.float32))
    if normalized:
        normalizer = 1.0 / ((jnp.sqrt(2.0 * jnp.pi) * ls) ** d)
    else:
        normalizer = jnp.asarray(1.0, dtype=ls.dtype)
    return kde_density(
        query,
        data,
        ls,
        _rbf_kernel_fn,
        _rbf_kernel_vjp_fn,
        weights=weights,
        normalizer=normalizer,
        log_density=log_density,
        block_q=block_q,
        block_k=block_k,
        interpret=interpret,
        implementation=implementation,
    )
