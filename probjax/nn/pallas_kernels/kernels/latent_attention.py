"""Pallas kernel for multi-head latent attention (shared KV cache).

The kernel is the flex/MHA forward with one change: the key and value tiles are
formed inside the KV loop from a single shared latent cache instead of being read
from separate ``k``/``v`` tensors.  Each KV block loads the latent ``C`` once and
applies the per-head up-projections ``W_K``/``W_V``::

    K_h = C @ W_K[h]^T,   V_h = C @ W_V[h]^T

so the cache holds ``kv_lora_rank`` numbers per token rather than
``heads * (qk_head_dim + v_head_dim)``.  Passing ``wk=None`` / ``wv=None`` gives
the absorbed form ``K = V = C`` (the query head dimension is then the latent
width), where the value projection is applied to the attention output instead.

Mask and additive-bias handling match :func:`probjax.nn.pallas_kernels.mha`
(the same :class:`~probjax.nn.pallas_kernels.AttentionMask` /
:class:`~probjax.nn.pallas_kernels.AttentionBias` objects and the same dense-bias
path).  No token selection or block-sparse decoding is performed.

Backward reuses the tested :func:`mha` kernels: the custom VJP differentiates
the mathematically identical materialized computation ``mha(q, C@W_K, C@W_V)``,
which is exact because the kernel's forward is that computation with the
up-projection fused into the KV loop.  That backward materializes the per-head
``K``/``V`` and runs a second attention forward, so **only inference (forward
only) gets the cache saving**; training pays the materialized cost.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from ..kernel_utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    dot,
    get_dot_precision,
)
from ..kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    make_kernel_primitive,
    shardable_kernel,
)
from .attention import (
    BlockSizes,
    _dropout_mask_counter,
    _extract_fwd_data,
    _flatten_optional_pytree,
    _load_mask_data,
    _rng_seed_from_key,
    _unflatten_optional_pytree,
    mha,
    pallas_load,
    pallas_store,
)


def _pad_to_multiple(x: jax.Array, axis: int, multiple: int) -> jax.Array:
    """Zero-pad ``x`` along ``axis`` up to a multiple of ``multiple``."""
    rem = -x.shape[axis] % multiple
    if rem == 0:
        return x
    pads = [(0, 0)] * x.ndim
    pads[axis] = (0, rem)
    return jnp.pad(x, pads)


def _mla_forward_kernel(
    q_ref,
    c_ref,
    wk_ref,
    wv_ref,
    rng_ref,
    b_ref,
    id_q_ref,
    id_k_ref,
    dropout_mask_ref,
    o_ref,
    *,
    sm_scale: float,
    qk_head_dim: int,
    v_head_dim: int,
    mask_fn=None,
    bias_fn=None,
    dropout_rate: float = 0.0,
    block_q: int,
    block_k: int,
    block_dk: int,
    block_dv: int,
):
    """Flash-style forward over a shared latent cache.

    Mirrors :func:`probjax.nn.pallas_kernels.kernels.attention.mha_forward_kernel`
    but forms the key/value tiles from ``C`` (and the per-head ``W_K``/``W_V``)
    inside the KV loop.  ``C``, ``W_K`` and ``W_V`` are zero-padded to their tile
    widths by the caller; the padded columns are annihilated by the head masks
    and the padded latent rows never occur (the KV sequence must tile the loop).

    The positional argument order (``rng`` before ``b``/ids/dropout) matches the
    ``in_specs``/call order in :func:`_mla_build`.
    """
    seq_len = c_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    m_i = jnp.full((block_q,), NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    d_mask_k = jnp.arange(block_dk)[None] < qk_head_dim
    d_mask_v = jnp.arange(block_dv)[None] < v_head_dim
    o = jnp.zeros((block_q, block_dv), dtype=jnp.float32)

    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask_k, other=0.0)
    id_q = _load_mask_data(id_q_ref, curr_q_slice)
    span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634

    # Per-head up-projections, resident in SRAM for the whole KV loop.
    wk_h = (
        pallas_load(wk_ref, (slice(None), slice(None))) if wk_ref is not None else None
    )
    wv_h = (
        pallas_load(wv_ref, (slice(None), slice(None))) if wv_ref is not None else None
    )

    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        c = pallas_load(c_ref, (curr_k_slice, slice(None)))  # [block_k, block_r]
        # Form this block's keys (and values) from the shared latent.
        k = c if wk_h is None else dot(c, wk_h.T, precision=precision)
        qk = dot(q, k.T, precision=precision)
        if sm_scale != 1.0:
            qk *= sm_scale

        span_k = start_k * block_k + jnp.arange(block_k)
        if bias_fn is not None:
            if b_ref is not None:
                b_chunk = pallas_load(b_ref, (slice(None), curr_k_slice))
            else:
                b_chunk = None
            qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
        if mask_fn is not None:
            if id_k_ref is not None:
                id_k = _load_mask_data(id_k_ref, curr_k_slice)
            elif id_q is not None:
                id_k = _load_mask_data(id_q_ref, curr_k_slice)
            else:
                id_k = None
            mask = mask_fn(span_q, span_k, id_q, id_k)
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev

        v = c if wv_h is None else dot(c, wv_h.T, precision=precision)
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            s_curr = jnp.where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = dot(s_curr.astype(v.dtype), v, precision=precision)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    o, _m_i, l_i = lax.fori_loop(0, pl.cdiv(seq_len, block_k), body, (o, m_i, l_i))

    l_i = jnp.where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]
    pallas_store(
        o_ref, (slice(None), slice(None)), val=o.astype(o_ref.dtype), mask=d_mask_v
    )


# ---------------------------------------------------------------------------
# Kernel spec / primitive
# ---------------------------------------------------------------------------

_Q_DIMS = ("batch", "q_seq", "heads", "qk_head_dim")
_C_DIMS = ("batch", "kv_seq", "kv_lora_rank")
_O_DIMS = ("batch", "q_seq", "heads", "v_head_dim")
_O_DIMS_ABSORBED = ("batch", "q_seq", "heads", "kv_lora_rank")

# The output's ``v_head_dim`` factor is only defined when ``wv`` is present, and
# factor resolution skips optional operands.  Materialized and absorbed modes
# therefore get separate specs (each with no optional projection operands)
# rather than one spec with optional ones.
_MLA_FWD_OPTIONALS = (
    Operand("b_data", ("batch", "heads", "_", "_"), optional=True),
    Operand("q_id", ("batch", "_", "_", "_"), optional=True),
    Operand("k_id", ("batch", "_", "_", "_"), optional=True),
    Operand("dropout_mask", ("batch", "heads", "_", "_"), optional=True),
)

_MLA_FWD_SPEC = KernelSpec(
    name="mla_fwd",
    operands=(
        Operand("q", _Q_DIMS),
        Operand("cache", _C_DIMS),
        Operand("wk", ("heads", "qk_head_dim", "kv_lora_rank")),
        Operand("wv", ("heads", "v_head_dim", "kv_lora_rank")),
        Operand("rng_seed", ()),
    )
    + _MLA_FWD_OPTIONALS,
    outputs=(Output(_O_DIMS, dtype_like="q"),),
    shardable=frozenset({"batch", "heads"}),
)

_MLA_FWD_ABSORBED_SPEC = KernelSpec(
    name="mla_fwd_absorbed",
    operands=(
        Operand("q", _Q_DIMS),
        Operand("cache", _C_DIMS),
        Operand("rng_seed", ()),
    )
    + _MLA_FWD_OPTIONALS,
    outputs=(Output(_O_DIMS_ABSORBED, dtype_like="q"),),
    shardable=frozenset({"batch", "heads"}),
)


def _mla_build(
    q,
    cache,
    wk,
    wv,
    rng_seed,
    b_data,
    q_id,
    k_id,
    dropout_mask,
    *,
    mask_fn,
    bias_fn,
    b_spec,
    q_id_spec,
    k_id_spec,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
):
    """Shared builder for the materialized and absorbed forward kernels.

    Operand order is ``(q, cache, wk, wv, rng_seed, b_data, q_id, k_id,
    dropout_mask)`` and must match :func:`_mla_forward_kernel`'s parameters and
    the ``in_specs`` list below.
    """
    batch_size, q_seq_len, num_heads, qk_head_dim = q.shape
    kv_seq_len = cache.shape[1]
    v_head_dim = wv.shape[1] if wv is not None else cache.shape[-1]

    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_dk = max(pl.next_power_of_2(qk_head_dim), 16)
    block_dv = max(pl.next_power_of_2(v_head_dim), 16)
    block_r = max(pl.next_power_of_2(cache.shape[-1]), 16)

    # Like flash-attention-3, sequence lengths must tile the KV loop exactly.
    # The latent sequence is never padded, so no length mask is needed.
    if q_seq_len % block_q:
        raise ValueError(
            f"q_seq_len={q_seq_len} must be a multiple of block_q={block_q}"
        )
    if kv_seq_len % block_k:
        raise ValueError(
            f"kv_seq_len={kv_seq_len} must be a multiple of block_k={block_k}"
        )

    # Only the head and latent widths are padded; the zeros they introduce
    # contract away against the zero-padded columns of the other operand.
    q_pad = _pad_to_multiple(q, 3, block_dk)
    c_pad = _pad_to_multiple(cache, 2, block_r)

    grid_ = (q_seq_len // block_q, batch_size, num_heads)
    num_warps_ = num_warps or (4 if max(block_dk, block_dv) <= 64 else 8)

    kernel = functools.partial(
        _mla_forward_kernel,
        sm_scale=sm_scale,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        mask_fn=mask_fn,
        bias_fn=bias_fn,
        dropout_rate=dropout_rate,
        block_q=block_q,
        block_k=block_k,
        block_dk=block_dk,
        block_dv=block_dv,
    )

    wk_pad = (
        None
        if wk is None
        else _pad_to_multiple(_pad_to_multiple(wk, 1, block_dk), 2, block_r)
    )
    wv_pad = (
        None
        if wv is None
        else _pad_to_multiple(_pad_to_multiple(wv, 1, block_dv), 2, block_r)
    )

    in_specs: list[Any] = [
        pl.BlockSpec((None, block_q, None, block_dk), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, block_r), lambda _, j, k_: (j, 0, 0)),
        pl.BlockSpec((None, block_dk, block_r), lambda _, __, k_: (k_, 0, 0))
        if wk_pad is not None
        else None,
        pl.BlockSpec((None, block_dv, block_r), lambda _, __, k_: (k_, 0, 0))
        if wv_pad is not None
        else None,
        pl.BlockSpec((), lambda *_: ()),
        b_spec if b_data is not None else None,
    ]
    if q_id is not None or k_id is not None:
        in_specs.append(q_id_spec)
        in_specs.append(k_id_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)
    in_specs.append(
        pl.BlockSpec((None, None, block_q, kv_seq_len), lambda i, j, k_: (j, k_, i, 0))
        if dropout_mask is not None
        else None
    )

    out_spec = pl.BlockSpec(
        (None, block_q, None, block_dv), lambda i, j, k_: (j, i, k_, 0)
    )
    out_shape = jax.ShapeDtypeStruct(
        (batch_size, q_seq_len, num_heads, block_dv), q.dtype
    )

    out = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_spec,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mla_forward",
    )(q_pad, c_pad, wk_pad, wv_pad, rng_seed, b_data, q_id, k_id, dropout_mask)

    # Trim the head-dimension padding back to the logical output width.
    return out[..., :v_head_dim]


def _mla_fwd_absorbed_impl(q, cache, rng_seed, *rest, **static):
    """Absorbed spec has no ``wk``/``wv`` operands; supply them as ``None``."""
    return _mla_build(q, cache, None, None, rng_seed, *rest, **static)


mla_fwd_p = make_kernel_primitive(_MLA_FWD_SPEC, impl=_mla_build)
mla_fwd_absorbed_p = make_kernel_primitive(
    _MLA_FWD_ABSORBED_SPEC, impl=_mla_fwd_absorbed_impl
)


# ---------------------------------------------------------------------------
# Materialized reference (also the backward path)
# ---------------------------------------------------------------------------


def _materialize_kv(cache: jax.Array, wk, wv, num_heads: int):
    """``(K, V)`` from the shared latent; ``None`` projection -> ``C`` itself."""
    if wk is None:
        key = jnp.broadcast_to(
            cache[..., None, :], cache.shape[:-1] + (num_heads, cache.shape[-1])
        )
    else:
        key = jnp.einsum("...sr,hdr->...shd", cache, wk)
    if wv is None:
        value = jnp.broadcast_to(
            cache[..., None, :], cache.shape[:-1] + (num_heads, cache.shape[-1])
        )
    else:
        value = jnp.einsum("...sr,hdr->...shd", cache, wv)
    return key, value


def _pad_head_dim(x: jax.Array, target: int) -> jax.Array:
    """Zero-pad the last (head) axis of ``x`` up to ``target``."""
    return _pad_to_multiple(x, x.ndim - 1, target) if x.shape[-1] < target else x


def _materialized_mla(
    q,
    cache,
    wk,
    wv,
    *,
    mask,
    bias,
    rng,
    sm_scale,
    block_sizes,
    backward_pass_impl,
    num_warps,
    num_stages,
    interpret,
    debug,
    dropout_rate,
    dropout_impl,
):
    """``mha(q, C@W_K, C@W_V)`` sliced to the logical value width.

    ``mha`` requires the query, key and value head dimensions to coincide, while
    MLA lets the value head differ from the query/key head.  Padding the smaller
    side with zeros is exact: the zero channels carry no signal, so neither the
    attention weights nor the gradients of the real channels change.
    """
    num_heads = q.shape[-2]
    v_head_dim = wv.shape[1] if wv is not None else cache.shape[-1]
    key, value = _materialize_kv(cache, wk, wv, num_heads)
    common = max(key.shape[-1], value.shape[-1])
    out = mha(
        q=_pad_head_dim(q, common),
        k=_pad_head_dim(key, common),
        v=_pad_head_dim(value, common),
        mask=mask,
        bias=bias,
        rng=rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
    )
    return out[..., :v_head_dim]


def _make_mla_custom_vjp(
    *,
    mask_treedef,
    bias_treedef,
    num_mask_leaves: int,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
):
    def _combined(mask_leaves, bias_leaves):
        # Leaves are split by position: the first ``num_mask_leaves`` belong to
        # the mask, the rest to the bias.
        mask = _unflatten_optional_pytree(mask_treedef, tuple(mask_leaves))
        bias = _unflatten_optional_pytree(bias_treedef, tuple(bias_leaves))
        return mask, bias

    def _run_forward(q, cache, wk, wv, rng, mask, bias):
        batch_size, q_seq_len, n_heads, qk_head_dim = q.shape
        kv_seq_len = cache.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)
        rng_val = rng if rng is not None else jax.random.key(0)
        rng_seed = _rng_seed_from_key(rng_val)
        arrays, specs = _extract_fwd_data(
            mask,
            bias,
            rng,
            batch_size=batch_size,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            num_heads=n_heads,
            block_q=block_q,
            block_k=block_k,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )
        absorbed = wk is None and wv is None
        prim = mla_fwd_absorbed_p if absorbed else mla_fwd_p
        operands = (
            {"q": q, "cache": cache, "rng_seed": rng_seed, **arrays}
            if absorbed
            else {
                "q": q,
                "cache": cache,
                "wk": wk,
                "wv": wv,
                "rng_seed": rng_seed,
                **arrays,
            }
        )
        return shardable_kernel(
            prim,
            operands,
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            b_spec=specs["b_spec"],
            q_id_spec=specs["q_id_spec"],
            k_id_spec=specs["k_id_spec"],
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )

    @jax.custom_vjp
    def _mla(q, cache, wk, wv, rng, *mask_bias_leaves):
        n_mask = num_mask_leaves
        mask, bias = _combined(mask_bias_leaves[:n_mask], mask_bias_leaves[n_mask:])
        return _run_forward(q, cache, wk, wv, rng, mask, bias)

    def _mla_fwd(q, cache, wk, wv, rng, *mask_bias_leaves):
        out = _mla(q, cache, wk, wv, rng, *mask_bias_leaves)
        return out, (q, cache, wk, wv, rng, mask_bias_leaves)

    def _mla_bwd(res, do):
        q, cache, wk, wv, rng, mask_bias_leaves = res
        n_mask = num_mask_leaves
        mask, bias = _combined(mask_bias_leaves[:n_mask], mask_bias_leaves[n_mask:])
        # Exact: the kernel output equals mha(q, C@W_K, C@W_V) with the same
        # mask/bias, so its VJP is that of the materialized computation.
        _, vjp = jax.vjp(
            lambda q_, c_, wk_, wv_: _materialized_mla(
                q_,
                c_,
                wk_,
                wv_,
                mask=mask,
                bias=bias,
                rng=rng,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
                backward_pass_impl=backward_pass_impl,
                num_warps=num_warps,
                num_stages=num_stages,
                interpret=interpret,
                debug=debug,
                dropout_rate=dropout_rate,
                dropout_impl=dropout_impl,
            ),
            q,
            cache,
            wk,
            wv,
        )
        dq, dcache, dwk, dwv = vjp(do)
        return (dq, dcache, dwk, dwv, None, *(None,) * len(mask_bias_leaves))

    _mla.defvjp(_mla_fwd, _mla_bwd)
    return _mla


def mla(
    q: jax.Array,
    cache: jax.Array,
    wk: jax.Array | None = None,
    wv: jax.Array | None = None,
    *,
    mask: Any = None,
    bias: Any = None,
    rng: jax.Array | None = None,
    sm_scale: float | None = None,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "auto",
    num_warps: int | None = None,
    num_stages: int = 2,
    interpret: bool = False,
    debug: bool = False,
    dropout_rate: float = 0.0,
    dropout_impl: str = "counter",
):
    """Attention of ``q`` over a shared latent ``cache``.

    Args:
        q: queries ``[batch, q_seq, heads, qk_head_dim]``.
        cache: shared latent ``[batch, kv_seq, kv_lora_rank]``.
        wk: per-head key up-projection ``[heads, qk_head_dim, kv_lora_rank]``
            (``None`` -> ``K = cache``; then ``qk_head_dim == kv_lora_rank``).
        wv: per-head value up-projection ``[heads, v_head_dim, kv_lora_rank]``
            (``None`` -> ``V = cache``).
        mask, bias, rng, sm_scale, block_sizes, backward_pass_impl, num_warps,
        num_stages, interpret, debug, dropout_rate, dropout_impl: as in
        :func:`probjax.nn.pallas_kernels.mha`.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be rank 4 [B,Q,H,D], got shape {q.shape}")
    if cache.ndim != 3:
        raise ValueError(f"cache must be rank 3 [B,S,R], got shape {cache.shape}")
    if q.shape[0] != cache.shape[0]:
        raise ValueError(
            f"batch mismatch: q has {q.shape[0]}, cache has {cache.shape[0]}"
        )
    num_heads = q.shape[-2]
    qk_head_dim = q.shape[-1]
    kv_lora_rank = cache.shape[-1]
    if (wk is None) != (wv is None):
        raise ValueError(
            "provide both wk and wv (materialized) or neither (absorbed); "
            "mixing the two is not supported"
        )
    if wk is None and qk_head_dim != kv_lora_rank:
        raise ValueError(
            "absorbed mode (wk=None) requires qk_head_dim == kv_lora_rank "
            f"({qk_head_dim} != {kv_lora_rank})"
        )
    if wk is not None and wk.shape != (num_heads, qk_head_dim, kv_lora_rank):
        raise ValueError(
            f"wk must be {(num_heads, qk_head_dim, kv_lora_rank)}, got {wk.shape}"
        )
    if wv is not None and (wv.shape[0] != num_heads or wv.shape[2] != kv_lora_rank):
        raise ValueError(
            f"wv must be [heads, v_head_dim, {kv_lora_rank}], got {wv.shape}"
        )

    if sm_scale is None:
        # Materialized: logits are Q . (C@W_K) over the qk head dim.  Absorbed:
        # logits are Q . C over the latent width.
        sm_scale = 1.0 / math.sqrt(qk_head_dim)

    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)

    if jax.default_backend() == "cpu":
        interpret = True

    fn = _make_mla_custom_vjp(
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
        num_mask_leaves=len(mask_leaves),
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
    )
    return fn(q, cache, wk, wv, rng, *mask_leaves, *bias_leaves)


__all__ = ["mla", "mla_fwd_p", "mla_fwd_absorbed_p"]
