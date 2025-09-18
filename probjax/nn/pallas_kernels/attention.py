# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing fused attention forward and backward pass."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from .attention_mask_bias import (
    AttentionBias,
    AttentionMask,
)
from .utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    get_dot_precision,
    get_dropout_mask,
)


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    """
    Tile sizes parameterizing the attention kernel. These block sizes
    should be tuned for the model and hardware for optimal performance.

    Attributes:
        block_q: Block size along Q sequence length for forward kernel.
        block_k: Block size along KV sequence length for forward kernel.
        block_q_dkv: Block size along Q sequence length for dKV backward kernel.
        block_kv_dkv: Block size along KV sequence length for dKV backward kernel.
        block_q_dq: Block size along Q sequence length for dQ backward kernel.
        block_kv_dq: Block size along KV sequence length for dQ backward kernel.
    """

    block_q: int
    block_k: int

    block_q_dkv: int | None = None
    block_kv_dkv: int | None = None
    block_q_dq: int | None = None
    block_kv_dq: int | None = None

    @classmethod
    def get_default(cls):
        return BlockSizes(
            block_q=128,
            block_k=128,
            block_q_dkv=64,
            block_kv_dkv=64,
            block_q_dq=64,
            block_kv_dq=64,
        )

    @property
    def has_backward_blocks(self) -> bool:
        """Returns True if all backward blocks are specified for the fused

        dq and dk/dv backwards pass.
        """
        backward_blocks = [
            self.block_q_dkv,
            self.block_kv_dkv,
            self.block_q_dq,
            self.block_kv_dq,
        ]

        return all(b is not None for b in backward_blocks)


def mha_forward_kernel(
    q_ref: jax.Array,  # Query tensor
    k_ref: jax.Array,  # Key tensor
    v_ref: jax.Array,  # Input arrays
    # Dense bias that need to be materialized will be passed as a tensor
    b_ref: jax.Array | None,
    id_q_ref: jax.Array | None,  # optional mask data for q positions
    id_k_ref: jax.Array | None,  # optional mask data for k positions
    dropout_mask_ref: jax.Array | None,  # dropout mask
    # dynamic kv iterators (optional)
    index_offset_ref: jax.Array | None,
    index_offset_size_ref: jax.Array | None,
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    # Static elements
    sm_scale: float,
    head_dim: int,
    mask_fn: Callable | None = None,
    bias_fn: Callable | None = None,
    dropout_rate: float = 0.0,
    block_q: int,
    block_d: int,
    block_k: int,
):
    """
    Multi-Head Attention forward kernel.

    Args:
        q_ref: Query tensor reference.
        k_ref: Key tensor reference.
        v_ref: Value tensor reference.
        segment_ids_ref: Segment IDs tensor reference.
        o_ref: Output tensor reference.
        residual_refs: Residual tensor references.
        sm_scale: Softmax scaling factor.
        score_mod: Custom score modification function.
        mask_mod: Custom mask modification function.
        block_q: Block size for query.
        block_d: Block size for head dimension.
        block_k: Block size for key/value.
    """
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.full((block_q,), NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    d_mask = jnp.arange(block_d)[None] < head_dim
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    # Load the current Q tile into SRAM
    q = pl.load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
    if mask_fn or bias_fn:
        span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634  # log2(e)

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        if index_offset_ref is not None:
            # We retrieve the dynamic indices for the current block if offset is provided.
            start_k = jnp.sum(pl.load(index_offset_ref, (pl.dslice(start_k, 1),)))
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        qk = pl.dot(q, k.T, precision=precision)  # [block_q, block_k]

        # Scale this by user-provided factor (1 / sqrt(d_k) for original transformer).
        if sm_scale != 1.0:
            qk *= sm_scale


        # Seq ids for mask and bias
        if (bias_fn is not None) or (mask_fn is not None):
            span_k = start_k * block_k + jnp.arange(block_k)
        # Apply bias to qk: dense tensor via b_ref; function via bias_fn
        if bias_fn is not None:
            if b_ref is not None:
                b_chunk = pl.load(b_ref, (slice(None), curr_k_slice))
            else:
                b_chunk = None
            qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
        # boolean mask for the current qk slice
        if mask_fn is not None:
            if id_k_ref is not None:
                id_k = None if id_k_ref is None else pl.load(id_k_ref, (curr_k_slice,))
            elif id_q is not None:
                # Otherwise reuse id_q if available
                id_k = None if id_q_ref is None else pl.load(id_q_ref, (curr_k_slice,))
            else:
                id_k = None
            mask = mask_fn(span_q, span_k, id_q, id_k)
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        # Scale logits to convert from base-2 to the natural log domain.
        # This is based on the identity: e^x = 2^(x * log2(e)).
        qk *= LOG2E
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=jnp.nan)
        if dropout_rate > 0 and dropout_mask_ref is not None:
            dmask = pl.load(dropout_mask_ref, (slice(None), curr_k_slice))
            s_curr = jnp.where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = pl.dot(s_curr.astype(v.dtype), v, precision=precision)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)

    if index_offset_size_ref is not None:
        # Dynamic per-(B,H,QB) iterators over KV blocks.
        # Convert to a scalar upper bound; spec may expose a length-1 vector per tile.
        iters = index_offset_size_ref[...]
        o, m_i, l_i = lax.fori_loop(lower_bound, iters, body, (o, m_i, l_i))
    else:
        o, m_i, l_i = lax.fori_loop(lower_bound, upper_bound, body, (o, m_i, l_i))

    # We keep an unscaled version of o during the scan over seq_len. Scaling it
    # by the last l_i gives us the correct final output. See section 3.1.1 in the
    # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
    l_i = jnp.where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log2(l_i)
    # Write output to dram.
    pl.store(o_ref, (slice(None), slice(None)), val=o.astype(o_ref.dtype), mask=d_mask)


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref):
    """
    Preprocesses the backward pass by computing the delta.

    Args:
        out_ref: Output tensor reference.
        dout_ref: Gradient of the output tensor reference.
        delta_ref: Delta tensor reference.
    """
    # load
    o = out_ref[...].astype(jnp.float32)
    do = dout_ref[...].astype(jnp.float32)
    # compute
    delta = jnp.sum(o * do, axis=1)
    # write-back
    delta_ref[...] = delta.astype(delta_ref.dtype)


@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int, debug: bool, interpret: bool):
    """
    Preprocesses the backward pass.

    Args:
        out: Output tensor.
        do: Gradient of the output tensor.
        lse: Log-sum-exp tensor.
        block_q: Block size for query.
        debug: Whether to enable debugging.
        interpret: Whether to interpret the kernel.

    Returns:
        The delta tensor.
    """
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
    delta = pl.pallas_call(
        _preprocess_backward_kernel,
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        compiler_params=plgpu.TritonCompilerParams(num_warps=4, num_stages=3),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do)
    return delta


# This kernel computes dK_i, dV_i and dQ_i in parallel across the sequence
# length.
# Inspired by the triton tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    id_q_ref: jax.Array | None,
    id_k_ref: jax.Array | None,
    b_ref: jax.Array | None,
    dropout_mask_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Optional dynamic iterators
    q_index_offset_ref: jax.Array | None,
    q_index_offset_size_ref: jax.Array | None,
    kv_index_offset_ref: jax.Array | None,
    kv_index_offset_size_ref: jax.Array | None,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    # Static
    sm_scale: float,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    block_d: int,
    bias_fn: Callable | None = None,
    mask_fn: Callable | None = None,
    bias_fn_grad=None,
    dropout_rate: float = 0.0,
    head_dim: int,
):
    """
    Multi-Head Attention backward kernel.

    Args:
        q_ref: Query tensor reference.
        k_ref: Key tensor reference.
        v_ref: Value tensor reference.
        segment_ids_ref: Segment IDs tensor reference.
        out_ref: Output tensor reference.
        do_scaled_ref: Scaled gradient of the output tensor reference.
        lse_ref: Log-sum-exp tensor reference.
        delta_ref: Delta tensor reference.
        dq_ref: Gradient of the query tensor reference.
        dk_ref: Gradient of the key tensor reference.
        dv_ref: Gradient of the value tensor reference.
        sm_scale: Softmax scaling factor.
        block_q_dkv: Block size for query in dKV backward kernel.
        block_kv_dkv: Block size for key/value in dKV backward kernel.
        block_q_dq: Block size for query in dQ backward kernel.
        block_kv_dq: Block size for key/value in dQ backward kernel.
        block_d: Block size for head dimension.
    """
    del out_ref  # Not needed
    q_seq_len = q_ref.shape[0]
    kv_seq_len = k_ref.shape[0]

    # Scan #1: dK and dV
    #   1. Load a block of K and V of size (block_kv_dkv, head_dim) in SMEM.
    #   2. Iterate through Q in chunks of (block_q_dkv, head_dim) to accumulate
    #      dK and dV.
    start_b = pl.program_id(0)
    start_h = pl.program_id(1)
    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)

    dv = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)
    mask_d = jnp.arange(block_d)[None] < head_dim

    v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
    k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
    if id_k_ref is not None:
        id_k = pl.load(id_k_ref, (curr_k_slice,))
    elif id_q_ref is not None:
        # Otherwise reuse id_q if available
        id_k = pl.load(id_q_ref, (curr_k_slice,))
    else:
        id_k = None

    LOG2E = 1.4426950408889634  # log2(e)

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

        q = pl.load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
            # boolean mask for the current qk slice
            if bias_fn is not None:
                b_chunk = (
                    pl.load(b_ref, (curr_q_slice, curr_k_slice)) if b_ref is not None else None
                )
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)

            if mask_fn is not None:
                id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
                mask = mask_fn(span_q, span_k, id_q, id_k)
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
        # No built-in causal; pass as mask via mask if needed.

        qk *= LOG2E
        lse = pl.load(lse_ref, (curr_q_slice,))
        di = pl.load(delta_ref, (curr_q_slice,))
        do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)
        dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
        dp = dp + dp_dropped
        # Apply dropout scaling consistently with forward if present
        if dropout_mask_ref is not None and dropout_rate > 0:
            dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            p = jnp.where(dmask, 0, p / (1 - dropout_rate))
            # Scale dp consistently with forward scaling
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (
                jnp.zeros_like(dp) - di[:, None]
            )
        # Accumulate dV
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad:
            # Compute the gradient of score_mod with respect to qk
            # TODO Refactor
            grad_score_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod  # Element-wise multiplication
        dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)

        return dv, dk

    # Iterate over Q blocks for this (B,H, KB-tile)
    if kv_index_offset_ref is not None and kv_index_offset_size_ref is not None:
        iters = kv_index_offset_size_ref[...]

        def dyn_q(iter_q, carry):
            start_q = jnp.sum(
                pl.load(kv_index_offset_ref, (pl.dslice(iter_q, 1),))
            )
            return inner_loop_dkdv(start_q, carry)

        dv, dk = lax.fori_loop(0, iters, dyn_q, (dv, dk))
    else:
        dv, dk = lax.fori_loop(
            0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
        )

    dv_ref = pl.store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=mask_d
    )
    dk_ref = pl.store(
        dk_ref, (slice(None), slice(None)), val=dk.astype(dk_ref.dtype), mask=mask_d
    )
    # dv_ref[...] = dv.astype(dv_ref.dtype)
    # dk_ref[...] = dk.astype(dk_ref.dtype)

    del dv, dk

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.dslice(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros([block_q_dq, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
    # segment ids not used in this kernel
    lse = pl.load(lse_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pl.load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
        v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)

        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
            # boolean mask for the current qk slice
            if bias_fn is not None:
                b_chunk = (
                    pl.load(b_ref, (curr_q_slice, curr_k_slice)) if b_ref is not None else None
                )
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)

            if mask_fn is not None:
                id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
                if id_k_ref is not None:
                    id_k = pl.load(id_k_ref, (curr_k_slice,))
                elif id_q_ref is not None:
                    # Otherwise reuse id_q if available
                    id_k = pl.load(id_q_ref, (curr_k_slice,))
                else:
                    id_k = None

                mask = mask_fn(span_q, span_k, id_q, id_k)
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
        # No built-in causal; pass as mask via mask if needed.

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + dp_dropped
        if dropout_mask_ref is not None and dropout_rate > 0:
            dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            p = jnp.where(dmask, 0, p / (1 - dropout_rate))
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (
                jnp.zeros_like(dp) - di[:, None]
            )
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        if bias_fn_grad:
            # Compute the gradient of score_mod with respect to qk
            # TODO Refactor
            grad_score_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod  # Element-wise multiplication

        dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)

        return dq

    # Iterate over KV blocks intersecting this (B,H, QB-tile)
    if q_index_offset_ref is not None and q_index_offset_size_ref is not None:
        iters = q_index_offset_size_ref[...]

        def dyn_k(iter_k, dq_c):
            start_k = jnp.sum(
                pl.load(q_index_offset_ref, (pl.dslice(iter_k, 1),))
            )
            return inner_loop_dq(start_k, dq_c)

        dq = lax.fori_loop(0, iters, dyn_k, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop_dq, dq)

    pl.store(
        dq_ref, (slice(None), slice(None)), val=dq.astype(dq_ref.dtype), mask=mask_d
    )
    # dq_ref[...] = dq.astype(dq_ref.dtype)


def _mha_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | None,
    bias: AttentionBias | None,
    rng: jax.Array | None,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    *,
    output_activations: bool = False,
):
    """Shared implementation for MHA forward.

    If output_activations=True returns (out, (q,k,v,mask,bias_mod,out,lse)).
    Otherwise returns out only. Mirrors flash_attention.py structure.
    """
    del backward_pass_impl  # Only one impl at the moment.
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_d = pl.next_power_of_2(head_dim)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if block_d <= 64 else 8)

    # Optional block-sparse iterators
    index_offset = index_offset_size = None
    if mask is not None:
        index_offset, index_offset_size = mask.query_iterator_indices(
            q_seq_len, kv_seq_len, block_q, block_k
        )

    # Bias tensor (dense) extracted if available
    b_data = bias.get_data() if bias is not None else None

    # Mask data arrays
    if mask is not None:
        q_id, k_id = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
    else:
        q_id = k_id = None

    # Dropout mask
    if dropout_rate > 0:
        assert rng is not None, "prng_key must be provided when dropout_rate>0"
        dropout_mask = get_dropout_mask(
            (batch_size, num_heads, q_seq_len, kv_seq_len),
            prng_key=rng,
            rate=dropout_rate,
        )
    else:
        dropout_mask = None

    # Build kernel
    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        head_dim=head_dim,
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        mask_fn=mask.__call__ if mask is not None else None,
        bias_fn=bias.__call__ if bias is not None else None,
        dropout_rate=dropout_rate,
    )

    # Input specs (q,k,v)
    in_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
    ]

    # Bias spec
    if b_data is not None:
        b_specs = bias.get_block_spec(
            q_len=q_seq_len, kv_len=kv_seq_len, block_q=block_q, block_kv=block_k
        )
        in_specs.append(b_specs)
    else:
        in_specs.append(None)

    # q/k mask data specs
    if q_id is not None or k_id is not None:
        q_id_spec, k_id_spec = mask.get_data_block_spec(q_seq_len, kv_seq_len)
        in_specs.append(q_id_spec)
        in_specs.append(k_id_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)

    # Dropout mask spec
    if dropout_mask is not None:
        in_specs.append(
            pl.BlockSpec(
                (None, None, block_q, kv_seq_len), lambda i, j, k_: (j, k_, i, 0)
            )
        )
    else:
        in_specs.append(None)

    # Dynamic iterator specs (kv)
    if index_offset is not None and index_offset_size is not None:
        index_offset_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: (i, 0)), block_shape=((None, block_k))
        )
        index_offset_size_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: (i)), block_shape=((None,))
        )
        in_specs.append(index_offset_spec)
        in_specs.append(index_offset_size_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)

    # Output specs & shapes
    if output_activations:
        out_shape = [
            jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            jax.ShapeDtypeStruct(
                shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32
            ),
        ]
        out_specs = [
            pl.BlockSpec(
                (None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)
            ),
            pl.BlockSpec((None, None, block_q), lambda i, j, k_: (j, k_, i)),
        ]
    else:
        out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
        out_specs = pl.BlockSpec(
            (None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)
        )

    pallas_out = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, b_data, q_id, k_id, dropout_mask, index_offset, index_offset_size)

    if output_activations:
        out, lse = pallas_out
        return out, (q, k, v, mask, bias, rng, out, lse)
    return pallas_out


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=[6, 7, 8, 9, 10, 11, 12, 13, 14],
)
def mha(
    q,
    k,
    v,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    rng: jax.Array | None = None,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    dropout_rate: float = 0.0,
):
    """Multi-Head Attention public API (forward only in primal eval)."""
    return _mha_impl(
        **locals(),
        output_activations=False,
    )


def _mha_forward(*args):
    """Forward wrapper for custom VJP using shared impl."""
    out, residuals = _mha_impl(
        *args,
        output_activations=True,
    )
    return out, residuals


def _mha_backward(
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    res,
    do,
):
    """
    Backward pass for the Multi-Head Attention mechanism.

    Args:
        sm_scale: Softmax scaling factor.
        block_sizes: Block sizes for the attention kernel.
        backward_pass_impl: Implementation for the backward pass.
        num_warps: Number of warps for Triton.
        num_stages: Number of stages for Triton.
        grid: Grid dimensions for Triton.
        interpret: Whether to interpret the kernel.
        debug: Whether to enable debugging.
        res: Residuals from the forward pass.
        do: Gradient of the output tensor.

    Returns:
        Gradients of the query, key, and value tensors.
    """
    del num_stages, grid
    q, k, v, mask, bias, rng, out, lse = res
    mask = mask if mask is not None else mask
    bias = bias if bias is not None else bias

    if backward_pass_impl == "triton":
        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        batch_size, q_seq_len, num_heads, head_dim = q.shape
        block_d = pl.next_power_of_2(head_dim)
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
        block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
        block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
        block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)
        # Debug prints removed

        if q_seq_len // block_q_dq != kv_seq_len // block_kv_dkv:
            raise ValueError(
                "q_seq_len and kv_seq_len must be divided into the same "
                "number of blocks for the fused backward pass."
            )

        delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)
        out_shapes = [
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]

        # Prepare bias array for backward (for dense bias instances)
        b_data = bias.get_data() if (bias is not None) else None

        in_specs = [
            # q, k, v
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            # segment_ids
            # data_q
            None,
            # data_k
            None,
            # bias
            (
                None
                if b_data is None
                else bias.get_block_spec(
                    q_len=q_seq_len,
                    kv_len=kv_seq_len,
                    # Use full sequence extents so both dKdV and dQ loops
                    # can slice (curr_q_slice, curr_k_slice) regardless of
                    # their per-loop tile sizes.
                    block_q=q_seq_len,
                    block_kv=kv_seq_len,
                )
            ),
            # dropout mask
            None,
            # out, do, lse, delta
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]
        # Reserve 4 slots for optional dynamic iterators
        in_specs.extend([None, None, None, None])
        # Prepare optional mask data specs (q_id_ref, k_id_ref) via the mask
        # Backward: pass mask data arrays with BlockSpecs adapted to backward grid (B, H, KV).
        q_data = k_data = None
        if isinstance(mask, AttentionMask):
            q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
        # q_id_ref spec (per-query indices). Kernel loads with (curr_q_slice,)
        if q_data is not None:
            if getattr(q_data, "ndim", None) == 2:
                # Shape (B, Q) -> slice by batch via grid dim 0
                in_specs[3] = pl.BlockSpec((None, q_seq_len), lambda i, j, k: (i, 0))
            elif getattr(q_data, "ndim", None) == 1:
                # Shape (Q,) -> head-independent
                in_specs[3] = pl.BlockSpec((q_seq_len,), lambda i, j, k: (0,))
        # k_id_ref spec (per-key indices). Kernel loads with (curr_k_slice,)
        if k_data is not None:
            if getattr(k_data, "ndim", None) == 2:
                # Shape (B, K) -> slice by batch via grid dim 0
                in_specs[4] = pl.BlockSpec((None, kv_seq_len), lambda i, j, k: (i, 0))
            elif getattr(k_data, "ndim", None) == 1:
                # Shape (K,) -> head-independent
                in_specs[4] = pl.BlockSpec((kv_seq_len,), lambda i, j, k: (0,))

        if dropout_rate > 0:
            assert rng is not None
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len),
                prng_key=rng,
                rate=dropout_rate,
            )
            in_specs[6] = pl.BlockSpec(
                (None, None, q_seq_len, kv_seq_len), lambda i, j, _: (i, j, 0, 0)
            )
        else:
            dropout_mask = None

        grid = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv))
        num_warps_ = num_warps
        if num_warps_ is None:
            if (
                block_q_dkv * block_kv_dkv < 128 * 128
                or block_q_dq * block_kv_dq < 128 * 128
            ):
                num_warps_ = 4
            else:
                num_warps_ = 8

        # Provide gradient modifier for stateless biases (no dense data)
        bias_fn_grad = bias.grad if (bias is not None) else None

        # Optional block-sparse iterators
        q_index_offset = q_index_offset_size = kv_index_offset = (
            kv_index_offset_size
        ) = None
        if mask is not None:
            # Build block masks using the respective backward block sizes
            # Per-QB iterators over KV blocks for dQ
            q_index_offset, q_index_offset_size = mask.query_iterator_indices(
                q_seq_len, kv_seq_len, block_q_dq, block_kv_dq
            )
            kv_index_offset, kv_index_offset_size = mask.kv_iterator_indices(
                q_seq_len, kv_seq_len, block_q_dkv, block_kv_dkv
            )

            num_kv_blocks_dq = pl.cdiv(kv_seq_len, block_kv_dq)
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            # Debug prints removed
            # Map per-tile vectors/scalars. Grid dims are (B, H, KB) and we also reuse KB as QB
            # (enforced by the check above).
            if q_index_offset is not None:
                q_index_offset_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k, 0)), block_shape=((None, num_kv_blocks_dq))
                )
                q_index_offset_size_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k)), block_shape=((None,))
                )
                in_specs[-4] = q_index_offset_spec  # q_index_offset
                in_specs[-3] = q_index_offset_size_spec  # q_index_offset_size
            if kv_index_offset is not None:
                kv_index_offset_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k, 0)), block_shape=((None, num_q_blocks_dkdv))
                )

                kv_index_offset_size_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k)), block_shape=((None,))
                )
                in_specs[-2] = kv_index_offset_spec  # kv_index_offset
                in_specs[-1] = kv_index_offset_size_spec  # kv_index_offset_size

        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                sm_scale=sm_scale,
                bias_fn=bias.__call__ if bias is not None else None,
                mask_fn=mask.__call__ if mask is not None else None,
                bias_fn_grad=bias_fn_grad,
                dropout_rate=dropout_rate,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_q_dq=block_q_dq,
                block_kv_dq=block_kv_dq,
                block_d=block_d,
                head_dim=head_dim,
            ),
            out_shape=out_shapes,
            in_specs=in_specs,
            grid=grid,
            out_specs=[
                pl.BlockSpec(
                    (None, block_q_dq, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),  # dq
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),  # dk
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),  # dv
                ),
            ],
            name="mha_backward",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.TritonCompilerParams(
                num_warps=num_warps_, num_stages=2
            ),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            out,
            do,
            lse,
            delta,
            q_index_offset,
            q_index_offset_size,
            kv_index_offset,
            kv_index_offset_size,
        )
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, None, None, None


mha.defvjp(_mha_forward, _mha_backward)
