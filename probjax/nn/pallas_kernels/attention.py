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
from jax._src import ad_util
from jax import lax
from jax.extend.core import Primitive
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.interpreters import ad, batching, mlir

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


def _dropout_mask_counter(
    rng_key: jax.Array,
    batch_idx: jax.Array,
    head_idx: jax.Array,
    q_idx: jax.Array,
    k_idx: jax.Array,
    rate: float,
) -> jax.Array:
    """Counter-based dropout mask for a [Q, K] tile."""
    seed = rng_key.astype(jnp.uint32)
    seed = seed[0] ^ (seed[1] * jnp.uint32(0x9E3779B9))
    b = jnp.uint32(batch_idx)
    h = jnp.uint32(head_idx)
    q = q_idx.astype(jnp.uint32)[:, None]
    k = k_idx.astype(jnp.uint32)[None, :]
    x = seed
    x ^= b * jnp.uint32(0x85EBCA6B)
    x ^= h * jnp.uint32(0xC2B2AE35)
    x ^= q * jnp.uint32(0x27D4EB2F)
    x ^= k * jnp.uint32(0x165667B1)
    x ^= x >> jnp.uint32(16)
    x *= jnp.uint32(0x7FEB352D)
    x ^= x >> jnp.uint32(15)
    x *= jnp.uint32(0x846CA68B)
    x ^= x >> jnp.uint32(16)
    u = x.astype(jnp.float32) * (1.0 / float(2**32))
    return u < rate


@jax.tree_util.register_pytree_node_class
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

    block_q: int = 128
    block_k: int = 128

    block_q_dkv: int = 64
    block_kv_dkv: int = 64
    block_q_dq: int = 64
    block_kv_dq: int = 64

    @classmethod
    def init_default(
        cls,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        block_q_dkv: int,
        block_kv_dkv: int,
        block_q_dq: int,
        block_kv_dq: int,
        backward_pass_impl: str = "triton_fused",
    ) -> BlockSizes:
        """Return block sizes adjusted to be backward-compatible when fused.

        The fused backward pass requires that the number of Q tiles and KV tiles
        match along the grid dimension. Concretely, we need
            ceil_div(q_len, block_q_dq) == ceil_div(kv_len, block_kv_dkv).

        This method adjusts only `block_q_dq` and `block_kv_dkv` to satisfy the
        equality while keeping the forward and the other backward block specs
        unchanged. If the provided specs already satisfy the constraint, they
        are returned as-is.
        """

        # Helper for ceil-div without importing pallas utilities here.
        def cdiv(a: int, b: int) -> int:
            return (a + b - 1) // b if b > 0 else 0

        # Start from requested specs
        bq = block_q
        bk = block_k
        bq_dkv = block_q_dkv
        bkv_dkv = block_kv_dkv
        bq_dq = block_q_dq
        bkv_dq = block_kv_dq

        # Only the fused backward cares about matching tile counts. If the
        # split (separate dKdV and dQ) backward is used, we can keep blocks
        # independent to allow efficiency when q_len << kv_len.
        if backward_pass_impl == "triton_fused":
            nq = max(cdiv(q_len, max(bq_dq, 1)), 1)
            nkv = max(cdiv(kv_len, max(bkv_dkv, 1)), 1)
            if nq != nkv:
                # Target a common number of tiles. Choose the larger to avoid
                # decreasing parallelism unnecessarily.
                n = max(nq, nkv)
                # Derive block sizes that yield exactly `n` tiles.
                # Using ceil_div ensures cdiv(len, block) == n.
                bq_dq = max((q_len + n - 1) // n, 1)
                bkv_dkv = max((kv_len + n - 1) // n, 1)

        return cls(
            block_q=bq,
            block_k=bk,
            block_q_dkv=bq_dkv,
            block_kv_dkv=bkv_dkv,
            block_q_dq=bq_dq,
            block_kv_dq=bkv_dq,
        )

    @classmethod
    def get_default(cls) -> BlockSizes:
        """Returns default block sizes."""
        return cls()

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

    def tree_flatten(self):
        aux = (
            self.block_q,
            self.block_k,
            self.block_q_dkv,
            self.block_kv_dkv,
            self.block_q_dq,
            self.block_kv_dq,
        )
        return (), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)




def mha_forward_kernel(
    q_ref: jax.Array,  # Query tensor
    k_ref: jax.Array,  # Key tensor
    v_ref: jax.Array,  # Input arrays
    # Dense bias that need to be materialized will be passed as a tensor
    b_ref: jax.Array | None,
    id_q_ref: jax.Array | None,  # optional mask data for q positions
    id_k_ref: jax.Array | None,  # optional mask data for k positions
    dropout_mask_ref: jax.Array | None,  # dropout mask
    rng_ref: jax.Array | None,  # rng key for counter-based dropout
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
    # TODO For per batch id_q or id_k, we should not slice along curr_q_slice here
    id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
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
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
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


def mha_jvp_from_lse_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    dq_ref: jax.Array,
    dk_ref: jax.Array,
    dv_ref: jax.Array,
    b_ref: jax.Array | None,
    id_q_ref: jax.Array | None,
    id_k_ref: jax.Array | None,
    dropout_mask_ref: jax.Array | None,
    rng_ref: jax.Array | None,
    lse_ref: jax.Array,
    index_offset_ref: jax.Array | None,
    index_offset_size_ref: jax.Array | None,
    do_ref: Any,
    *,
    sm_scale: float,
    head_dim: int,
    mask_fn: Callable | None = None,
    bias_fn: Callable | None = None,
    bias_fn_grad: Callable | None = None,
    dropout_rate: float = 0.0,
    block_q: int,
    block_d: int,
    block_k: int,
):
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    d_mask = jnp.arange(block_d)[None] < head_dim
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pl.load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    dq = pl.load(dq_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
    span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634  # log2(e)

    lse = pl.load(lse_ref, (curr_q_slice,))
    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, do_acc):
        if index_offset_ref is not None:
            start_k = jnp.sum(pl.load(index_offset_ref, (pl.dslice(start_k, 1),)))
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pl.load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pl.load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = pl.dot(q, k.T, precision=precision)
        dqk = pl.dot(dq, k.T, precision=precision) + pl.dot(q, dk.T, precision=precision)
        if sm_scale != 1.0:
            qk *= sm_scale
            dqk *= sm_scale
        qk_pre_mod = qk

        span_k = start_k * block_k + jnp.arange(block_k)
        if bias_fn is not None:
            if b_ref is not None:
                b_chunk = pl.load(b_ref, (slice(None), curr_k_slice))
            else:
                b_chunk = None
            qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
        if mask_fn is not None:
            if id_k_ref is not None:
                id_k = pl.load(id_k_ref, (curr_k_slice,))
            elif id_q is not None:
                id_k = pl.load(id_q_ref, (curr_k_slice,))
            else:
                id_k = None
            mask = mask_fn(span_q, span_k, id_q, id_k)
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
            dqk = jnp.where(mask, dqk, 0.0)

        if bias_fn_grad is not None:
            grad_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            dqk = dqk * grad_mod

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
            p_drop = jnp.where(dmask, 0, p / (1 - dropout_rate))
        else:
            p_drop = p

        row_sum = jnp.sum(dqk * p, axis=-1)
        dP = p * (dqk - row_sum[:, None])
        if dropout_rate > 0:
            dP = jnp.where(dmask, 0, dP / (1 - dropout_rate))

        do_acc = do_acc + pl.dot(dP.astype(v.dtype), v, precision=precision)
        do_acc = do_acc + pl.dot(p_drop.astype(v.dtype), dv, precision=precision)
        return do_acc

    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)
    if index_offset_size_ref is not None:
        iters = index_offset_size_ref[...]
        do = lax.fori_loop(lower_bound, iters, body_jvp, do)
    else:
        do = lax.fori_loop(lower_bound, upper_bound, body_jvp, do)

    pl.store(do_ref, (slice(None), slice(None)), val=do.astype(do_ref.dtype), mask=d_mask)


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
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do)
    return delta


# This kernel computes dK_i, dV_i and dQ_i in parallel across the sequence
# length. Specifically, it fuses the two scans over Q and KV into a single kernel.
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
    rng_ref: jax.Array | None,
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
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)

        q = pl.load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            # boolean mask for the current qk slice
            if bias_fn is not None:
                b_chunk = (
                    pl.load(b_ref, (curr_q_slice, curr_k_slice))
                    if b_ref is not None
                    else None
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
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
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
            start_q = jnp.sum(pl.load(kv_index_offset_ref, (pl.dslice(iter_q, 1),)))
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
                    pl.load(b_ref, (curr_q_slice, curr_k_slice))
                    if b_ref is not None
                    else None
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
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
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
            start_k = jnp.sum(pl.load(q_index_offset_ref, (pl.dslice(iter_k, 1),)))
            return inner_loop_dq(start_k, dq_c)

        dq = lax.fori_loop(0, iters, dyn_k, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop_dq, dq)

    pl.store(
        dq_ref, (slice(None), slice(None)), val=dq.astype(dq_ref.dtype), mask=mask_d
    )
    # dq_ref[...] = dq.astype(dq_ref.dtype)


# ------------------ Split Backward Kernels (separate passes) -----------------


def mha_backward_kernel_split_dkdv(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    id_q_ref: jax.Array | None,
    id_k_ref: jax.Array | None,
    b_ref: jax.Array | None,
    dropout_mask_ref: jax.Array | None,
    rng_ref: jax.Array | None,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Optional dynamic iterators (KV -> Q)
    index_offset_ref: jax.Array | None,
    index_offset_size_ref: jax.Array | None,
    # Outputs
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    mask_fn: Callable | None,
    bias_fn: Callable | None,
    bias_fn_grad: Callable | None,
    dropout_rate: float,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_d: int,
    head_dim: int,
):
    """Computes dK and dV in a dedicated pass iterating over Q blocks."""
    q_seq_len = q_ref.shape[0]
    block_mask = jnp.arange(block_d)[None] < head_dim
    start_b = pl.program_id(0)
    start_h = pl.program_id(1)
    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)

    dv = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)

    v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
    k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)

    if id_k_ref is not None:
        id_k = pl.load(id_k_ref, (curr_k_slice,))
    elif id_q_ref is not None:
        id_k = pl.load(id_q_ref, (curr_k_slice,))
    else:
        id_k = None

    LOG2E = 1.4426950408889634

    def inner_loop(start_q, carry):
        if index_offset_ref is not None:
            start_q = jnp.sum(pl.load(index_offset_ref, (pl.dslice(start_q, 1),)))
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
        dv_acc, dk_acc = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)
        q = pl.load(q_ref, (curr_q_slice, slice(None)), mask=block_mask, other=0.0)
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre = qk
        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            if b_ref is not None and bias_fn is not None:
                b_chunk = pl.load(b_ref, (curr_q_slice, curr_k_slice))
            else:
                b_chunk = None
            if bias_fn is not None:
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
            if mask_fn is not None:
                id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
                m = mask_fn(span_q, span_k, id_q, id_k)
                qk = jnp.where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        lse = pl.load(lse_ref, (curr_q_slice,))
        di = pl.load(delta_ref, (curr_q_slice,))
        do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)
        dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
        dp = dp + dp_dropped
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
            p = jnp.where(dmask, 0, p / (1 - dropout_rate))
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (
                jnp.zeros_like(dp) - di[:, None]
            )
        dv_acc = dv_acc + pl.dot(p.astype(do.dtype).T, do)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad is not None:
            grad_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_mod
        dk_acc = dk_acc + pl.dot(ds.astype(q_ref.dtype).T, q)
        return dv_acc, dk_acc

    if index_offset_size_ref is not None:
        dv, dk = lax.fori_loop(0, index_offset_size_ref[...], inner_loop, (dv, dk))
    else:
        dv, dk = lax.fori_loop(0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop, (dv, dk))

    pl.store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=block_mask
    )
    pl.store(
        dk_ref, (slice(None), slice(None)), val=dk.astype(dk_ref.dtype), mask=block_mask
    )


def mha_backward_kernel_split_dq(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    id_q_ref: jax.Array | None,
    id_k_ref: jax.Array | None,
    b_ref: jax.Array | None,
    dropout_mask_ref: jax.Array | None,
    rng_ref: jax.Array | None,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Optional dynamic iterators (Q -> KV)
    index_offset_ref: jax.Array | None,
    index_offset_size_ref: jax.Array | None,
    # Outputs
    dq_ref,
    *,
    sm_scale: float,
    mask_fn: Callable | None,
    bias_fn: Callable | None,
    bias_fn_grad: Callable | None,
    dropout_rate: float,
    block_q_dq: int,
    block_kv_dq: int,
    block_d: int,
    head_dim: int,
):
    """Computes dQ in a dedicated pass iterating over KV blocks."""
    kv_seq_len = k_ref.shape[0]
    block_mask = jnp.arange(block_d)[None] < head_dim
    start_b = pl.program_id(0)
    start_h = pl.program_id(1)
    start_q = pl.program_id(2)
    curr_q_slice = pl.dslice(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)

    dq = jnp.zeros([block_q_dq, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)), mask=block_mask, other=0.0)
    lse = pl.load(lse_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pl.load(delta_ref, (curr_q_slice,))

    LOG2E = 1.4426950408889634

    def inner_loop(start_k, dq_c):
        if index_offset_ref is not None:
            start_k = jnp.sum(pl.load(index_offset_ref, (pl.dslice(start_k, 1),)))
        span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pl.load(k_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
        v = pl.load(v_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre = qk
        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            if b_ref is not None and bias_fn is not None:
                b_chunk = pl.load(b_ref, (curr_q_slice, curr_k_slice))
            else:
                b_chunk = None
            if bias_fn is not None:
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
            if mask_fn is not None:
                id_q = None if id_q_ref is None else pl.load(id_q_ref, (curr_q_slice,))
                if id_k_ref is not None:
                    id_k = pl.load(id_k_ref, (curr_k_slice,))
                elif id_q_ref is not None:
                    id_k = pl.load(id_q_ref, (curr_k_slice,))
                else:
                    id_k = None
                m = mask_fn(span_q, span_k, id_q, id_k)
                qk = jnp.where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + dp_dropped
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_key = pl.load(rng_ref, (slice(None),))
                dmask = _dropout_mask_counter(
                    rng_key, start_b, start_h, span_q, span_k, dropout_rate
                )
            p = jnp.where(dmask, 0, p / (1 - dropout_rate))
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (
                jnp.zeros_like(dp) - di[:, None]
            )
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad is not None:
            grad_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_mod
        dq_c = dq_c + pl.dot(ds.astype(k.dtype), k).astype(dq_c.dtype)
        return dq_c

    if index_offset_size_ref is not None:
        dq = lax.fori_loop(0, index_offset_size_ref[...], inner_loop, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop, dq)

    pl.store(
        dq_ref, (slice(None), slice(None)), val=dq.astype(dq_ref.dtype), mask=block_mask
    )


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
    dropout_impl: str,
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
        if dropout_impl == "materialize":
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len),
                prng_key=rng,
                rate=dropout_rate,
            )
        elif dropout_impl == "counter":
            dropout_mask = None
        else:
            raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")
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
        q_id_spec, k_id_spec = mask.get_data_block_spec(
            q_seq_len, kv_seq_len, block_q, block_k
        )
        in_specs.append(q_id_spec)
        in_specs.append(k_id_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)

    # Dropout mask + rng specs
    if dropout_mask is not None:
        in_specs.append(
            pl.BlockSpec(
                (None, None, block_q, kv_seq_len), lambda i, j, k_: (j, k_, i, 0)
            )
        )
    else:
        in_specs.append(None)
    in_specs.append(pl.BlockSpec((2,), lambda *_: (0,)))

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
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(
        q,
        k,
        v,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        rng,
        index_offset,
        index_offset_size,
    )

    if output_activations:
        out, lse = pallas_out
        return out, (q, k, v, rng, out, lse)
    return pallas_out


def _mha_impl_jvp_from_lse(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    lse: jax.Array,
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
    dropout_impl: str,
):
    """JVP using precomputed lse (from forward)."""
    del backward_pass_impl  # Only one impl at the moment.
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_d = pl.next_power_of_2(head_dim)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if block_d <= 64 else 8)

    index_offset = index_offset_size = None
    if mask is not None:
        index_offset, index_offset_size = mask.query_iterator_indices(
            q_seq_len, kv_seq_len, block_q, block_k
        )

    b_data = bias.get_data() if bias is not None else None

    if mask is not None:
        q_id, k_id = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
    else:
        q_id = k_id = None

    if dropout_rate > 0:
        assert rng is not None, "prng_key must be provided when dropout_rate>0"
        if dropout_impl == "materialize":
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len),
                prng_key=rng,
                rate=dropout_rate,
            )
        elif dropout_impl == "counter":
            dropout_mask = None
        else:
            raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")
    else:
        dropout_mask = None

    bias_fn_grad = bias.grad if (bias is not None) else None

    kernel = functools.partial(
        mha_jvp_from_lse_kernel,
        sm_scale=sm_scale,
        head_dim=head_dim,
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        mask_fn=mask.__call__ if mask is not None else None,
        bias_fn=bias.__call__ if bias is not None else None,
        bias_fn_grad=bias_fn_grad,
        dropout_rate=dropout_rate,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
    ]

    if b_data is not None:
        b_specs = bias.get_block_spec(
            q_len=q_seq_len, kv_len=kv_seq_len, block_q=block_q, block_kv=block_k
        )
        in_specs.append(b_specs)
    else:
        in_specs.append(None)

    if q_id is not None or k_id is not None:
        q_id_spec, k_id_spec = mask.get_data_block_spec(
            q_seq_len, kv_seq_len, block_q, block_k
        )
        in_specs.append(q_id_spec)
        in_specs.append(k_id_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)

    if dropout_mask is not None:
        in_specs.append(
            pl.BlockSpec(
                (None, None, block_q, kv_seq_len), lambda i, j, k_: (j, k_, i, 0)
            )
        )
    else:
        in_specs.append(None)
    in_specs.append(pl.BlockSpec((2,), lambda *_: (0,)))

    in_specs.append(pl.BlockSpec((None, None, block_q), lambda i, j, k_: (j, k_, i)))

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

    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_specs = pl.BlockSpec(
        (None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)
    )

    tangent_out = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_jvp_from_lse",
    )(
        q,
        k,
        v,
        dq,
        dk,
        dv,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        rng,
        lse,
        index_offset,
        index_offset_size,
    )

    return tangent_out


def _flatten_optional_pytree(value):
    if value is None:
        return None, ()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    return treedef, tuple(leaves)


def _unflatten_optional_pytree(treedef, leaves):
    if treedef is None:
        return None
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _split_mha_operands(args, *, mask_num_leaves: int, bias_num_leaves: int):
    q, k, v, rng, *rest = args
    mask_leaves = rest[:mask_num_leaves]
    bias_leaves = rest[mask_num_leaves : mask_num_leaves + bias_num_leaves]
    return q, k, v, rng, mask_leaves, bias_leaves


def _mha_reference(
    q,
    k,
    v,
    *,
    mask: AttentionMask | jax.Array | None,
    bias: AttentionBias | jax.Array | None,
    rng: jax.Array | None,
    sm_scale: float,
    dropout_rate: float,
):
    batch_size, q_len, num_heads, _ = q.shape
    kv_len = k.shape[1]

    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale

    if mask is not None:
        if isinstance(mask, AttentionMask):
            if getattr(mask, "stateful", False):
                seg_q, seg_k = mask.get_data(q_seq_len=q_len, kv_seq_len=kv_len)
                if seg_q is not None and getattr(seg_q, "ndim", 0) >= 2:
                    seg_q = seg_q[:batch_size]
                if seg_k is not None and getattr(seg_k, "ndim", 0) >= 2:
                    seg_k = seg_k[:batch_size]
                dense_mask = mask.dense(
                    q_len,
                    kv_len,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    seg_q=seg_q,
                    seg_k=seg_k,
                )
            else:
                dense_mask = mask.dense(
                    q_len, kv_len, batch_size=batch_size, num_heads=num_heads
                )
        else:
            dense_mask = mask
        scores = jnp.where(dense_mask, scores, DEFAULT_MASK_VALUE)

    if bias is not None:
        if isinstance(bias, AttentionBias):
            dense_bias = bias.dense(
                q_len, kv_len, batch_size=batch_size, num_heads=num_heads
            )
        else:
            dense_bias = bias
        scores = scores + dense_bias

    weights = jax.nn.softmax(scores, axis=-1)

    if dropout_rate > 0:
        if rng is None:
            raise ValueError("dropout_rate > 0 requires a non-None rng.")
        dropout_mask = get_dropout_mask(
            (batch_size, num_heads, q_len, kv_len), prng_key=rng, rate=dropout_rate
        )
        weights = jnp.where(dropout_mask, 0, weights / (1 - dropout_rate))

    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


_mha_p = Primitive("mha")
_mha_lin_p = Primitive("mha_lin")


def _mha_bind(
    q,
    k,
    v,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    rng: jax.Array | None = None,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton_fused",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    dropout_rate: float = 0.0,
    dropout_impl: str = "materialize",
):
    """Multi-Head Attention public API (forward only in primal eval)."""
    if dropout_rate > 0 and rng is None:
        raise ValueError("dropout_rate > 0 requires a non-None rng.")
    rng = rng if rng is not None else jax.random.PRNGKey(0)

    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)

    out = _mha_p.bind(
        q,
        k,
        v,
        rng,
        *mask_leaves,
        *bias_leaves,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
        mask_num_leaves=len(mask_leaves),
        bias_num_leaves=len(bias_leaves),
    )
    return out


@functools.partial(
    jax.custom_jvp,
    nondiff_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
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
    backward_pass_impl: str = "triton_fused",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    dropout_rate: float = 0.0,
    dropout_impl: str = "materialize",
):
    return _mha_bind(
        q=q,
        k=k,
        v=v,
        mask=mask,
        bias=bias,
        rng=rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
    )


@mha.defjvp
def _mha_jvp_rule(
    mask,
    bias,
    rng,
    sm_scale,
    block_sizes,
    backward_pass_impl,
    num_warps,
    num_stages,
    grid,
    interpret,
    debug,
    dropout_rate,
    dropout_impl,
    primals,
    tangents,
):
    (q, k, v) = primals
    (dq, dk, dv) = tangents

    if dropout_rate > 0 and rng is None:
        raise ValueError("dropout_rate > 0 requires a non-None rng.")
    rng = rng if rng is not None else jax.random.PRNGKey(0)

    out, res = _mha_impl(
        q=q,
        k=k,
        v=v,
        mask=mask,
        bias=bias,
        rng=rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
        output_activations=True,
    )
    q_res, k_res, v_res, rng_res, out_res, lse_res = res

    def _tangent_or_zero(t, primal):
        if isinstance(t, ad_util.Zero):
            return jnp.zeros_like(primal)
        return t

    dq = _tangent_or_zero(dq, q)
    dk = _tangent_or_zero(dk, k)
    dv = _tangent_or_zero(dv, v)

    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)

    tangent_out = _mha_lin_p.bind(
        q_res,
        k_res,
        v_res,
        rng_res,
        out_res,
        lse_res,
        dq,
        dk,
        dv,
        *mask_leaves,
        *bias_leaves,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
        mask_num_leaves=len(mask_leaves),
        bias_num_leaves=len(bias_leaves),
    )
    return out, tangent_out




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
    dropout_impl: str,
    res,
    do,
    *,
    mask: AttentionMask | None,
    bias: AttentionBias | None,
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
    q, k, v, rng, out, lse = res

    if backward_pass_impl == "triton_fused":
        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        batch_size, q_seq_len, num_heads, head_dim = q.shape
        block_d = pl.next_power_of_2(head_dim)
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)
        block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
        block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
        block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
        block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)

        if pl.cdiv(q_seq_len, block_q_dq) != pl.cdiv(kv_seq_len, block_kv_dkv):
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
        # rng key
        pl.BlockSpec((2,), lambda *_: (0,)),
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
            q_data_spec, k_data_spec = mask.get_data_block_spec_backward_pass(
                q_seq_len,
                kv_seq_len,
                block_q,
                block_k,
                block_kv_dkv,
                block_kv_dq,
                block_q_dkv,
                block_q_dq,
            )
            in_specs[3] = q_data_spec
            in_specs[4] = k_data_spec

        if dropout_rate > 0:
            assert rng is not None
            if dropout_impl == "materialize":
                dropout_mask = get_dropout_mask(
                    (batch_size, num_heads, q_seq_len, kv_seq_len),
                    prng_key=rng,
                    rate=dropout_rate,
                )
                in_specs[6] = pl.BlockSpec(
                    (None, None, q_seq_len, kv_seq_len), lambda i, j, _: (i, j, 0, 0)
                )
            elif dropout_impl == "counter":
                dropout_mask = None
            else:
                raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")
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
                    index_map=(lambda i, _, k: (k, 0)),
                    block_shape=((None, num_kv_blocks_dq)),
                )
                q_index_offset_size_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k)), block_shape=((None,))
                )
                in_specs[-4] = q_index_offset_spec  # q_index_offset
                in_specs[-3] = q_index_offset_size_spec  # q_index_offset_size
            if kv_index_offset is not None:
                kv_index_offset_spec = pl.BlockSpec(
                    index_map=(lambda i, _, k: (k, 0)),
                    block_shape=((None, num_q_blocks_dkdv)),
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
            compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=2),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng,
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
        # Split backward into two kernels: first dKdV, then dQ.
        if backward_pass_impl not in ("triton_split", "split", "triton_2pass"):
            raise ValueError(
                f"Invalid backward pass implementation: {backward_pass_impl}"
            )

        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        batch_size, q_seq_len, num_heads, head_dim = q.shape
        kv_seq_len = k.shape[1]
        block_d = pl.next_power_of_2(head_dim)
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)
        block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
        block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
        block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
        block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)

        # Preprocess to compute delta
        delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)

        # Prepare mask data and bias tensors
        b_data = bias.get_data() if (bias is not None) else None
        q_data = k_data = None
        if isinstance(mask, AttentionMask):
            q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)

        # Dropout mask
        if dropout_rate > 0:
            assert rng is not None
            if dropout_impl == "materialize":
                dropout_mask = get_dropout_mask(
                    (batch_size, num_heads, q_seq_len, kv_seq_len),
                    prng_key=rng,
                    rate=dropout_rate,
                )
            elif dropout_impl == "counter":
                dropout_mask = None
            else:
                raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")
        else:
            dropout_mask = None

        # Optional block-sparse iterators
        q_index_offset = q_index_offset_size = kv_index_offset = (
            kv_index_offset_size
        ) = None
        if mask is not None:
            q_index_offset, q_index_offset_size = mask.query_iterator_indices(
                q_seq_len, kv_seq_len, block_q_dq, block_kv_dq
            )
            kv_index_offset, kv_index_offset_size = mask.kv_iterator_indices(
                q_seq_len, kv_seq_len, block_q_dkv, block_kv_dkv
            )

        # Compiler params
        num_warps_ = num_warps
        if num_warps_ is None:
            if (
                block_q_dkv * block_kv_dkv < 128 * 128
                or block_q_dq * block_kv_dq < 128 * 128
            ):
                num_warps_ = 4
            else:
                num_warps_ = 8

        # Bias grad function for stateless biases
        bias_fn_grad = bias.grad if (bias is not None) else None

        # Build mask data specs for backward
        if q_data is not None:
            q_data_spec, k_data_spec = mask.get_data_block_spec_backward_pass(
                q_seq_len,
                kv_seq_len,
                block_q,
                block_k,
                block_q_dkv,
                block_kv_dkv,
                block_q_dq,
                block_kv_dq,
            )
        else:
            q_data_spec = k_data_spec = None

        # Common input specs across both kernels
        common_in_specs = [
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
            # mask data
            q_data_spec,
            k_data_spec,
            # bias (dense)
            (
                None
                if b_data is None
                else bias.get_block_spec(
                    q_len=q_seq_len,
                    kv_len=kv_seq_len,
                    block_q=q_seq_len,
                    block_kv=kv_seq_len,
                )
            ),
            # dropout mask
            (
                None
                if dropout_mask is None
                else pl.BlockSpec(
                    (None, None, q_seq_len, kv_seq_len),
                    lambda i, j, _: (i, j, 0, 0),
                )
            ),
            # rng key
            pl.BlockSpec((2,), lambda *_: (0,)),
            # do, lse, delta
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]

        # 1) dKdV kernel call

        # dKdV call
        dkdv_in_specs = common_in_specs + [None, None]  # reserve index offset slots
        if kv_index_offset is not None:
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            kv_index_offset_spec = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_q_blocks_dkdv)),
            )
            kv_index_offset_size_spec = pl.BlockSpec(
                index_map=(lambda i, _, k: (k)), block_shape=((None,))
            )
            dkdv_in_specs[-2] = kv_index_offset_spec
            dkdv_in_specs[-1] = kv_index_offset_size_spec

        dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_split_dkdv,
                sm_scale=sm_scale,
                mask_fn=mask.__call__ if mask is not None else None,
                bias_fn=bias.__call__ if bias is not None else None,
                bias_fn_grad=bias_fn_grad,
                dropout_rate=dropout_rate,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_d=block_d,
                head_dim=head_dim,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                jax.ShapeDtypeStruct(v.shape, v.dtype),
            ],
            in_specs=dkdv_in_specs,
            grid=(batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv)),
            out_specs=[
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),
                ),
            ],
            name="mha_backward_split_dkdv",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=2),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng,
            do,
            lse,
            delta,
            kv_index_offset,
            kv_index_offset_size,
        )

        # dQ call
        dq_in_specs = common_in_specs + [None, None]
        if q_index_offset is not None:
            num_kv_blocks_dq = pl.cdiv(kv_seq_len, block_kv_dq)
            q_index_offset_spec = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_kv_blocks_dq)),
            )
            q_index_offset_size_spec = pl.BlockSpec(
                index_map=(lambda i, _, k: (k)), block_shape=((None,))
            )
            dq_in_specs[-2] = q_index_offset_spec
            dq_in_specs[-1] = q_index_offset_size_spec

        dq = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_split_dq,
                sm_scale=sm_scale,
                mask_fn=mask.__call__ if mask is not None else None,
                bias_fn=bias.__call__ if bias is not None else None,
                bias_fn_grad=bias_fn_grad,
                dropout_rate=dropout_rate,
                block_q_dq=block_q_dq,
                block_kv_dq=block_kv_dq,
                block_d=block_d,
                head_dim=head_dim,
            ),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            in_specs=dq_in_specs,
            grid=(batch_size, num_heads, pl.cdiv(q_seq_len, block_q_dq)),
            out_specs=pl.BlockSpec(
                (None, block_q_dq, None, head_dim),
                lambda i, j, k: (i, k, j, 0),
            ),
            name="mha_backward_split_dq",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=2),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng,
            do,
            lse,
            delta,
            q_index_offset,
            q_index_offset_size,
        )

        return dq.astype(q.dtype), dk, dv, None, None, None
    return dq.astype(q.dtype), dk, dv, None, None, None






def _mha_prim_impl(
    q,
    k,
    v,
    rng,
    *rest,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
    mask_treedef,
    bias_treedef,
    mask_num_leaves: int,
    bias_num_leaves: int,
):
    _, _, _, _, mask_leaves, bias_leaves = _split_mha_operands(
        (q, k, v, rng, *rest),
        mask_num_leaves=mask_num_leaves,
        bias_num_leaves=bias_num_leaves,
    )
    mask = _unflatten_optional_pytree(mask_treedef, mask_leaves)
    bias = _unflatten_optional_pytree(bias_treedef, bias_leaves)
    return _mha_impl(
        q=q,
        k=k,
        v=v,
        mask=mask,
        bias=bias,
        rng=rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
        output_activations=False,
    )


def _mha_lin_prim_impl(
    q,
    k,
    v,
    rng,
    out,
    lse,
    dq,
    dk,
    dv,
    *rest,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
    mask_treedef,
    bias_treedef,
    mask_num_leaves: int,
    bias_num_leaves: int,
):
    del out
    _, _, _, _, mask_leaves, bias_leaves = _split_mha_operands(
        (q, k, v, rng, *rest),
        mask_num_leaves=mask_num_leaves,
        bias_num_leaves=bias_num_leaves,
    )
    mask = _unflatten_optional_pytree(mask_treedef, mask_leaves)
    bias = _unflatten_optional_pytree(bias_treedef, bias_leaves)
    return _mha_impl_jvp_from_lse(
        q=q,
        k=k,
        v=v,
        dq=dq,
        dk=dk,
        dv=dv,
        lse=lse,
        mask=mask,
        bias=bias,
        rng=rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
    )


def _mha_lin_prim_abstract_eval(
    q_aval,
    k_aval,
    v_aval,
    rng_aval,
    out_aval,
    lse_aval,
    dq_aval,
    dk_aval,
    dv_aval,
    *rest,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
    mask_treedef,
    bias_treedef,
    mask_num_leaves: int,
    bias_num_leaves: int,
):
    del (
        k_aval,
        v_aval,
        rng_aval,
        out_aval,
        lse_aval,
        dq_aval,
        dk_aval,
        dv_aval,
        rest,
        sm_scale,
        block_sizes,
        backward_pass_impl,
        num_warps,
        num_stages,
        grid,
        interpret,
        debug,
        dropout_rate,
        dropout_impl,
        mask_treedef,
        bias_treedef,
        mask_num_leaves,
        bias_num_leaves,
    )
    return q_aval


def _mha_lin_prim_transpose(
    ct,
    q,
    k,
    v,
    rng,
    out,
    lse,
    dq,
    dk,
    dv,
    *rest,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
    mask_treedef,
    bias_treedef,
    mask_num_leaves: int,
    bias_num_leaves: int,
):
    if isinstance(ct, ad_util.Zero):
        return (None,) * (9 + mask_num_leaves + bias_num_leaves)

    _, _, _, _, mask_leaves, bias_leaves = _split_mha_operands(
        (q, k, v, rng, *rest),
        mask_num_leaves=mask_num_leaves,
        bias_num_leaves=bias_num_leaves,
    )
    mask = _unflatten_optional_pytree(mask_treedef, mask_leaves)
    bias = _unflatten_optional_pytree(bias_treedef, bias_leaves)

    res = (q, k, v, rng, out, lse)
    dq_ct, dk_ct, dv_ct, _, _, _ = _mha_backward(
        sm_scale,
        block_sizes,
        backward_pass_impl,
        num_warps,
        num_stages,
        grid,
        interpret,
        debug,
        dropout_rate,
        dropout_impl,
        res,
        ct,
        mask=mask,
        bias=bias,
    )
    grads = [None, None, None, None, None, None, dq_ct, dk_ct, dv_ct]
    grads.extend([None] * mask_num_leaves)
    grads.extend([None] * bias_num_leaves)
    return tuple(grads)


def _mha_prim_abstract_eval(
    q_aval,
    k_aval,
    v_aval,
    rng_aval,
    *rest,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
    mask_treedef,
    bias_treedef,
    mask_num_leaves: int,
    bias_num_leaves: int,
):
    del (
        k_aval,
        v_aval,
        rng_aval,
        rest,
        sm_scale,
        block_sizes,
        backward_pass_impl,
        num_warps,
        num_stages,
        grid,
        interpret,
        debug,
        dropout_rate,
        dropout_impl,
        mask_treedef,
        bias_treedef,
        mask_num_leaves,
        bias_num_leaves,
    )
    return q_aval


_mha_p.def_impl(_mha_prim_impl)
_mha_p.def_abstract_eval(_mha_prim_abstract_eval)
mlir.register_lowering(_mha_p, mlir.lower_fun(_mha_prim_impl, multiple_results=False))

_mha_lin_p.def_impl(_mha_lin_prim_impl)
_mha_lin_p.def_abstract_eval(_mha_lin_prim_abstract_eval)
mlir.register_lowering(
    _mha_lin_p, mlir.lower_fun(_mha_lin_prim_impl, multiple_results=False)
)
ad.primitive_transposes[_mha_lin_p] = _mha_lin_prim_transpose

def _mha_batching_rule(batched_args, batch_dims, **params):
    mask_treedef = params["mask_treedef"]
    bias_treedef = params["bias_treedef"]
    mask_num_leaves = params["mask_num_leaves"]
    bias_num_leaves = params["bias_num_leaves"]

    any_batched = any(d is not batching.not_mapped for d in batch_dims)
    if not any_batched:
        out = _mha_p.bind(*batched_args, **params)
        return out, batching.not_mapped

    q, k, v, rng, *mask_bias = batched_args
    q_bdim, k_bdim, v_bdim, rng_bdim, *other_bdims = batch_dims

    can_merge = (
        q_bdim == k_bdim == v_bdim == 0
        and q_bdim is not batching.not_mapped
        and all(d is batching.not_mapped for d in other_bdims)
        and (rng_bdim is batching.not_mapped or params["dropout_rate"] == 0.0)
        and params["grid"] is None
        and mask_num_leaves == 0
        and bias_num_leaves == 0
    )

    if can_merge:
        outer = q.shape[0]
        inner = q.shape[1]
        q = q.reshape((outer * inner,) + q.shape[2:])
        k = k.reshape((outer * inner,) + k.shape[2:])
        v = v.reshape((outer * inner,) + v.shape[2:])
        if rng_bdim is batching.not_mapped:
            rng_merged = rng
        else:
            rng_merged = rng[0]
        out = _mha_impl(
            q=q,
            k=k,
            v=v,
            mask=_unflatten_optional_pytree(mask_treedef, ()),
            bias=_unflatten_optional_pytree(bias_treedef, ()),
            rng=rng_merged,
            sm_scale=params["sm_scale"],
            block_sizes=params["block_sizes"],
            backward_pass_impl=params["backward_pass_impl"],
            num_warps=params["num_warps"],
            num_stages=params["num_stages"],
            grid=params["grid"],
            interpret=params["interpret"],
            debug=params["debug"],
            dropout_rate=params["dropout_rate"],
            dropout_impl=params["dropout_impl"],
            output_activations=False,
        )
        out = out.reshape((outer, inner) + out.shape[1:])
        return out, 0

    new_args = []
    in_axes = []
    for x, d in zip(batched_args, batch_dims):
        if d is batching.not_mapped:
            new_args.append(x)
            in_axes.append(None)
        else:
            new_args.append(batching.moveaxis(x, d, 0) if d != 0 else x)
            in_axes.append(0)

    def _impl(q, k, v, rng, *mask_bias_leaves):
        mask_leaves = mask_bias_leaves[:mask_num_leaves]
        bias_leaves = mask_bias_leaves[mask_num_leaves:]
        mask = _unflatten_optional_pytree(mask_treedef, mask_leaves)
        bias = _unflatten_optional_pytree(bias_treedef, bias_leaves)
        return _mha_impl(
            q=q,
            k=k,
            v=v,
            mask=mask,
            bias=bias,
            rng=rng,
            sm_scale=params["sm_scale"],
            block_sizes=params["block_sizes"],
            backward_pass_impl=params["backward_pass_impl"],
            num_warps=params["num_warps"],
            num_stages=params["num_stages"],
            grid=params["grid"],
            interpret=params["interpret"],
            debug=params["debug"],
            dropout_rate=params["dropout_rate"],
            dropout_impl=params["dropout_impl"],
            output_activations=False,
        )

    out = jax.vmap(_impl, in_axes=tuple(in_axes), out_axes=0)(*new_args)
    return out, 0


batching.primitive_batchers[_mha_p] = _mha_batching_rule


def _mha_lin_batching_rule(batched_args, batch_dims, **params):
    dq_bdim = batch_dims[6]
    dk_bdim = batch_dims[7]
    dv_bdim = batch_dims[8]

    if dq_bdim is batching.not_mapped:
        out = _mha_lin_p.bind(*batched_args, **params)
        return out, batching.not_mapped

    if not (dq_bdim == dk_bdim == dv_bdim):
        raise NotImplementedError("mha_lin: mismatched tangent batch dims")

    q, k, v, rng, out, lse, dq, dk, dv, *rest = batched_args
    dq = batching.moveaxis(dq, dq_bdim, 0)
    dk = batching.moveaxis(dk, dk_bdim, 0)
    dv = batching.moveaxis(dv, dv_bdim, 0)

    def _impl(dq_i, dk_i, dv_i):
        return _mha_lin_prim_impl(
            q, k, v, rng, out, lse, dq_i, dk_i, dv_i, *rest, **params
        )

    y = jax.vmap(_impl, in_axes=(0, 0, 0), out_axes=0)(dq, dk, dv)
    return y, 0


batching.primitive_batchers[_mha_lin_p] = _mha_lin_batching_rule
