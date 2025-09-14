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
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from .utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    get_dropout_mask,
)
from .attention_mask_bias import (
    AttentionMaskBase,
    AttentionBiasBase,
    MaskModFn,
    ScoreModFn,
    compute_block_mask,
    compute_block_iterators,
    compute_kv_iterators,
    DenseBias,
    apply_mask,
    apply_bias,
    get_bias_grad,
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
    b_ref: jax.Array | None,  # bias tensor
    data_q_ref: jax.Array | None,  # optional mask data for q positions
    data_k_ref: jax.Array | None,  # optional mask data for k positions
    dropout_mask_ref: jax.Array | None,  # dropout mask
    index_offset_ref: jax.Array | None,  # dynamic kv iterators (optional)
    index_offset_size_ref: jax.Array | None,  # sizes for dynamic kv iterators (optional)
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    sm_scale: float,
    score_mod: ScoreModFn | None = None,
    mask_mod: MaskModFn | None = None,
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

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    # Load the current Q tile into SRAM
    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    q_data = None if data_q_ref is None else pl.load(data_q_ref, (curr_q_slice,))

    LOG2E = 1.4426950408889634  # log2(e)

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T)  # [block_q, block_k]

        # Scale this by user-provided factor (1 / sqrt(d_k) for original transformer).
        if sm_scale != 1.0:
            qk *= sm_scale

        # Seq ids for mask and bias 
        if (score_mod is not None) or (mask_mod is not None) or (b_ref is not None):
            # Apply the custom score modification function here
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
        # boolean mask for the current qk slice
        if score_mod is not None:
            q_tup = None if q_data is None else (q_data,)
            k_data = None if (data_k_ref is None) else pl.load(data_k_ref, (curr_k_slice,))
            k_tup = None if k_data is None else (k_data,)
            qk = apply_bias(score_mod, qk, start_b, start_h, span_q, span_k, q_tup, k_tup)
        if b_ref is not None:
            # bias expected shape [B|1, H|1, Q, K] mapped via BlockSpec to (b,h)
            qk = qk + pl.load(b_ref, (curr_q_slice, curr_k_slice))

        # Base change from e to 2.
        qk *= LOG2E

        # Apply the maskings.
        if mask_mod is not None:
            k_data = None if data_k_ref is None else pl.load(data_k_ref, (curr_k_slice,))
            q_tup = None if q_data is None else (q_data,)
            k_tup = None if k_data is None else (k_data,)
            mask = apply_mask(mask_mod, start_b, start_h, span_q, span_k, q_tup, k_tup)
            # Apply mask to qk.
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        # Scale logits to convert from base-2 to the natural log domain.
        # This is based on the identity: e^x = 2^(x * log2(e)).
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        # If all values are -inf, m_next will be -inf. We need to avoid this
        if mask_mod is not None:
            # NOTE: This will slightly change numerical results, but it is necessary
            # to avoid -inf if all values are masked.
            # Either way it is "more" correct than without!
            m_next = m_next == DEFAULT_MASK_VALUE
            m_next = jnp.where(m_next, 0.0, m_next)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        if dropout_rate > 0 and dropout_mask_ref is not None:
            dmask = pl.load(dropout_mask_ref, (slice(None), curr_k_slice))
            s_curr = jnp.where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    # This will avoid iterating over blocks that are not needed.
    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)

    if index_offset_size_ref is not None and index_offset_ref is not None:
        # Dynamic per-(B,H,QB) iterators over KV blocks.
        # Convert to a scalar upper bound; spec may expose a length-1 vector per tile.
        iters = jnp.sum(index_offset_size_ref[...])
        def dyn_body(iter_k, carry):
            # Load the selected KV-block index for this tile.
            # Our spec yields shape [1, nKB] per tile; index across the second dim.
            start_k = jnp.sum(pl.load(index_offset_ref, (slice(None), pl.dslice(iter_k, 1))))
            return body(start_k, carry)
        o, m_i, l_i = lax.fori_loop(0, iters, dyn_body, (o, m_i, l_i))
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
    o_ref[...] = o.astype(o_ref.dtype)


@functools.partial(
    jax.custom_vjp,
    # Keep original nondiff positions to match forward/backward signatures.
    nondiff_argnums=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
)
def mha(
    q,
    k,
    v,
    mask: AttentionMaskBase | None = None,
    sm_scale: float = 1.0,
    bias_mod: AttentionBiasBase | None = None,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    prng_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
    block_sparse: bool = True,
):
    """Multi-Head Attention using mask/bias classes only.

    Args:
        q, k, v: Input tensors [B, T{q|kv}, H, D].
        mask: AttentionMaskBase instance (optional). Supplies data/specs.
        sm_scale: Softmax scaling factor.
        bias_mod: AttentionBiasBase instance (optional), e.g., DenseBias.
        block_sizes: Block sizes for kernels.
        backward_pass_impl, num_warps, num_stages, grid, interpret, debug: execution knobs.
        prng_key, dropout_rate: dropout controls.
        block_sparse: enable block-sparse iteration based on mask.
    """
    del backward_pass_impl
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)

    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if head_dim <= 64 else 8)

    # Avoid capturing large array-bearing objects in the kernel closure.
    # For AttentionBiasBase, pass dense bias via b_ref and set score_impl=None.
    # For pure callables, we can capture the function directly.
    mask_impl = mask
    score_impl = bias_mod if not isinstance(bias_mod, DenseBias) else None

    # Optional block-sparse iterators via mask
    index_offset = index_offset_size = None
    if block_sparse and mask_impl is not None:
        B, Q, H = batch_size, q_seq_len, num_heads
        bm = compute_block_mask(
            mask_impl, batch_size=B, num_heads=H, q_len=Q, kv_len=kv_seq_len, block_q=block_q, block_k=block_k, segment_ids=None
        )
        index_offset, index_offset_size = compute_block_iterators(bm)

    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        score_mod=score_impl,
        mask_mod=mask_impl,
        dropout_rate=dropout_rate,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]

    # Bias from bias_mod.get_data
    bias = None
    if isinstance(bias_mod, AttentionBiasBase):
        bdata = bias_mod.get_data()
        if len(bdata) > 0:
            bias = bdata[0]
    if bias is not None:
        spec = bias_mod.pallas_bias_spec(block_q=block_q, kv_seq_len=kv_seq_len) if isinstance(bias_mod, AttentionBiasBase) else None
        if spec is None:
            spec = pl.BlockSpec(
                index_map=lambda i, j, k: (
                    j if bias.shape[0] != 1 else 0,
                    k if bias.shape[1] != 1 else 0,
                    i,
                    0,
                ),
                block_shape=(None, None, block_q, kv_seq_len),
            )
        in_specs.append(spec)
    else:
        in_specs.append(None)

    # Optional mask data from mask.get_data + specs
    q_data = k_data = None
    if isinstance(mask, AttentionMaskBase):
        q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
    in_specs.append(mask.pallas_q_data_spec(q_seq_len) if isinstance(mask, AttentionMaskBase) and q_data is not None else None)
    in_specs.append(mask.pallas_k_data_spec(kv_seq_len) if isinstance(mask, AttentionMaskBase) and k_data is not None else None)

    # Dropout mask
    if dropout_rate > 0:
        assert prng_key is not None
        dropout_mask = get_dropout_mask((batch_size, num_heads, q_seq_len, kv_seq_len), prng_key=prng_key, rate=dropout_rate)
        in_specs.append(pl.BlockSpec((None, None, block_q, kv_seq_len), lambda i, j, k: (j, k, i, 0)))
    else:
        dropout_mask = None
        in_specs.append(None)

    # Block-sparse iterators (always add two slots to match kernel signature)
    if index_offset is not None and index_offset_size is not None:
        # Expose, per tile (q_block, batch, head), the entire 1D vector of KV block indices
        # and a scalar size for dynamic iteration. This mirrors flash_attention.
        num_kv_blocks = pl.cdiv(kv_seq_len, block_k)
        in_specs.append(
            pl.BlockSpec(
                (None, None, 1, num_kv_blocks),
                lambda i, j, k: (j, k, i, 0),
            )
        )
        in_specs.append(
            pl.BlockSpec(
                (None, None, 1),
                lambda i, j, k: (j, k, i),
            )
        )
    else:
        in_specs.append(None)
        in_specs.append(None)

    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, bias, q_data, k_data, dropout_mask, index_offset, index_offset_size)


def _mha_forward(
    q,
    k,
    v,
    mask: AttentionMaskBase | None,
    sm_scale: float,
    bias_mod: AttentionBiasBase | None,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    prng_key: jax.Array | None = None,
    dropout_rate: float = 0.0,
    block_sparse: bool = True,
):
    """Forward for custom VJP; returns (out, residuals)."""
    del backward_pass_impl
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if head_dim <= 64 else 8)

    # Build data from classes
    mask_impl = mask
    score_impl = bias_mod if not isinstance(bias_mod, DenseBias) else None
    q_data = k_data = None
    if isinstance(mask, AttentionMaskBase):
        q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
    bias = None
    if isinstance(bias_mod, AttentionBiasBase):
        bdata = bias_mod.get_data()
        if len(bdata) > 0:
            bias = bdata[0]

    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        score_mod=score_impl,
        mask_mod=mask_impl,
        dropout_rate=dropout_rate,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32),
    ]
    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        (bias_mod.pallas_bias_spec(block_q=block_q, kv_seq_len=kv_seq_len) if (isinstance(bias_mod, AttentionBiasBase) and bias is not None) else None),
        (mask.pallas_q_data_spec(q_seq_len) if (isinstance(mask, AttentionMaskBase) and q_data is not None) else None),
        (mask.pallas_k_data_spec(kv_seq_len) if (isinstance(mask, AttentionMaskBase) and k_data is not None) else None),
    ]
    if dropout_rate > 0:
        assert prng_key is not None
        dropout_mask = get_dropout_mask((batch_size, num_heads, q_seq_len, kv_seq_len), prng_key=prng_key, rate=dropout_rate)
        in_specs.append(pl.BlockSpec((None, None, block_q, kv_seq_len), lambda i, j, k: (j, k, i, 0)))
    else:
        dropout_mask = None
        in_specs.append(None)

    # Add two None slots for optional iterators to match kernel signature
    in_specs.append(None)
    in_specs.append(None)

    out, lse = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        ],
        compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps_, num_stages=num_stages),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, bias, q_data, k_data, dropout_mask, None, None)
    return out, (q, k, v, mask, bias_mod, out, lse)


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
    data_q_ref: jax.Array | None,
    data_k_ref: jax.Array | None,
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
    sm_scale: float,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    block_d: int,
    score_mod: ScoreModFn | None = None,
    mask_mod: MaskModFn | None = None,
    score_mod_grad: ScoreModFn | None = None,
    dropout_rate: float = 0.0,
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

    v = pl.load(v_ref, (curr_k_slice, slice(None)))
    k = pl.load(k_ref, (curr_k_slice, slice(None)))
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
    kv_data = None if data_k_ref is None else pl.load(data_k_ref, (curr_k_slice,))

    LOG2E = 1.4426950408889634  # log2(e)

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

        q = pl.load(q_ref, (curr_q_slice, slice(None)))
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (score_mod is not None) or (mask_mod is not None) or (b_ref is not None):
            span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
            # boolean mask for the current qk slice
            if score_mod is not None:
                q_tup = None if (data_q_ref is None) else (pl.load(data_q_ref, (curr_q_slice,)),)
                k_tup = None if kv_data is None else (kv_data,)
                qk = apply_bias(score_mod, qk, start_b, start_h, span_q, span_k, q_tup, k_tup)
            if b_ref is not None:
                qk = qk + pl.load(b_ref, (curr_q_slice, curr_k_slice))
            if mask_mod is not None:
                q_loaded = None if data_q_ref is None else pl.load(data_q_ref, (curr_q_slice,))
                q_tup = None if q_loaded is None else (q_loaded,)
                k_tup = None if kv_data is None else (kv_data,)
                m2 = apply_mask(mask_mod, start_b, start_h, span_q, span_k, q_tup, k_tup)
                qk = jnp.where(m2, qk, DEFAULT_MASK_VALUE)
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
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (jnp.zeros_like(dp) - di[:, None])
        # Accumulate dV
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if score_mod_grad:
            # Compute the gradient of score_mod with respect to qk
            grad_score_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                score_mod_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod  # Element-wise multiplication
        dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)

        return dv, dk

    # Iterate over Q blocks for this (B,H, KB-tile)
    if kv_index_offset_ref is not None and kv_index_offset_size_ref is not None:
        iters = jnp.sum(kv_index_offset_size_ref[...])
        def dyn_q(iter_q, carry):
            start_q = jnp.sum(pl.load(kv_index_offset_ref, (slice(None), pl.dslice(iter_q, 1))))
            return inner_loop_dkdv(start_q, carry)
        dv, dk = lax.fori_loop(0, iters, dyn_q, (dv, dk))
    else:
        dv, dk = lax.fori_loop(
            0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
        )
    dv_ref[...] = dv.astype(dv_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)

    del dv, dk

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.dslice(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros([block_q_dq, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    # segment ids not used in this kernel
    lse = pl.load(lse_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pl.load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        v = pl.load(v_ref, (curr_k_slice, slice(None)))

        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (score_mod is not None) or (mask_mod is not None) or (b_ref is not None):
            span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
            # boolean mask for the current qk slice
            if score_mod is not None:
                q_loaded = None if data_q_ref is None else pl.load(data_q_ref, (curr_q_slice,))
                k_loaded = None if data_k_ref is None else pl.load(data_k_ref, (curr_k_slice,))
                q_tup = None if q_loaded is None else (q_loaded,)
                k_tup = None if k_loaded is None else (k_loaded,)
                qk = apply_bias(score_mod, qk, start_b, start_h, span_q, span_k, q_tup, k_tup)
            if b_ref is not None:
                qk = qk + pl.load(b_ref, (curr_q_slice, curr_k_slice))
            if mask_mod is not None:
                q_loaded = None if data_q_ref is None else pl.load(data_q_ref, (curr_q_slice,))
                k_loaded = None if data_k_ref is None else pl.load(data_k_ref, (curr_k_slice,))
                q_tup = None if q_loaded is None else (q_loaded,)
                k_tup = None if k_loaded is None else (k_loaded,)
                m2 = apply_mask(mask_mod, start_b, start_h, span_q, span_k, q_tup, k_tup)
                qk = jnp.where(m2, qk, DEFAULT_MASK_VALUE)
        # No built-in causal; pass as mask via mask if needed.

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + dp_dropped
        if dropout_mask_ref is not None and dropout_rate > 0:
            dmask = pl.load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            p = jnp.where(dmask, 0, p / (1 - dropout_rate))
            dp = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate)) + (jnp.zeros_like(dp) - di[:, None])
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        if score_mod_grad:
            # Compute the gradient of score_mod with respect to qk
            grad_score_mod = jnp.where(
                qk != DEFAULT_MASK_VALUE,
                score_mod_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod  # Element-wise multiplication

        dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)

        return dq

    # Iterate over KV blocks intersecting this (B,H, QB-tile)
    if q_index_offset_ref is not None and q_index_offset_size_ref is not None:
        iters = jnp.sum(q_index_offset_size_ref[...])
        def dyn_k(iter_k, dq_c):
            start_k = jnp.sum(pl.load(q_index_offset_ref, (slice(None), pl.dslice(iter_k, 1))))
            return inner_loop_dq(start_k, dq_c)
        dq = lax.fori_loop(0, iters, dyn_k, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop_dq, dq)
    dq_ref[...] = dq.astype(dq_ref.dtype)


def _mha_backward(
    mask: AttentionMaskBase | None,
    sm_scale: float,
    bias_mod: AttentionBiasBase | None,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    prng_key: jax.Array | None,
    dropout_rate: float,
    block_sparse: bool,
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
    q, k, v, mask_res, bias_res, out, lse = res
    mask = mask if mask is not None else mask_res
    bias_mod = bias_mod if bias_mod is not None else bias_res

    if backward_pass_impl == "triton":
        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        batch_size, q_seq_len, num_heads, head_dim = q.shape
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

        # Prepare bias array for backward (for AttentionBiasBase instances)
        bias = None
        if isinstance(bias_mod, AttentionBiasBase):
            bdata = bias_mod.get_data()
            if len(bdata) > 0:
                bias = bdata[0]

        # Build in_specs to match args: q, k, v, data_q, data_k, bias, dropout_mask, out, do, lse, delta,
        #                                q_index_offset, q_index_offset_size, kv_index_offset, kv_index_offset_size
        in_specs = [
            # q, k, v
            pl.BlockSpec((None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),
            # segment_ids
            # data_q
            None,
            # data_k
            None,
            # bias
            (
                None
                if bias is None
                else pl.BlockSpec(
                    index_map=lambda i, j, _: (
                        i if bias.shape[0] != 1 else 0,
                        j if bias.shape[1] != 1 else 0,
                        0,
                        0,
                    ),
                    block_shape=(None, None, q_seq_len, kv_seq_len),
                )
            ),
            # dropout mask
            None,
            # out, do, lse, delta
            pl.BlockSpec((None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]
        # Reserve 4 slots for optional dynamic iterators
        in_specs.extend([None, None, None, None])
        # Prepare optional mask data specs
        q_data, k_data = (None, None)
        if isinstance(mask, AttentionMaskBase):
            q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
        if q_data is not None:
            in_specs[3] = pl.BlockSpec((None, q_seq_len), lambda i, j, _: (i, 0))
        if k_data is not None:
            in_specs[4] = pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0))

        if dropout_rate > 0:
            assert prng_key is not None
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len), prng_key=prng_key, rate=dropout_rate
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

        # Try to provide a gradient for score_mod if available
        score_mod_grad = get_bias_grad(bias_mod)

        # Optional block-sparse iterators
        q_index_offset = q_index_offset_size = kv_index_offset = kv_index_offset_size = None
        if block_sparse and mask is not None:
            # Build block masks using the respective backward block sizes
            bm_dq = compute_block_mask(
                mask,
                batch_size=batch_size,
                num_heads=num_heads,
                q_len=q_seq_len,
                kv_len=kv_seq_len,
                block_q=block_q_dq,
                block_k=block_kv_dq,
                segment_ids=None,
            )
            bm_dkdv = compute_block_mask(
                mask,
                batch_size=batch_size,
                num_heads=num_heads,
                q_len=q_seq_len,
                kv_len=kv_seq_len,
                block_q=block_q_dkv,
                block_k=block_kv_dkv,
                segment_ids=None,
            )
            # Per-QB iterators over KV blocks for dQ
            q_index_offset, q_index_offset_size = compute_block_iterators(bm_dq)
            # Per-KB iterators over Q blocks for dK/dV
            kv_index_offset, kv_index_offset_size = compute_kv_iterators(bm_dkdv)

            num_kv_blocks_dq = pl.cdiv(kv_seq_len, block_kv_dq)
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            # Map per-tile vectors/scalars. Grid dims are (B, H, KB) and we also reuse KB as QB
            # (enforced by the check above).
            in_specs[-4] = pl.BlockSpec(
                (None, None, 1, num_kv_blocks_dq), lambda i, j, k: (i, j, k, 0)
            )  # q_index_offset
            in_specs[-3] = pl.BlockSpec(
                (None, None, 1), lambda i, j, k: (i, j, k)
            )  # q_index_offset_size
            in_specs[-2] = pl.BlockSpec(
                (None, None, 1, num_q_blocks_dkdv), lambda i, j, k: (i, j, k, 0)
            )  # kv_index_offset
            in_specs[-1] = pl.BlockSpec(
                (None, None, 1), lambda i, j, k: (i, j, k)
            )  # kv_index_offset_size

        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                sm_scale=sm_scale,
                score_mod=(None if isinstance(bias_mod, AttentionBiasBase) else bias_mod),
                mask_mod=mask,
                score_mod_grad=score_mod_grad,
                dropout_rate=dropout_rate,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_q_dq=block_q_dq,
                block_kv_dq=block_kv_dq,
                block_d=head_dim,
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
        )(q, k, v, q_data, k_data, bias, dropout_mask, out, do, lse, delta,
          q_index_offset, q_index_offset_size, kv_index_offset, kv_index_offset_size)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv


mha.defvjp(_mha_forward, _mha_backward)
