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
import numpy as np
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

ScoreModFn = Callable[[Array, Array, Array, Array, Array], Array]
MaskModFn = Callable[[Array, Array, Array, Array], Array]
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


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
            block_q=64,
            block_k=64,
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
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    sm_scale: float,
    causal: bool,
    window_size=None,
    score_mod: ScoreModFn | None = None,
    mask_mod: MaskModFn | None = None,
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
        causal: Whether to apply causal masking.
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
    q = q_ref[...]
    q_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_q_slice,))
    )

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

        # Apply custom score modification function.
        if score_mod or mask_mod:
            # Apply the custom score modification function here
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            # boolean mask for the current qk slice
        if score_mod:
            qk = score_mod(qk, start_b, start_h, span_q, span_k)

        # Base change from e to 2.
        qk *= LOG2E

        # Apply the maskings.
        if mask_mod or causal or (segment_ids_ref is not None):
            mask = None
            # Segmend ID and causal can be absorbed into mask_mod
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            if mask_mod:
                if segment_ids_ref is not None:
                    kv_segment_ids = pl.load(segment_ids_ref, (curr_k_slice,))
                else:
                    kv_segment_ids = None
                mask = (
                    mask_mod(
                        start_b, start_h, span_q, span_k, q_segment_ids, kv_segment_ids
                    )
                    if mask is None
                    else jnp.logical_and(
                        mask,
                        mask_mod(
                            start_b,
                            start_h,
                            span_q,
                            span_k,
                            q_segment_ids,
                            kv_segment_ids,
                        ),
                    )
                )
            # Apply mask to qk.
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        # Scale logits to convert from base-2 to the natural log domain.
        # This is based on the identity: e^x = 2^(x * log2(e)).
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        # If all values are -inf, m_next will be -inf. We need to avoid this
        if mask_mod:
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
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    # This will avoid iterating over blocks that are not needed.
    lower_bound = 0
    if causal:
        # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
        upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        upper_bound = pl.cdiv(seq_len, block_k)

    if window_size:
        # Should compute lower and upper bounds for local attention.
        # Given as tupel i.e. (2,2) means each token should only attent to neightboring
        # 2 tokens.

        left_window, right_window = window_size

        # Compute the lower bound (start index for keys/values for this query)
        lower_bound = lax.max(
            lower_bound, lax.div(block_q * start_q + block_k - 1 - left_window, block_k)
        )

        # Compute the upper bound (end index for keys/values for this query)
        upper_bound = lax.min(
            upper_bound,
            lax.div(block_q * (start_q + 1) + block_k - 1 + right_window, block_k),
        )

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
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "window_size",
        "score_mod",
        "mask_mod",
        "score_mod_grad",
        "block_sizes",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    window_size: tuple[int, int] | None = None,
    score_mod: ScoreModFn | None = None,
    mask_mod: MaskModFn | None = None,
    score_mod_grad: ScoreModFn | None = None,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
    """
    Multi-Head Attention function with custom VJP and JIT compilation.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        segment_ids: Segment IDs tensor.
        sm_scale: Softmax scaling factor.
        causal: Whether to apply causal masking.
        score_mod: Custom score modification function.
        mask_mod: Custom mask modification function.
        block_sizes: Block sizes for the attention kernel.
        backward_pass_impl: Implementation for the backward pass.
        num_warps: Number of warps for Triton.
        num_stages: Number of stages for Triton.
        grid: Grid dimensions for Triton.
        interpret: Whether to interpret the kernel.
        debug: Whether to enable debugging.

    Returns:
        The output of the attention mechanism.
    """
    del backward_pass_impl, score_mod_grad
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        causal=causal,
        window_size=window_size,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    in_specs.append(
        None  # type: ignore[arg-type]
        if segment_ids is None
        else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
        ),
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, segment_ids)


def _mha_forward(
    q,
    k,
    v,
    segment_ids: jax.Array | None,
    sm_scale: float,
    causal: bool,
    window_size: tuple[int, int] | None,
    score_mod: ScoreModFn | None,
    mask_mod: MaskModFn | None,
    score_mod_grad: ScoreModFn | None,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    """
    Forward pass for the Multi-Head Attention mechanism.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        segment_ids: Segment IDs tensor.
        sm_scale: Softmax scaling factor.
        causal: Whether to apply causal masking.
        score_mod: Custom score modification function.
        mask_mod: Custom mask modification function.
        score_mod_grad: Custom score modification function.
        block_sizes: Block sizes for the attention kernel.
        backward_pass_impl: Implementation for the backward pass.
        num_warps: Number of warps for Triton.
        num_stages: Number of stages for Triton.
        grid: Grid dimensions for Triton.
        interpret: Whether to interpret the kernel.
        debug: Whether to enable debugging.

    Returns:
        The output and residuals of the forward pass.
    """
    del backward_pass_impl, score_mod_grad
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        causal=causal,
        window_size=window_size,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),  # out
        jax.ShapeDtypeStruct(
            shape=(batch_size, num_heads, q_seq_len),
            dtype=jnp.float32,  # lse
        ),
    ]
    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    in_specs.append(
        None  # type: ignore[arg-type]
        if segment_ids is None
        else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    out, lse = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        ],
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, segment_ids)
    return out, (q, k, v, segment_ids, out, lse)


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
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    window_size: tuple[int, int] | None,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    block_d: int,
    score_mod: ScoreModFn | None = None,
    mask_mod: MaskModFn | None = None,
    score_mod_grad: ScoreModFn | None = None,
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
        causal: Whether to apply causal masking.
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
    kv_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_k_slice,))
    )

    LOG2E = 1.4426950408889634  # log2(e)

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

        q = pl.load(q_ref, (curr_q_slice, slice(None)))
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if score_mod_grad or mask_mod:
            span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
            # boolean mask for the current qk slice
            if score_mod:
                qk = score_mod(qk, start_b, start_h, span_q, span_k)
            if mask_mod:
                if segment_ids_ref is not None:
                    q_segment_ids = pl.load(segment_ids_ref, (curr_q_slice,))
                else:
                    q_segment_ids = None
                qk = jnp.where(
                    mask_mod(
                        start_b, start_h, span_q, span_k, q_segment_ids, kv_segment_ids
                    ),
                    qk,
                    DEFAULT_MASK_VALUE,
                )

        if causal:
            mask = None
            if causal:
                span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        lse = pl.load(lse_ref, (curr_q_slice,))
        di = pl.load(delta_ref, (curr_q_slice,))
        do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
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

    lower_bound = lax.div(start_k * block_kv_dkv, block_q_dkv) if causal else 0

    dv, dk = lax.fori_loop(
        lower_bound, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
    )
    dv_ref[...] = dv.astype(dv_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)

    del dv, dk

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.ds(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros([block_q_dq, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    q_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_q_slice,))
    )
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

        if score_mod_grad or mask_mod:
            span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
            # boolean mask for the current qk slice
            if score_mod:
                qk = score_mod(qk, start_b, start_h, span_q, span_k)
            if mask_mod:
                if segment_ids_ref is not None:
                    q_segment_ids = pl.load(segment_ids_ref, (curr_q_slice,))
                else:
                    q_segment_ids = None
                qk = jnp.where(
                    mask_mod(
                        start_b, start_h, span_q, span_k, q_segment_ids, kv_segment_ids
                    ),
                    qk,
                    DEFAULT_MASK_VALUE,
                )

        if causal:
            mask = None
            if causal:
                span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
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

    if causal:
        upper_bound = pl.cdiv((start_q + 1) * block_q_dq, block_kv_dq)
    else:
        upper_bound = pl.cdiv(kv_seq_len, block_kv_dq)

    dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
    dq_ref[...] = dq.astype(dq_ref.dtype)


def _mha_backward(
    sm_scale: float,
    causal: bool,
    window_size: tuple[int, int] | None,
    score_mod: ScoreModFn | None,
    mask_mod: MaskModFn | None,
    score_mod_grad: ScoreModFn | None,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    res,
    do,
):
    """
    Backward pass for the Multi-Head Attention mechanism.

    Args:
        sm_scale: Softmax scaling factor.
        causal: Whether to apply causal masking.
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
    q, k, v, segment_ids, out, lse = res

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
        print("Kernel", q_seq_len, kv_seq_len)
        print(block_sizes.block_q_dq, block_sizes.block_kv_dkv)

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

        in_specs = [
            pl.BlockSpec(
                (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]
        if segment_ids is None:
            in_specs.insert(3, None)  # type: ignore[arg-type]
        else:
            in_specs.insert(3, pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0)))

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

        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                sm_scale=sm_scale,
                causal=causal,
                score_mod=score_mod,
                mask_mod=mask_mod,
                score_mod_grad=score_mod_grad,
                window_size=window_size,
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
        )(q, k, v, segment_ids, out, do, lse, delta)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, None


mha.defvjp(_mha_forward, _mha_backward)
