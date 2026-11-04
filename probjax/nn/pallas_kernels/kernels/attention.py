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
from jax._src import ad_util
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.interpreters import ad

from probjax.utils.guards import is_tracing
from probjax.utils.jaxutils import where

from ..attention_mask_bias import (
    AttentionBias,
    AttentionMask,
    CausalLocalWindowMask,
    CausalMask,
    LocalWindowMask,
    NoMask,
    QKVLengthMask,
)
from ..kernel_utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    ceil_div,
    dot,
    get_dot_precision,
    get_dropout_mask,
)
from ..kernel_utils.kernel_primitive import (
    KernelSpec,
    Operand,
    Output,
    make_kernel_primitive,
    shardable_kernel,
)
from ..kernel_utils.relaxed_atomic import RELAXED_ATOMICS_AVAILABLE, relaxed_atomic_add
from .attention_splitkv import splitkv_attention
from .attention_xla import use_xla_attention, xla_attention


def pallas_load(ref, idx, *, mask=None, other=None):
    return pallas_primitives.load(ref, idx, mask=mask, other=other)


def _load_mask_data(ref, seq_slice):
    """Load mask data from a ref, handling scalar-like and 4D refs.

    Mask data is stored in 4D format (B, T, 1, 1) for uniform sharding.
    BlockSpecs page the batch dim via ``None``, so the ref seen inside the
    kernel may have shape ``(T, 1, 1)`` or ``(1, 1, 1)`` (scalar-like).
    For shared (non-batched) data the ref may be ``(1, T, 1, 1)`` where the
    leading 1 is the un-paged batch placeholder.

    After stripping leading and trailing unit dims we fall back to the logic:
      - scalar / empty → load directly as scalar
      - ``(1,)`` → extract the single element as a scalar
      - ``(T,)`` → load the ``seq_slice`` window
    """
    if ref is None:
        return None

    # Strip trailing singleton dims introduced by the 4D convention.
    shape = ref.shape
    while len(shape) > 1 and shape[-1] == 1:
        shape = shape[:-1]

    # Also strip leading singletons (un-paged batch dim for shared data).
    n_leading = 0
    while n_leading < len(shape) - 1 and shape[n_leading] == 1:
        n_leading += 1
    core = shape[n_leading:]  # The meaningful part of the shape.

    if core == () or ref.shape == ():
        return pallas_load(ref, tuple(slice(None) for _ in ref.shape)).reshape(())
    if core == (1,):
        slices = tuple(pl.dslice(0, 1) for _ in ref.shape)
        return pallas_load(ref, slices).reshape(())
    # core is (T,) — load along the sequence dimension.
    # Build slices: leading dims get dslice(0,1), the seq dim gets seq_slice,
    # trailing dims get dslice(0,1).
    slices = (
        tuple(pl.dslice(0, 1) for _ in range(n_leading))
        + (seq_slice,)
        + tuple(pl.dslice(0, 1) for _ in ref.shape[n_leading + 1 :])
    )
    return pallas_load(ref, slices).reshape(-1)


def pallas_store(ref, idx, *, val, mask=None):
    return pallas_primitives.store(ref, idx, val, mask=mask)


def _dropout_mask_counter(
    rng_seed: jax.Array,
    batch_idx: jax.Array,
    head_idx: jax.Array,
    q_idx: jax.Array,
    k_idx: jax.Array,
    rate: float,
) -> jax.Array:
    """Counter-based dropout mask for a [Q, K] tile."""
    seed = rng_seed.astype(jnp.uint32)
    if seed.shape:
        x = seed[0]
        if seed.shape[0] > 1:
            x ^= seed[1] * jnp.uint32(0x9E3779B9)
            for i in range(2, seed.shape[0]):
                x ^= seed[i] * jnp.uint32(0x85EBCA6B + i)
        seed = x
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


def _rng_seed_from_key(rng: jax.Array) -> jax.Array:
    key = jax.random.key_data(rng).astype(jnp.uint32).reshape((-1,))
    seed = key[0]
    if key.shape[0] > 1:
        seed = seed ^ (key[1] * jnp.uint32(0x9E3779B9))
        for i in range(2, key.shape[0]):
            seed = seed ^ (key[i] * jnp.uint32(0x85EBCA6B + i))
    return seed


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
        num_warps_bwd: ``num_warps`` of the backward kernels (``None``: the
            call's ``num_warps``).
        num_stages_bwd: ``num_stages`` of the backward kernels (``None``: the
            call's ``num_stages``).
    """

    block_q: int = 128
    block_k: int = 128

    block_q_dkv: int = 64
    block_kv_dkv: int = 64
    block_q_dq: int = 64
    block_kv_dq: int = 64

    # Backward-only launch parameters; ``None`` uses the call's
    # ``num_warps`` / ``num_stages`` (shared with the forward kernel).
    num_warps_bwd: int | None = None
    num_stages_bwd: int | None = None

    @classmethod
    def _should_use_split_backward(
        cls,
        q_len: int,
        kv_len: int,
        block_q_dq: int,
        block_kv_dkv: int,
        min_efficient_tile_size: int = 16,
        max_asymmetry_ratio: float = 4.0,
    ) -> bool:
        """
        Heuristic to determine if split backward should be used instead of fused.

        Args:
            q_len: Query sequence length
            kv_len: Key-Value sequence length
            block_q_dq: Desired Q block size for dQ pass
            block_kv_dkv: Desired KV block size for dKV pass
            min_efficient_tile_size: Minimum tile size considered efficient
            max_asymmetry_ratio: Max ratio before considering sequences asymmetric

        Returns:
            True if split backward should be used for efficiency
        """

        # Calculate what tile counts would be with current block sizes
        nq = max(ceil_div(q_len, max(block_q_dq, 1)), 1)
        nkv = max(ceil_div(kv_len, max(block_kv_dkv, 1)), 1)

        # If tile counts already match, fused is fine
        if nq == nkv:
            return False

        # Check sequence length asymmetry
        ratio = max(q_len, kv_len) / max(min(q_len, kv_len), 1)
        if ratio > max_asymmetry_ratio:
            # For highly asymmetric sequences, check if fused would create tiny tiles
            n = max(nq, nkv)
            forced_q_tile_size = max((q_len + n - 1) // n, 1)
            forced_kv_tile_size = max((kv_len + n - 1) // n, 1)

            # Use split if either tile would become too small
            if (
                forced_q_tile_size < min_efficient_tile_size
                or forced_kv_tile_size < min_efficient_tile_size
            ):
                return True

        return False

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
        backward_pass_impl: str = "auto",
    ) -> BlockSizes:
        """Return block sizes with optimal backward pass implementation selection.

        For backward_pass_impl="auto", automatically chooses between fused and split
        backward based on sequence length asymmetry and tile efficiency:
        - Self-attention (q_len ≈ kv_len): Uses fused backward for efficiency
        - Cross-attention (q_len << kv_len or q_len >> kv_len): Uses split backward
          to avoid tiny tile sizes that hurt performance

        For backward_pass_impl="triton_fused", forces fused backward and adjusts
        block sizes to satisfy: ceil_div(q_len, block_q_dq)
        == ceil_div(kv_len, block_kv_dkv).
        This may create inefficient tiny tiles for asymmetric sequence lengths.

        For backward_pass_impl="triton_split", uses split backward with independent
        tile sizes optimized for each pass.
        """

        # Start from requested specs
        bq = block_q
        bk = block_k
        bq_dkv = block_q_dkv
        bkv_dkv = block_kv_dkv
        bq_dq = block_q_dq
        bkv_dq = block_kv_dq

        # Auto-select backward implementation based on efficiency heuristics
        if backward_pass_impl == "auto":
            if cls._should_use_split_backward(q_len, kv_len, bq_dq, bkv_dkv):
                # Use split backward - no tile count matching needed
                # Keep original block sizes for optimal efficiency
                pass  # bq_dq and bkv_dkv remain unchanged
            else:
                # Use fused backward - enforce tile count matching
                backward_pass_impl = "triton_fused"

        # Handle backward compatibility: "triton_fused" was the old default
        if backward_pass_impl in ["triton_fused", "fused"]:
            nq = max(ceil_div(q_len, max(bq_dq, 1)), 1)
            nkv = max(ceil_div(kv_len, max(bkv_dkv, 1)), 1)
            if nq != nkv:
                # Target a common number of tiles. Choose the larger to avoid
                # decreasing parallelism unnecessarily.
                n = max(nq, nkv)
                # Derive block sizes that yield exactly `n` tiles.
                # Using ceil_div ensures ceil_div(len, block) == n.
                bq_dq = max((q_len + n - 1) // n, 1)
                bkv_dkv = max((kv_len + n - 1) // n, 1)

        # For split backward ("triton_split", "split", "triton_2pass"),
        # keep independent block sizes - no matching required

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

    @classmethod
    def get_recommended_backward_impl(
        cls, q_len: int, kv_len: int, block_q_dq: int = 64, block_kv_dkv: int = 64
    ) -> str:
        """
        Returns the recommended backward implementation for given sequence lengths.

        Args:
            q_len: Query sequence length
            kv_len: Key-Value sequence length
            block_q_dq: Desired Q block size for dQ pass (default: 64)
            block_kv_dkv: Desired KV block size for dKV pass (default: 64)

        Returns:
            "triton_split" for asymmetric lengths, "triton_fused" for symmetric lengths
        """
        if cls._should_use_split_backward(q_len, kv_len, block_q_dq, block_kv_dkv):
            return "triton_split"
        else:
            return "triton_fused"

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
            self.num_warps_bwd,
            self.num_stages_bwd,
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
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    id_q = _load_mask_data(id_q_ref, curr_q_slice)
    span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634  # log2(e)

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        if index_offset_ref is not None:
            # We retrieve the dynamic indices for the current block if offset is
            # provided.
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        qk = dot(q, k.T, precision=precision)  # [block_q, block_k]

        # Scale this by user-provided factor (1 / sqrt(d_k) for original transformer).
        if sm_scale != 1.0:
            qk *= sm_scale

        # Seq ids for mask and bias
        span_k = start_k * block_k + jnp.arange(block_k)
        # Apply bias to qk: dense tensor via b_ref; function via bias_fn
        if bias_fn is not None:
            if b_ref is not None:
                b_chunk = pallas_load(b_ref, (slice(None), curr_k_slice))
            else:
                b_chunk = None
            qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
        # boolean mask for the current qk slice
        if mask_fn is not None:
            if id_k_ref is not None:
                id_k = _load_mask_data(id_k_ref, curr_k_slice)
            elif id_q is not None:
                # Reuse id_q ref for k-side (e.g. SeqLenMask uses same lengths).
                id_k = _load_mask_data(id_q_ref, curr_k_slice)
            else:
                id_k = None
            mask = mask_fn(span_q, span_k, id_q, id_k)
            qk = where(mask, qk, DEFAULT_MASK_VALUE)

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
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=jnp.nan)
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            s_curr = where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = dot(s_curr.astype(v.dtype), v, precision=precision)
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
    l_i = where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log2(l_i)
    # Write output to dram.
    pallas_store(
        o_ref, (slice(None), slice(None)), val=o.astype(o_ref.dtype), mask=d_mask
    )


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
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    dq = pallas_load(dq_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    id_q = _load_mask_data(id_q_ref, curr_q_slice)
    span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634  # log2(e)

    # BlockSpec already selects this query tile. Index relative to the tile.
    lse = pallas_load(lse_ref, (slice(None),))
    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, carry):
        do_acc, out_acc, row_acc = carry
        if index_offset_ref is not None:
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pallas_load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pallas_load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = dot(q, k.T, precision=precision)
        dqk = dot(dq, k.T, precision=precision) + dot(q, dk.T, precision=precision)
        if sm_scale != 1.0:
            qk *= sm_scale
            dqk *= sm_scale
        qk_pre_mod = qk

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
            qk = where(mask, qk, DEFAULT_MASK_VALUE)
            dqk = where(mask, dqk, 0.0)

        if bias_fn_grad is not None:
            grad_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            dqk = dqk * grad_mod

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            p_drop = where(dmask, 0, p / (1 - dropout_rate))
        else:
            p_drop = p

        # Softmax normalization couples ALL key tiles. Accumulate its global
        # directional mean, then subtract mean * output after the key loop.
        row_acc = row_acc + jnp.sum(dqk * p, axis=-1)
        do_acc = do_acc + dot((p_drop * dqk).astype(v.dtype), v, precision=precision)
        do_acc = do_acc + dot(p_drop.astype(v.dtype), dv, precision=precision)
        out_acc = out_acc + dot(p_drop.astype(v.dtype), v, precision=precision)
        return do_acc, out_acc, row_acc

    carry = (do, jnp.zeros_like(do), jnp.zeros((block_q,), dtype=jnp.float32))
    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)
    if index_offset_size_ref is not None:
        iters = index_offset_size_ref[...]
        carry = lax.fori_loop(lower_bound, iters, body_jvp, carry)
    else:
        carry = lax.fori_loop(lower_bound, upper_bound, body_jvp, carry)

    do, out, row_sum = carry
    do = do - row_sum[:, None] * out
    pallas_store(
        do_ref, (slice(None), slice(None)), val=do.astype(do_ref.dtype), mask=d_mask
    )


def mha_forward_jvp_kernel(
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
    index_offset_ref: jax.Array | None,
    index_offset_size_ref: jax.Array | None,
    o_ref: Any,
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
    """Fused forward + JVP kernel."""
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    d_mask = jnp.arange(block_d)[None] < head_dim
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    dq = pallas_load(dq_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    id_q = _load_mask_data(id_q_ref, curr_q_slice)
    span_q = start_q * block_q + jnp.arange(block_q)
    LOG2E = 1.4426950408889634  # log2(e)

    m_i = jnp.full((block_q,), NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_fwd(start_k, carry):
        if index_offset_ref is not None:
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
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
            qk = where(mask, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev

        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=jnp.nan)
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            s_curr = where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = dot(s_curr.astype(v.dtype), v, precision=precision)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)
    if index_offset_size_ref is not None:
        iters = index_offset_size_ref[...]
        o, m_i, l_i = lax.fori_loop(lower_bound, iters, body_fwd, (o, m_i, l_i))
    else:
        o, m_i, l_i = lax.fori_loop(lower_bound, upper_bound, body_fwd, (o, m_i, l_i))

    l_i = where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]
    lse = m_i + jnp.log2(l_i)

    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, do_acc):
        if index_offset_ref is not None:
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pallas_load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pallas_load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = dot(q, k.T, precision=precision)
        dqk = dot(dq, k.T, precision=precision) + dot(q, dk.T, precision=precision)
        if sm_scale != 1.0:
            qk *= sm_scale
            dqk *= sm_scale
        qk_pre_mod = qk

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
            qk = where(mask, qk, DEFAULT_MASK_VALUE)
            dqk = where(mask, dqk, 0.0)

        if bias_fn_grad is not None:
            grad_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            dqk = dqk * grad_mod

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        row_sum = jnp.sum(dqk * p, axis=-1)
        dP = p * (dqk - row_sum[:, None])
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            p_drop = where(dmask, 0, p / (1 - dropout_rate))
            dP = where(dmask, 0, dP / (1 - dropout_rate))
        else:
            p_drop = p

        do_acc = do_acc + dot(dP.astype(v.dtype), v, precision=precision)
        do_acc = do_acc + dot(p_drop.astype(v.dtype), dv, precision=precision)
        return do_acc

    if index_offset_size_ref is not None:
        iters = index_offset_size_ref[...]
        do = lax.fori_loop(lower_bound, iters, body_jvp, do)
    else:
        do = lax.fori_loop(lower_bound, upper_bound, body_jvp, do)

    pallas_store(
        o_ref, (slice(None), slice(None)), val=o.astype(o_ref.dtype), mask=d_mask
    )
    pallas_store(
        do_ref, (slice(None), slice(None)), val=do.astype(do_ref.dtype), mask=d_mask
    )


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

    v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
    k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
    if id_k_ref is not None:
        id_k = _load_mask_data(id_k_ref, curr_k_slice)
    elif id_q_ref is not None:
        id_k = _load_mask_data(id_q_ref, curr_k_slice)
    else:
        id_k = None

    LOG2E = 1.4426950408889634  # log2(e)

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)

        q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
        qk = dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            # boolean mask for the current qk slice
            if bias_fn is not None:
                b_chunk = (
                    pallas_load(b_ref, (curr_q_slice, curr_k_slice))
                    if b_ref is not None
                    else None
                )
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)

            if mask_fn is not None:
                id_q = _load_mask_data(id_q_ref, curr_q_slice)
                mask = mask_fn(span_q, span_k, id_q, id_k)
                qk = where(mask, qk, DEFAULT_MASK_VALUE)
        qk *= LOG2E
        lse = pallas_load(lse_ref, (curr_q_slice,))
        di = pallas_load(delta_ref, (curr_q_slice,))
        do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            p_drop = where(dmask, 0, p / (1 - dropout_rate))
            g_drop = where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            p_drop = p
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        dv = dv + dot(p_drop.astype(do.dtype).T, do)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad:
            grad_score_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod
        dk = dk + dot(ds.astype(q_ref.dtype).T, q)

        return dv, dk

    # Iterate over Q blocks for this (B,H, KB-tile)
    if kv_index_offset_ref is not None and kv_index_offset_size_ref is not None:
        iters = kv_index_offset_size_ref[...]

        def dyn_q(iter_q, carry):
            start_q = jnp.sum(pallas_load(kv_index_offset_ref, (pl.dslice(iter_q, 1),)))
            return inner_loop_dkdv(start_q, carry)

        dv, dk = lax.fori_loop(0, iters, dyn_q, (dv, dk))
    else:
        dv, dk = lax.fori_loop(
            0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
        )

    pallas_store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=mask_d
    )
    pallas_store(
        dk_ref, (slice(None), slice(None)), val=dk.astype(dk_ref.dtype), mask=mask_d
    )

    del dv, dk

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.dslice(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros([block_q_dq, block_d], dtype=jnp.float32)

    q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
    # segment ids not used in this kernel
    lse = pallas_load(lse_ref, (curr_q_slice,))
    do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pallas_load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)

        qk = dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        # Needed by the mask/bias path AND by counter-based dropout below, so
        # it must be defined unconditionally.
        span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            # boolean mask for the current qk slice
            if bias_fn is not None:
                b_chunk = (
                    pallas_load(b_ref, (curr_q_slice, curr_k_slice))
                    if b_ref is not None
                    else None
                )
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)

            if mask_fn is not None:
                id_q = _load_mask_data(id_q_ref, curr_q_slice)
                if id_k_ref is not None:
                    id_k = _load_mask_data(id_k_ref, curr_k_slice)
                elif id_q_ref is not None:
                    id_k = _load_mask_data(id_q_ref, curr_k_slice)
                else:
                    id_k = None

                mask = mask_fn(span_q, span_k, id_q, id_k)
                qk = where(mask, qk, DEFAULT_MASK_VALUE)
        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            g_drop = where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        if bias_fn_grad:
            grad_score_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre_mod, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_score_mod

        dq = dq + dot(ds.astype(k.dtype), k).astype(dq.dtype)

        return dq

    # Iterate over KV blocks intersecting this (B,H, QB-tile)
    if q_index_offset_ref is not None and q_index_offset_size_ref is not None:
        iters = q_index_offset_size_ref[...]

        def dyn_k(iter_k, dq_c):
            start_k = jnp.sum(pallas_load(q_index_offset_ref, (pl.dslice(iter_k, 1),)))
            return inner_loop_dq(start_k, dq_c)

        dq = lax.fori_loop(0, iters, dyn_k, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop_dq, dq)

    pallas_store(
        dq_ref, (slice(None), slice(None)), val=dq.astype(dq_ref.dtype), mask=mask_d
    )


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
    dq_acc_ref: jax.Array | None = None,
    dq_atomic: bool = True,
):
    """Computes dK and dV in a dedicated pass iterating over Q blocks.

    With ``dq_acc_ref`` (a float32 ``[Lq, D]`` buffer, zero-initialized) the
    kernel also adds each tile's ``dS @ K`` to it atomically: the single-pass
    (FlashAttention-2 style) backward, which computes ``Q K^T`` and
    ``dO V^T`` once per tile instead of once per pass.
    """
    q_seq_len = q_ref.shape[0]
    block_mask = jnp.arange(block_d)[None] < head_dim
    start_b = pl.program_id(0)
    start_h = pl.program_id(1)
    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)

    dv = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_kv_dkv, block_d], dtype=jnp.float32)

    v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
    k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)

    if id_k_ref is not None:
        id_k = _load_mask_data(id_k_ref, curr_k_slice)
    elif id_q_ref is not None:
        id_k = _load_mask_data(id_q_ref, curr_k_slice)
    else:
        id_k = None

    LOG2E = 1.4426950408889634

    num_q_blocks = pl.cdiv(q_seq_len, block_q_dkv)

    def inner_loop(start_q, carry):
        if dq_acc_ref is not None:
            # Stagger the Q order across KV-block programs so concurrent
            # programs add dQ into different tiles (less atomic contention).
            if index_offset_size_ref is not None:
                start_q = (start_q + start_k) % index_offset_size_ref[...]
            else:
                start_q = (start_q + start_k) % num_q_blocks
        if index_offset_ref is not None:
            start_q = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_q, 1),)))
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
        dv_acc, dk_acc = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)
        q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=block_mask, other=0.0)
        qk = dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre = qk
        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            if b_ref is not None and bias_fn is not None:
                b_chunk = pallas_load(b_ref, (curr_q_slice, curr_k_slice))
            else:
                b_chunk = None
            if bias_fn is not None:
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
            if mask_fn is not None:
                id_q = _load_mask_data(id_q_ref, curr_q_slice)
                m = mask_fn(span_q, span_k, id_q, id_k)
                qk = where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        lse = pallas_load(lse_ref, (curr_q_slice,))
        di = pallas_load(delta_ref, (curr_q_slice,))
        do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # p_drop for dV accumulation (forward consistency)
            p_drop = where(dmask, 0, p / (1 - dropout_rate))
            # g_drop for softmax Jacobian computation
            g_drop = where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            # No dropout case
            p_drop = p
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        # Accumulate dV: use dropout-scaled probabilities
        dv_acc = dv_acc + dot(p_drop.astype(do.dtype).T, do)
        # Softmax Jacobian: use ORIGINAL probabilities (not dropout-scaled)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad is not None:
            grad_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_mod
        ds_c = ds.astype(q_ref.dtype)
        dk_acc = dk_acc + dot(ds_c.T, q)
        if dq_acc_ref is not None:
            row_mask = (span_q < q_seq_len)[:, None] & block_mask
            dq_tile = dot(ds_c, k).astype(jnp.float32)
            dq_idx = (curr_q_slice, slice(None))
            if dq_atomic:
                relaxed_atomic_add(dq_acc_ref, dq_idx, dq_tile, mask=row_mask)
            else:
                # Interpret mode runs grid programs sequentially and has no
                # masked atomics: a read-modify-write is equivalent there.
                prev = pallas_load(dq_acc_ref, dq_idx, mask=row_mask, other=0.0)
                pallas_store(dq_acc_ref, dq_idx, val=prev + dq_tile, mask=row_mask)
        return dv_acc, dk_acc

    if index_offset_size_ref is not None:
        dv, dk = lax.fori_loop(0, index_offset_size_ref[...], inner_loop, (dv, dk))
    else:
        dv, dk = lax.fori_loop(0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop, (dv, dk))

    pallas_store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=block_mask
    )
    pallas_store(
        dk_ref, (slice(None), slice(None)), val=dk.astype(dk_ref.dtype), mask=block_mask
    )


def mha_backward_kernel_onepass(*refs, **kwargs):
    """Single-pass backward: dK/dV per KV block, dQ by atomic accumulation.

    Same inputs as :func:`mha_backward_kernel_split_dkdv` plus a trailing
    zero-initialized float32 dQ buffer aliased to the third output.
    """
    *inputs, _dq_in, dk_ref, dv_ref, dq_ref = refs
    return mha_backward_kernel_split_dkdv(
        *inputs, dk_ref, dv_ref, dq_acc_ref=dq_ref, **kwargs
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

    q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=block_mask, other=0.0)
    lse = pallas_load(lse_ref, (curr_q_slice,))
    do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pallas_load(delta_ref, (curr_q_slice,))

    LOG2E = 1.4426950408889634

    def inner_loop(start_k, dq_c):
        if index_offset_ref is not None:
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
        qk = dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre = qk
        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            if b_ref is not None and bias_fn is not None:
                b_chunk = pallas_load(b_ref, (curr_q_slice, curr_k_slice))
            else:
                b_chunk = None
            if bias_fn is not None:
                qk = bias_fn(qk, start_h, span_q, span_k, data=b_chunk)
            if mask_fn is not None:
                id_q = _load_mask_data(id_q_ref, curr_q_slice)
                if id_k_ref is not None:
                    id_k = _load_mask_data(id_k_ref, curr_k_slice)
                elif id_q_ref is not None:
                    id_k = _load_mask_data(id_q_ref, curr_k_slice)
                else:
                    id_k = None
                m = mask_fn(span_q, span_k, id_q, id_k)
                qk = where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # g_drop for softmax Jacobian computation
            g_drop = where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            # No dropout case
            dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        # Softmax Jacobian: use ORIGINAL probabilities (not dropout-scaled)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        if bias_fn_grad is not None:
            grad_mod = where(
                qk != DEFAULT_MASK_VALUE,
                bias_fn_grad(qk_pre, start_b, start_h, span_q, span_k),
                0.0,
            )
            ds = ds * grad_mod
        dq_c = dq_c + dot(ds.astype(k.dtype), k).astype(dq_c.dtype)
        return dq_c

    if index_offset_size_ref is not None:
        dq = lax.fori_loop(0, index_offset_size_ref[...], inner_loop, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop, dq)

    pallas_store(
        dq_ref, (slice(None), slice(None)), val=dq.astype(dq_ref.dtype), mask=block_mask
    )


def _extract_fwd_data(
    mask: AttentionMask | None,
    bias: AttentionBias | None,
    rng: jax.Array | None,
    *,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_q: int,
    block_k: int,
    dropout_rate: float,
    dropout_impl: str,
):
    """Extract all arrays and BlockSpecs from mask/bias objects for forward.

    Returns a dict of arrays and a dict of BlockSpecs/callables.
    Arrays must be passed through the CP boundary; specs/callables are safe
    for closures.
    """
    # Mask data arrays
    q_id = k_id = None
    if mask is not None:
        q_id, k_id = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)

    # Block-sparse iterators
    index_offset = index_offset_size = None
    if mask is not None:
        index_offset, index_offset_size = mask.query_iterator_indices(
            q_seq_len, kv_seq_len, block_q, block_k
        )

    # Bias data
    b_data = bias.get_data() if bias is not None else None

    # Dropout mask
    dropout_mask = None
    if dropout_rate > 0:
        assert rng is not None, "prng_key must be provided when dropout_rate>0"
        if dropout_impl == "materialize":
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len),
                prng_key=rng,
                rate=dropout_rate,
            )
        elif dropout_impl == "counter":
            pass
        else:
            raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")

    # BlockSpecs (not arrays — safe for closures)
    b_spec = None
    if bias is not None and b_data is not None:
        b_spec = bias.get_block_spec(
            q_len=q_seq_len, kv_len=kv_seq_len, block_q=block_q, block_kv=block_k
        )
    q_id_spec = k_id_spec = None
    if mask is not None and (q_id is not None or k_id is not None):
        q_id_spec, k_id_spec = mask.get_data_block_spec(
            q_seq_len, kv_seq_len, block_q, block_k
        )

    # Callables
    mask_fn = mask.__call__ if mask is not None else None
    bias_fn = bias.__call__ if bias is not None else None

    arrays = dict(
        b_data=b_data,
        q_id=q_id,
        k_id=k_id,
        index_offset=index_offset,
        index_offset_size=index_offset_size,
        dropout_mask=dropout_mask,
    )
    specs = dict(
        b_spec=b_spec,
        q_id_spec=q_id_spec,
        k_id_spec=k_id_spec,
        mask_fn=mask_fn,
        bias_fn=bias_fn,
    )
    return arrays, specs


def _resolve_attention_num_stages(num_stages: int, *, has_dense_bias: bool) -> int:
    """Clamp pipeline depth for dense-bias kernels to stay within shared memory."""

    if has_dense_bias:
        return 1
    return num_stages


# ---------------------------------------------------------------------------
# Shared-memory-aware tuning for consumer GPUs.
# ---------------------------------------------------------------------------
# Defaults (block_q=128, block_k=128, num_stages=2) target A100 (164KB) / H100
# (228KB). Consumer cards (e.g. Ampere/Ada, ~101KB opt-in) raise
# RESOURCE_EXHAUSTED for large sequence lengths. Query the real limit via
# libcuda when available and shrink tiles proactively (helps jit, where the
# failure surfaces at runtime, after tracing) plus a reactive retry for eager
# execution (exact, no estimation needed).

_SMEM_LIMIT_CACHE: int | None | str = "uninitialized"
# Above this limit we assume a datacenter GPU and keep requested configs.
_DATACENTER_SMEM_THRESHOLD = 150 * 1024
_MIN_TILE = 16
# Bytes kept free below the reported limit (Triton bookkeeping such as
# barriers; the estimates below reproduce the requested bytes of every
# RESOURCE_EXHAUSTED failure in the sm_86 tile sweep exactly or over-estimate).
_SMEM_RESERVE = 2048

# Default tiles on small-SMEM GPUs, keyed by (dtype bytes, padded head dim):
# (block_q, block_k, bwd_outer, bwd_inner, num_stages, num_warps or None for
# the head-dim based default). ``bwd_outer`` is the
# resident tile of the fused backward (``block_kv_dkv`` = ``block_q_dq``),
# ``bwd_inner`` the streamed one (``block_q_dkv`` = ``block_kv_dq``). Measured
# on an RTX 3090 with docs/scripts/benchmark_gpu_attention_tiles.py
# (docs/benchmarks/kernel_attn_gpu_tiles_before.json); only used when the
# caller kept the default block sizes / num_stages / num_warps.
_SMALL_SMEM_TUNED: dict[tuple[int, int], tuple[int, int, int, int, int, int | None]] = {
    (2, 32): (64, 64, 64, 64, 2, None),
    (2, 64): (64, 64, 64, 64, 2, None),
    (2, 128): (128, 64, 64, 32, 2, None),
    (4, 32): (128, 64, 64, 64, 2, None),
    (4, 64): (64, 32, 32, 32, 1, None),
    (4, 128): (64, 32, 64, 32, 1, 4),
}
_DEFAULT_BLOCK_SIZES = (128, 128, 64, 64, 64, 64)
_ONEPASS_BACKWARD_IMPLS = ("triton_onepass", "onepass")
# Single-pass backward defaults on small-SMEM GPUs, keyed by (dtype bytes,
# padded head_dim): (block_q_dkv, block_kv_dkv, num_stages_bwd,
# num_warps_bwd, min_kv_len, max_kv_len); ``None`` bounds are open. Chosen
# on an RTX 3090 for configurations that beat the two-pass default at every
# measured batch x heads (32, 64, 128) and length in range; the atomic dQ
# traffic grows with the grid, so wins at small B*H do not carry over to
# large ones; below 1024 keys bf16 d32 was neutral to 3% slower (and an
# 11% slower DiffusionTransformer step at 256 tokens), hence its bound.
# fp32 d128 is not listed: its single-pass backward wins only
# for small B*H (0.87-0.98x) and loses for B*H >= 64 (up to 1.10x).
_SMALL_SMEM_ONEPASS_TUNED: dict[
    tuple[int, int], tuple[int, int, int, int, int | None, int | None]
] = {
    (2, 32): (64, 128, 1, 4, 1024, None),
    (2, 64): (32, 128, 1, 8, 2048, None),
    (2, 128): (64, 64, 2, 8, 2048, None),
    (4, 32): (64, 64, 1, 4, None, None),
    (4, 64): (32, 64, 1, 4, 1024, None),
}
# Tiles for an explicit ``backward_pass_impl="triton_onepass"`` where the
# automatic choice stays two-pass (fastest measured fp32 d128 tiles).
_SMALL_SMEM_ONEPASS_EXPLICIT: dict[
    tuple[int, int], tuple[int, int, int, int, int | None, int | None]
] = {
    (4, 128): (32, 32, 1, 4, None, None),
}
_DEFAULT_NUM_STAGES = 2


def _query_triton_smem_limit() -> int | None:
    """Return opt-in shared-memory-per-block limit, or None if unknown."""
    global _SMEM_LIMIT_CACHE
    if _SMEM_LIMIT_CACHE != "uninitialized":
        return _SMEM_LIMIT_CACHE  # type: ignore[return-value]
    limit: int | None = None
    try:
        import ctypes

        for _name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(_name)
                break
            except OSError:
                lib = None  # type: ignore[assignment]
        else:
            lib = None
        if lib is not None and lib.cuInit(0) == 0:
            dev = ctypes.c_int()
            # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
            # MAX_SHARED_MEMORY_PER_BLOCK = 8 (fallback).
            if lib.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                val = ctypes.c_int()
                if (
                    lib.cuDeviceGetAttribute(ctypes.byref(val), 97, dev.value) == 0
                    or lib.cuDeviceGetAttribute(ctypes.byref(val), 8, dev.value) == 0
                ):
                    limit = int(val.value)
    except Exception:
        limit = None
    _SMEM_LIMIT_CACHE = limit
    return limit


def _is_smem_resource_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "shared memory" not in msg:
        return False
    return (
        "exceeded" in msg
        or "limit" in msg
        or "resource_exhausted" in msg
        or "resource exhausted" in msg
    )


def _next_pow2(n: int) -> int:
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _estimate_fwd_smem_bytes(
    bq: int, bk: int, bd: int, stages: int, dtype_bytes: int
) -> int:
    """Shared memory Triton allocates for the forward kernel (upper bound).

    Two allocation patterns were observed on sm_86 (RESOURCE_EXHAUSTED reports
    of the tile sweep); we take the larger: multi-buffered K and V tiles
    (``stages`` copies each), or the Q tile plus ``2 * (stages - 1)`` (at least
    one) K/V tiles. Both add the ``bq x bk`` probability tile staged for the
    second dot. Matches every requested size measured and never under-estimates.
    """
    kv_buffers = 2 * max(stages, 1) * bk * bd
    q_resident = bq * bd + max(1, 2 * (stages - 1)) * bk * bd
    return (max(kv_buffers, q_resident) + bq * bk) * dtype_bytes


def _estimate_bwd_smem_bytes(
    bq_dkv: int, bkv_dkv: int, bq_dq: int, bkv_dq: int, bd: int, dtype_bytes: int
) -> int:
    """Shared memory of the fused backward kernel (upper bound, see above).

    Resident (``outer``) and streamed (``inner``) tiles of two operands each;
    for 16-bit inputs one score tile is additionally staged for the cast that
    feeds the second dot (fp32 needs no cast). Pipelining adds only ~1 KB.
    """
    outer = max(bkv_dkv, bq_dq)
    inner = max(bq_dkv, bkv_dq)
    scores = max(bq_dkv * bkv_dkv, bq_dq * bkv_dq) if dtype_bytes < 4 else 0
    return (2 * (outer + inner) * bd + scores) * dtype_bytes


def _estimate_onepass_bwd_smem_bytes(
    bq_dkv: int, bkv_dkv: int, bd: int, dtype_bytes: int
) -> int:
    """Shared memory of the single-pass backward kernel.

    The two-pass dK/dV footprint (resident K/V, streamed Q/dO, the cast score
    tile for 16-bit inputs) plus one ``[block_kv, D]`` tile in the input
    dtype for the second layout of K that the ``dS @ K`` product needs.
    Matches the measured requirement of the configurations that overflowed
    on an RTX 3090 (131072 bytes for fp32 d64 64x128 and fp32 d128 32x64) and
    admits every configuration that compiled there.
    """
    scores = bq_dkv * bkv_dkv * dtype_bytes if dtype_bytes < 4 else 0
    return (2 * (bq_dkv + bkv_dkv) + bkv_dkv) * bd * dtype_bytes + scores


def _shrink_fwd_config(bq: int, bk: int, stages: int) -> tuple[int, int, int] | None:
    """One shrink step: prefer dropping pipelining, then halving tiles."""
    if stages > 1:
        return (bq, bk, 1)
    if bk >= bq and bk > _MIN_TILE:
        return (bq, max(_MIN_TILE, bk // 2), stages)
    if bq > _MIN_TILE:
        return (max(_MIN_TILE, bq // 2), bk, stages)
    if bk > _MIN_TILE:
        return (bq, max(_MIN_TILE, bk // 2), stages)
    return None


def _adjust_fwd_config_for_smem(
    bq: int,
    bk: int,
    bd: int,
    stages: int,
    dtype_bytes: int,
    limit: int | None,
) -> tuple[int, int, int]:
    if limit is None or limit >= _DATACENTER_SMEM_THRESHOLD:
        return (bq, bk, stages)
    # Clamp to sequence-independent safe starting point; loop guards jit path.
    while True:
        est = _estimate_fwd_smem_bytes(bq, bk, bd, stages, dtype_bytes)
        if est <= limit - _SMEM_RESERVE:
            return (bq, bk, stages)
        nxt = _shrink_fwd_config(bq, bk, stages)
        if nxt is None:
            return (bq, bk, stages)
        bq, bk, stages = nxt


def _shrink_all_backward_blocks(
    bq_dkv: int, bkv_dkv: int, bq_dq: int, bkv_dq: int, stages: int
) -> tuple[int, int, int, int, int] | None:
    if stages > 1:
        return (bq_dkv, bkv_dkv, bq_dq, bkv_dq, 1)
    nxt = max(bq_dkv, bkv_dkv, bq_dq, bkv_dq)
    if nxt <= _MIN_TILE:
        return None

    def _half(v: int) -> int:
        return max(_MIN_TILE, v // 2) if v > _MIN_TILE else v

    return (_half(bq_dkv), _half(bkv_dkv), _half(bq_dq), _half(bkv_dq), stages)


def _adjust_block_sizes_for_smem(
    block_sizes: BlockSizes,
    num_stages: int,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[BlockSizes, int]:
    """Pick tiles that fit (and are fast on) small-SMEM devices.

    Default block sizes / stages are replaced by the measured per-(dtype,
    head_dim) configuration in ``_SMALL_SMEM_TUNED``; any configuration is then
    shrunk until the shared-memory estimates fit. Never grows a requested tile.
    """
    limit = _query_triton_smem_limit()
    if limit is None or limit >= _DATACENTER_SMEM_THRESHOLD:
        return (block_sizes, num_stages)
    budget = limit - _SMEM_RESERVE
    bd = _next_pow2(head_dim)
    req = (
        block_sizes.block_q,
        block_sizes.block_k,
        block_sizes.block_q_dkv,
        block_sizes.block_kv_dkv,
        block_sizes.block_q_dq,
        block_sizes.block_kv_dq,
    )
    fwd_tiles = req[:2]
    bwd_tiles = req[2:]
    stages = num_stages
    tuned = _SMALL_SMEM_TUNED.get((dtype_bytes, bd))
    if tuned is not None and num_stages == _DEFAULT_NUM_STAGES:
        t_bq, t_bk, t_outer, t_inner, stages, _ = tuned
        if fwd_tiles == _DEFAULT_BLOCK_SIZES[:2]:
            fwd_tiles = (t_bq, t_bk)
        if bwd_tiles == _DEFAULT_BLOCK_SIZES[2:]:
            bwd_tiles = (t_inner, t_outer, t_outer, t_inner)
    bq = min(fwd_tiles[0], q_seq_len)
    bk = min(fwd_tiles[1], kv_seq_len)
    bq, bk, stages = _adjust_fwd_config_for_smem(bq, bk, bd, stages, dtype_bytes, limit)
    cur: tuple[int, int, int, int, int] | None = (
        min(bwd_tiles[0], q_seq_len),
        min(bwd_tiles[1], kv_seq_len),
        min(bwd_tiles[2], q_seq_len),
        min(bwd_tiles[3], kv_seq_len),
        stages,
    )
    while cur is not None:
        c_bq_dkv, c_bkv_dkv, c_bq_dq, c_bkv_dq, c_st = cur
        est = _estimate_bwd_smem_bytes(
            c_bq_dkv, c_bkv_dkv, c_bq_dq, c_bkv_dq, bd, dtype_bytes
        )
        if est <= budget:
            break
        cur = _shrink_all_backward_blocks(c_bq_dkv, c_bkv_dkv, c_bq_dq, c_bkv_dq, c_st)
    bq_dkv, bkv_dkv, bq_dq, bkv_dq = bwd_tiles
    if cur is not None:
        bq_dkv, bkv_dkv, bq_dq, bkv_dq, stages = cur
    # Rebuild full-size BlockSizes (min() with seq len happens at kernel entry),
    # never growing beyond the request.
    new = (bq, bk, bq_dkv, bkv_dkv, bq_dq, bkv_dq)
    return (
        BlockSizes(
            *(min(r, max(n, _MIN_TILE)) for r, n in zip(req, new, strict=True)),
            num_warps_bwd=block_sizes.num_warps_bwd,
            num_stages_bwd=block_sizes.num_stages_bwd,
        ),
        stages,
    )


def _small_smem_onepass_config(
    block_sizes: BlockSizes,
    requested: BlockSizes,
    num_stages: int,
    num_warps: int | None,
    backward_pass_impl: str,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[BlockSizes, str]:
    """Default calls on small-SMEM GPUs use the tuned single-pass backward.

    With ``backward_pass_impl="auto"``: only when the backward tiles and
    launch parameters are at their defaults, the lengths are not strongly
    asymmetric (where the split backward is chosen), the KV length is within
    the measured range and relaxed atomics are available. An explicit
    ``"triton_onepass"`` with default tiles gets the tuned tiles as well.
    """
    explicit = backward_pass_impl in _ONEPASS_BACKWARD_IMPLS
    if (
        (backward_pass_impl != "auto" and not explicit)
        or not RELAXED_ATOMICS_AVAILABLE
        or num_warps is not None
        or num_stages != _DEFAULT_NUM_STAGES
        or requested.num_warps_bwd is not None
        or requested.num_stages_bwd is not None
        or (
            requested.block_q_dkv,
            requested.block_kv_dkv,
            requested.block_q_dq,
            requested.block_kv_dq,
        )
        != _DEFAULT_BLOCK_SIZES[2:]
    ):
        return block_sizes, backward_pass_impl
    bd = _next_pow2(head_dim)
    limit = _query_triton_smem_limit()
    budget = None if limit is None else limit - _SMEM_RESERVE

    def fits(bq: int, bkv: int) -> bool:
        return budget is None or (
            _estimate_onepass_bwd_smem_bytes(bq, bkv, bd, dtype_bytes) <= budget
        )

    key = (dtype_bytes, bd)
    tuned = _SMALL_SMEM_ONEPASS_TUNED.get(key)
    if tuned is None and explicit:
        tuned = _SMALL_SMEM_ONEPASS_EXPLICIT.get(key)
    if tuned is None:
        if not explicit:
            return block_sizes, backward_pass_impl
        # Explicit request without a measured configuration: shrink the
        # current tiles until the single-pass kernel fits.
        bq, bkv = block_sizes.block_q_dkv, block_sizes.block_kv_dkv
        while not fits(bq, bkv) and max(bq, bkv) > _MIN_TILE:
            if bkv >= bq:
                bkv = max(_MIN_TILE, bkv // 2)
            else:
                bq = max(_MIN_TILE, bq // 2)
        tuned = (bq, bkv, block_sizes.num_stages_bwd, block_sizes.num_warps_bwd)
        tuned = (*tuned, None, None)
    bq, bkv, stages, warps, min_kv, max_kv = tuned
    if not fits(bq, bkv):
        return block_sizes, backward_pass_impl
    if not explicit:
        if max_kv is not None and kv_seq_len > max_kv:
            return block_sizes, backward_pass_impl
        if min_kv is not None and kv_seq_len < min_kv:
            return block_sizes, backward_pass_impl
        if BlockSizes._should_use_split_backward(q_seq_len, kv_seq_len, bq, bkv):
            return block_sizes, backward_pass_impl
    return (
        dataclasses.replace(
            block_sizes,
            block_q_dkv=bq,
            block_kv_dkv=bkv,
            block_q_dq=bq,
            block_kv_dq=bkv,
            num_stages_bwd=stages,
            num_warps_bwd=warps,
        ),
        "triton_onepass",
    )


def _small_smem_num_warps(
    num_warps: int | None, num_stages: int, *, head_dim: int, dtype_bytes: int
) -> int | None:
    """Tuned ``num_warps`` for default calls on small-SMEM devices (else as is)."""
    if num_warps is not None or num_stages != _DEFAULT_NUM_STAGES:
        return num_warps
    limit = _query_triton_smem_limit()
    if limit is None or limit >= _DATACENTER_SMEM_THRESHOLD:
        return num_warps
    tuned = _SMALL_SMEM_TUNED.get((dtype_bytes, _next_pow2(head_dim)))
    return num_warps if tuned is None else tuned[5]


def _iter_smaller_block_configs(block_sizes: BlockSizes, num_stages: int) -> Any:
    """Yield progressively smaller (BlockSizes, num_stages) for eager retry."""
    cur_bs, cur_st = block_sizes, num_stages
    yield (cur_bs, cur_st)
    seen = {(cur_bs, cur_st)}
    # First drop stages, then shrink fwd tiles, then shrink everything.
    while True:
        bq, bk, st = cur_bs.block_q, cur_bs.block_k, cur_st
        nxt_fwd = _shrink_fwd_config(bq, bk, st)
        if nxt_fwd is not None:
            nbq, nbk, nst = nxt_fwd
            cur_bs = dataclasses.replace(cur_bs, block_q=nbq, block_k=nbk)
            cur_st = nst
        else:
            shrunk = _shrink_all_backward_blocks(
                cur_bs.block_q_dkv,
                cur_bs.block_kv_dkv,
                cur_bs.block_q_dq,
                cur_bs.block_kv_dq,
                cur_st,
            )
            if shrunk is None:
                return
            bq_dkv, bkv_dkv, bq_dq, bkv_dq, nst = shrunk
            # Fwd already at minimum; halve it stays, just drop stages already
            # done — force-halve fwd too if still stuck.
            nbq = max(_MIN_TILE, cur_bs.block_q // 2)
            nbk = max(_MIN_TILE, cur_bs.block_k // 2)
            if nbq == cur_bs.block_q and nbk == cur_bs.block_k:
                return
            cur_bs = dataclasses.replace(
                cur_bs,
                block_q=nbq,
                block_k=nbk,
                block_q_dkv=bq_dkv,
                block_kv_dkv=bkv_dkv,
                block_q_dq=bq_dq,
                block_kv_dq=bkv_dq,
            )
            cur_st = nst
        key = (cur_bs, cur_st)
        if key in seen:
            return
        seen.add(key)
        yield (cur_bs, cur_st)


def _gqa_group(num_heads: int, num_kv_heads: int) -> int:
    """Query heads per KV head (1 for plain multi-head attention)."""
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(
            "GQA needs the number of query heads to be a multiple of the number "
            f"of key/value heads, got Hq={num_heads} and Hkv={num_kv_heads}."
        )
    return num_heads // num_kv_heads


def _sum_gqa_group(x: jax.Array, num_kv_heads: int, group: int) -> jax.Array:
    """Sum per-query-head gradients ``[B, L, Hq, D]`` over each KV head's group."""
    b, length, _, d = x.shape
    x = x.reshape(b, length, num_kv_heads, group, d)
    return jnp.sum(x, axis=3, dtype=jnp.float32).astype(x.dtype)


def _mha_impl_raw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    rng_seed: jax.Array,
    b_data,
    q_id,
    k_id,
    dropout_mask,
    index_offset,
    index_offset_size,
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
    grid: Any,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    output_activations: bool = False,
):
    """Raw MHA forward taking pre-extracted arrays. No mask/bias objects.

    All array data is passed as positional args (safe for custom_partitioning).
    BlockSpecs and callables are keyword-only (safe for closures).
    """
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    # GQA/MQA: query head h reads KV head h // group (the jnp.repeat layout).
    group = _gqa_group(num_heads, k.shape[2])
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_d = pl.next_power_of_2(head_dim)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if block_d <= 64 else 8)
    effective_num_stages = _resolve_attention_num_stages(
        num_stages, has_dense_bias=b_data is not None
    )

    kernel = functools.partial(
        mha_forward_kernel,
        sm_scale=sm_scale,
        head_dim=head_dim,
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        mask_fn=mask_fn,
        bias_fn=bias_fn,
        dropout_rate=dropout_rate,
    )

    in_specs: list[pl.BlockSpec | None] = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
    ]

    in_specs.append(b_spec if b_data is not None else None)

    if q_id is not None or k_id is not None:
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
    in_specs.append(pl.BlockSpec((), lambda *_: ()))

    if index_offset is not None and index_offset_size is not None:
        index_offset_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: (i, 0)), block_shape=((None, block_k))
        )
        index_offset_size_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: i), block_shape=((None,))
        )
        in_specs.append(index_offset_spec)
        in_specs.append(index_offset_size_spec)
    else:
        in_specs.append(None)
        in_specs.append(None)

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
            num_warps=num_warps_, num_stages=effective_num_stages
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
        rng_seed,
        index_offset,
        index_offset_size,
    )

    if output_activations:
        out, lse = pallas_out
        return out, (q, k, v, rng_seed, out, lse)
    return pallas_out


def _mha_impl_jvp_from_lse_raw(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    lse: jax.Array,
    rng_seed: jax.Array,
    b_data,
    q_id,
    k_id,
    dropout_mask,
    index_offset,
    index_offset_size,
    *,
    mask_fn,
    bias_fn,
    bias_fn_grad,
    b_spec,
    q_id_spec,
    k_id_spec,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
):
    """Raw JVP-from-LSE kernel taking pre-extracted arrays.

    This is the slow path (mask/bias/dropout present).
    """
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    # GQA/MQA: query head h reads KV head h // group (the jnp.repeat layout).
    group = _gqa_group(num_heads, k.shape[2])
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_d = pl.next_power_of_2(head_dim)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if block_d <= 64 else 8)
    effective_num_stages = _resolve_attention_num_stages(
        num_stages, has_dense_bias=b_data is not None
    )

    kernel = functools.partial(
        mha_jvp_from_lse_kernel,
        sm_scale=sm_scale,
        head_dim=head_dim,
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
        mask_fn=mask_fn,
        bias_fn=bias_fn,
        bias_fn_grad=bias_fn_grad,
        dropout_rate=dropout_rate,
    )

    in_specs: list[pl.BlockSpec | None] = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, block_d),
            lambda _, j, k_: (j, 0, k_ // group, 0),
        ),
    ]

    in_specs.append(b_spec if b_data is not None else None)

    if q_id is not None or k_id is not None:
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
    in_specs.append(pl.BlockSpec((), lambda *_: ()))

    in_specs.append(pl.BlockSpec((None, None, block_q), lambda i, j, k_: (j, k_, i)))

    if index_offset is not None and index_offset_size is not None:
        index_offset_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: (i, 0)), block_shape=((None, block_k))
        )
        index_offset_size_spec = pl.BlockSpec(
            index_map=(lambda i, _, k: i), block_shape=((None,))
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

    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps_, num_stages=effective_num_stages
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
        rng_seed,
        lse,
        index_offset,
        index_offset_size,
    )


def _flatten_optional_pytree(value):
    if value is None:
        return None, ()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    return treedef, tuple(leaves)


def _unflatten_optional_pytree(treedef, leaves):
    if treedef is None:
        return None
    return jax.tree_util.tree_unflatten(treedef, leaves)


# -- CP factory functions wrapping raw kernels -----------------------------


# ---------------------------------------------------------------------------
# MHA primitives
# ---------------------------------------------------------------------------
# Declarative specs replace the hand-written fwd/jvp/bwd sharding rules and
# the sentinel plumbing for optional mask/bias arrays. Note q and k/v carry
# distinct sequence factors (q_seq / kv_seq): unlike the previous shared
# "seq" factor, this is correct for cross-attention with different lengths.

_Q_DIMS = ("batch", "q_seq", "heads", "head_dim")
_KV_DIMS = ("batch", "kv_seq", "heads", "head_dim")
_LSE_DIMS = ("batch", "heads", "q_seq")

_MHA_FWD_OPTIONALS = (
    Operand("b_data", ("batch", "heads", "_", "_"), optional=True),
    Operand("q_id", ("batch", "_", "_", "_"), optional=True),
    Operand("k_id", ("batch", "_", "_", "_"), optional=True),
    Operand("dropout_mask", ("batch", "heads", "_", "_"), optional=True),
    Operand("index_offset", ("_", "_"), optional=True),
    Operand("index_offset_size", ("_",), optional=True),
)

_MHA_FWD_SPEC = KernelSpec(
    name="mha_fwd",
    operands=(
        Operand("q", _Q_DIMS),
        Operand("k", _KV_DIMS),
        Operand("v", _KV_DIMS),
        Operand("rng_seed", ()),
    )
    + _MHA_FWD_OPTIONALS,
    outputs=(Output(_Q_DIMS, dtype_like="q"),),
    shardable=frozenset({"batch", "heads"}),
)

_MHA_FWD_RES_SPEC = KernelSpec(
    name="mha_fwd_res",
    operands=_MHA_FWD_SPEC.operands,
    outputs=(
        Output(_Q_DIMS, dtype_like="q"),
        Output(_LSE_DIMS, dtype_like=jnp.float32),
    ),
    shardable=_MHA_FWD_SPEC.shardable,
)

_MHA_JVP_SPEC = KernelSpec(
    name="mha_jvp",
    operands=(
        Operand("q", _Q_DIMS),
        Operand("k", _KV_DIMS),
        Operand("v", _KV_DIMS),
        Operand("dq", _Q_DIMS),
        Operand("dk", _KV_DIMS),
        Operand("dv", _KV_DIMS),
        Operand("lse", _LSE_DIMS),
        Operand("rng_seed", ()),
    )
    + _MHA_FWD_OPTIONALS,
    outputs=(Output(_Q_DIMS, dtype_like="q"),),
    shardable=_MHA_FWD_SPEC.shardable,
)

_MHA_BWD_SPEC = KernelSpec(
    name="mha_bwd",
    operands=(
        Operand("do", _Q_DIMS),
        Operand("q", _Q_DIMS),
        Operand("k", _KV_DIMS),
        Operand("v", _KV_DIMS),
        Operand("rng_seed", ()),
        Operand("out", _Q_DIMS),
        Operand("lse", _LSE_DIMS),
        Operand("b_data", ("batch", "heads", "_", "_"), optional=True),
        Operand("q_data", ("batch", "_", "_", "_"), optional=True),
        Operand("k_data", ("batch", "_", "_", "_"), optional=True),
        Operand("dropout_mask", ("batch", "heads", "_", "_"), optional=True),
        Operand("q_index_offset", ("_", "_"), optional=True),
        Operand("q_index_offset_size", ("_",), optional=True),
        Operand("kv_index_offset", ("_", "_"), optional=True),
        Operand("kv_index_offset_size", ("_",), optional=True),
    ),
    outputs=(
        Output(_Q_DIMS, dtype_like="q"),
        Output(_KV_DIMS, dtype_like="k"),
        Output(_KV_DIMS, dtype_like="v"),
    ),
    shardable=_MHA_FWD_SPEC.shardable,
)


# GQA/MQA variants: K/V carry their own head factor, so the KV head count may
# differ from the query head count. Heads are not shardable here -- the
# kernel's ``h // group`` indexing needs both head axes split consistently,
# which independent factors cannot guarantee -- so they stay replicated.
_KV_DIMS_GQA = ("batch", "kv_seq", "kv_heads", "head_dim")
_GQA_SHARDABLE = frozenset({"batch"})


def _gqa_spec(spec: KernelSpec, name: str) -> KernelSpec:
    def kv_dims(dims):
        return _KV_DIMS_GQA if dims == _KV_DIMS else dims

    return KernelSpec(
        name=name,
        operands=tuple(
            Operand(op.name, kv_dims(op.dims), optional=op.optional)
            for op in spec.operands
        ),
        outputs=tuple(
            Output(kv_dims(out.dims), dtype_like=out.dtype_like) for out in spec.outputs
        ),
        shardable=_GQA_SHARDABLE,
    )


_MHA_FWD_GQA_SPEC = _gqa_spec(_MHA_FWD_SPEC, "mha_fwd_gqa")
_MHA_FWD_RES_GQA_SPEC = _gqa_spec(_MHA_FWD_RES_SPEC, "mha_fwd_res_gqa")
_MHA_JVP_GQA_SPEC = _gqa_spec(_MHA_JVP_SPEC, "mha_jvp_gqa")
_MHA_BWD_GQA_SPEC = _gqa_spec(_MHA_BWD_SPEC, "mha_bwd_gqa")


def _mha_fwd_prim_impl(*args, **static):
    return _mha_impl_raw(*args, output_activations=False, **static)


def _mha_fwd_res_prim_impl(*args, **static):
    out, (_q, _k, _v, _seed, _out_res, lse) = _mha_impl_raw(
        *args, output_activations=True, **static
    )
    return out, lse


def _mha_jvp_prim_impl(*args, **static):
    # Late binding: _mha_impl_jvp_from_lse_raw is defined further down.
    return _mha_impl_jvp_from_lse_raw(*args, **static)


def _mha_bwd_prim_impl(*args, **static):
    # Late binding: _mha_backward_raw is defined further down.
    return _mha_backward_raw(*args, **static)


mha_fwd_p = make_kernel_primitive(_MHA_FWD_SPEC, impl=_mha_fwd_prim_impl)
mha_fwd_res_p = make_kernel_primitive(_MHA_FWD_RES_SPEC, impl=_mha_fwd_res_prim_impl)
mha_jvp_p = make_kernel_primitive(_MHA_JVP_SPEC, impl=_mha_jvp_prim_impl)
mha_bwd_p = make_kernel_primitive(_MHA_BWD_SPEC, impl=_mha_bwd_prim_impl)
mha_fwd_gqa_p = make_kernel_primitive(_MHA_FWD_GQA_SPEC, impl=_mha_fwd_prim_impl)
mha_fwd_res_gqa_p = make_kernel_primitive(
    _MHA_FWD_RES_GQA_SPEC, impl=_mha_fwd_res_prim_impl
)
mha_jvp_gqa_p = make_kernel_primitive(_MHA_JVP_GQA_SPEC, impl=_mha_jvp_prim_impl)
mha_bwd_gqa_p = make_kernel_primitive(_MHA_BWD_GQA_SPEC, impl=_mha_bwd_prim_impl)


def _mha_prims(q, k):
    """(fwd, fwd_res, jvp, bwd) primitives for these head counts."""
    if q.shape[-2] == k.shape[-2]:
        return mha_fwd_p, mha_fwd_res_p, mha_jvp_p, mha_bwd_p
    return mha_fwd_gqa_p, mha_fwd_res_gqa_p, mha_jvp_gqa_p, mha_bwd_gqa_p


def _aligned_seq_len(seq_len: int, *blocks: int) -> int:
    """Smallest length >= ``seq_len`` that every effective tile divides.

    Each kernel uses ``min(block, seq_len)`` as its tile, so a length is safe
    when it is a multiple of the largest requested block, or (when shorter than
    that block) a power of two that the smaller power-of-two blocks divide.
    """
    largest = max(blocks)
    if seq_len >= largest:
        return ceil_div(seq_len, largest) * largest
    return _next_pow2(seq_len)


# Masks whose value depends only on (q_idx, k_idx), so appending padded
# positions does not change the mask on the original positions.
_POSITION_ONLY_MASKS = (CausalMask, LocalWindowMask, CausalLocalWindowMask, NoMask)


def _check_paddable(mask, bias, dropout_rate, dropout_impl, q_len, kv_len) -> None:
    reason = None
    if dropout_rate > 0 and dropout_impl != "counter":
        # Counter-based dropout hashes absolute positions, so padding is safe.
        reason = f"dropout_impl={dropout_impl!r}"
    elif mask is not None and type(mask) not in _POSITION_ONLY_MASKS:
        reason = f"mask {type(mask).__name__}"
    elif bias is not None:
        reason = f"bias {type(bias).__name__}"
    if reason is not None:
        raise ValueError(
            f"mha needs block-aligned sequence lengths (got q_len={q_len}, "
            f"kv_len={kv_len}) when used with {reason}: the kernel would read past "
            "the end of the sequence. Pad the inputs (and the mask/bias data) to a "
            "multiple of the block size, or use "
            "probjax.nn.layers.attention.flex_attention, which pads for you."
        )


# Split-KV forward ("flash-decoding") dispatch. The regular forward runs one
# program per (batch, head, query tile); with few queries that underfills the
# GPU while each program walks the whole KV sequence. Thresholds measured on an
# RTX 3090 (docs/benchmarks/kernel_attn_ext_gpu.json, suite ``splitsweep``).
_SPLITKV_MAX_Q_LEN = 64
_SPLITKV_MIN_KV_LEN = 1024


def use_splitkv(
    q,
    k,
    *,
    mask=None,
    bias=None,
    dropout_rate: float = 0.0,
    interpret: bool = False,
    grid=None,
    split_kv: bool | None = None,
) -> bool:
    """Whether :func:`mha` runs the split-KV forward for these inputs.

    Only exact cases: no mask/bias/dropout, GPU (not interpret mode), default
    grid. ``split_kv=False`` disables it; ``None`` decides by shape;
    ``True`` forces it whenever it is exact.
    """
    if split_kv is False or interpret or grid is not None:
        return False
    if mask is not None and not isinstance(mask, NoMask):
        return False
    if bias is not None or dropout_rate > 0:
        return False
    if q.ndim != 4 or k.ndim != 4 or use_xla_attention(q, interpret):
        return False
    if q.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
        return False
    if _query_triton_smem_limit() is None:
        return False
    if split_kv:
        return True
    q_len, kv_len = q.shape[1], k.shape[1]
    # Measured: split-KV (>= 2 splits) is faster than or equal to the regular
    # forward for every Lq <= 64, Lkv >= 1024 case swept (B in {1, 8},
    # Hkv in {8, 32}); 1.1-1.4x when the grid is already full (B=8, MHA), up
    # to ~6x for single-sequence GQA decoding.
    return q_len <= _SPLITKV_MAX_Q_LEN and kv_len >= _SPLITKV_MIN_KV_LEN


def mha(
    q,
    k,
    v,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    rng: jax.Array | None = None,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "auto",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    dropout_rate: float = 0.0,
    dropout_impl: str = "counter",
    diff_mode: str = "reverse",
    split_kv: bool | None = None,
):
    """Multi-Head Attention public API.

    ``k``/``v`` may have fewer heads than ``q`` (GQA/MQA): query head ``h``
    uses KV head ``h // (Hq // Hkv)`` -- the ``jnp.repeat`` convention -- read
    directly by the kernels (no repeated K/V copy); dK/dV are summed over each
    group.

    ``split_kv``: with few queries and a long KV sequence (decoding), the
    forward partitions the KV sequence over extra programs and combines the
    partial results by log-sum-exp (see ``attention_splitkv``). ``None``
    (default) picks it by shape when exact (no mask/bias/dropout), ``False``
    never, ``True`` whenever exact. Differentiation always uses the regular
    kernels.

    Reverse-mode differentiation uses a lightweight ``custom_vjp`` wrapper
    around the raw Pallas forward/backward kernels.  When a multi-device mesh
    is active, those kernels are lowered per-shard via ``custom_partitioning``
    so ordinary ``jax.jit(fn)(NamedSharding inputs)`` call sites do not incur
    GSPMD all-gathers before ``pallas_call``.

    Forward-mode differentiation uses a separate fused-JVP path.  The public
    API automatically routes JVP traces to that path so plain ``jax.jvp(mha)``
    continues to work without involving the reverse-mode wrapper.
    """
    if diff_mode not in ("reverse", "forward"):
        raise ValueError(
            f"diff_mode must be 'reverse' or 'forward', got {diff_mode!r}."
        )
    # GQA/MQA: K/V may have fewer heads than Q (query head h uses KV head
    # h // (Hq // Hkv)); validated here, handled natively by the kernels.
    _gqa_group(q.shape[-2], k.shape[-2])
    if v.shape[-2] != k.shape[-2]:
        raise ValueError(
            f"key and value need the same number of heads, got {k.shape[-2]} "
            f"and {v.shape[-2]}."
        )

    # CPU: the Triton kernels only run in (very slow) Pallas interpret mode
    # there, so use the XLA implementation unless interpret mode was requested.
    if use_xla_attention(q, interpret):
        return xla_attention(
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            rng=rng,
            sm_scale=sm_scale,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
            diff_mode=diff_mode,
        )

    if (
        diff_mode == "reverse"
        and use_splitkv(
            q,
            k,
            mask=mask,
            bias=bias,
            dropout_rate=dropout_rate,
            interpret=interpret,
            grid=grid,
            split_kv=split_kv,
        )
        and not _is_direct_jvp_trace(q, k, v)
    ):
        regular = functools.partial(
            mha,
            mask=mask,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            backward_pass_impl=backward_pass_impl,
            num_warps=num_warps,
            num_stages=num_stages,
            debug=debug,
            split_kv=False,
        )
        return splitkv_attention(q, k, v, sm_scale=sm_scale, fallback=regular)

    # The kernels tile Q and KV with power-of-two blocks and do not mask a
    # partial last tile, so unaligned sequence lengths would read past the end
    # of K/V. Pad to aligned lengths and mask the padded keys instead.
    q_len, kv_len = int(q.shape[-3]), int(k.shape[-3])
    q_pad = _aligned_seq_len(
        q_len, block_sizes.block_q, block_sizes.block_q_dkv, block_sizes.block_q_dq
    )
    kv_pad = _aligned_seq_len(
        kv_len, block_sizes.block_k, block_sizes.block_kv_dkv, block_sizes.block_kv_dq
    )
    if (q_pad, kv_pad) != (q_len, kv_len):
        _check_paddable(mask, bias, dropout_rate, dropout_impl, q_len, kv_len)

        def _pad_seq(x, target):
            widths = [(0, 0)] * x.ndim
            widths[-3] = (0, target - x.shape[-3])
            return jnp.pad(x, widths)

        length_mask = QKVLengthMask(
            q_length=q_pad, kv_length=kv_len, block_sparse=False
        )
        out = mha(
            _pad_seq(q, q_pad),
            _pad_seq(k, kv_pad),
            _pad_seq(v, kv_pad),
            mask=length_mask if mask is None else mask & length_mask,
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
            diff_mode=diff_mode,
            split_kv=False,
        )
        return lax.slice_in_dim(out, 0, q_len, axis=out.ndim - 3)

    # Flatten mask/bias pytrees: array leaves become regular traced args,
    # treedefs are truly static metadata.
    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)

    force_forward_mode = diff_mode == "forward" or _is_direct_jvp_trace(
        q, k, v, rng, mask_leaves, bias_leaves
    )
    factory = _make_mha_forward_jvp if force_forward_mode else _make_mha_custom_vjp

    def _build_inner(
        bs: BlockSizes,
        st: int,
        warps: int | None = num_warps,
        bwd_impl: str = backward_pass_impl,
    ):
        return factory(
            mask_treedef=mask_treedef,
            bias_treedef=bias_treedef,
            sm_scale=sm_scale,
            block_sizes=bs,
            backward_pass_impl=bwd_impl,
            num_warps=warps,
            num_stages=st,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )

    # Fast path (zero overhead, original behavior): datacenter GPUs, unknown
    # devices (TPU/CPU/no libcuda), or interpret mode. Only confirmed-small
    # NVIDIA GPUs (<150KB opt-in SMEM) take the adaptive slow path below.
    try:
        _smem_limit = None if interpret else _query_triton_smem_limit()
    except Exception:
        _smem_limit = None
    if interpret or _smem_limit is None or _smem_limit >= _DATACENTER_SMEM_THRESHOLD:
        return _build_inner(block_sizes, num_stages)(
            q, k, v, rng, mask_leaves, bias_leaves
        )

    def _maybe_proactive() -> tuple[BlockSizes, int]:
        try:
            q_shape = tuple(q.shape)
            k_shape = tuple(k.shape)
            q_seq_len = int(q_shape[-3])
            kv_seq_len = int(k_shape[-3])
            head_dim = int(q_shape[-1])
            dtype_bytes = int(jnp.dtype(q.dtype).itemsize)
        except Exception:
            return (block_sizes, num_stages)
        try:
            return _adjust_block_sizes_for_smem(
                block_sizes,
                num_stages,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                head_dim=head_dim,
                dtype_bytes=dtype_bytes,
            )
        except Exception:
            return (block_sizes, num_stages)

    start_bs, start_st = _maybe_proactive()
    start_impl = backward_pass_impl
    try:
        start_bs, start_impl = _small_smem_onepass_config(
            start_bs,
            block_sizes,
            num_stages,
            num_warps,
            backward_pass_impl,
            q_seq_len=int(q.shape[-3]),
            kv_seq_len=int(k.shape[-3]),
            head_dim=int(q.shape[-1]),
            dtype_bytes=int(jnp.dtype(q.dtype).itemsize),
        )
    except Exception:
        start_impl = backward_pass_impl
    try:
        start_warps = _small_smem_num_warps(
            num_warps,
            num_stages,
            head_dim=int(q.shape[-1]),
            dtype_bytes=int(jnp.dtype(q.dtype).itemsize),
        )
    except Exception:
        start_warps = num_warps

    def _is_tracer_arg(x) -> bool:
        try:
            return is_tracing(x)
        except Exception:
            return False

    try:
        leaves = jax.tree_util.tree_leaves((q, k, v, rng, mask_leaves, bias_leaves))
    except Exception:
        leaves = [q, k, v]
    eager = not any(_is_tracer_arg(x) for x in leaves)
    # Under jit/vmap/grad tracing the kernel runs later at runtime; the
    # proactive config above is the best we can do there. For eager execution
    # we can also retry with smaller tiles on RESOURCE_EXHAUSTED.
    if not eager:
        return _build_inner(start_bs, start_st, start_warps, start_impl)(
            q, k, v, rng, mask_leaves, bias_leaves
        )

    last_exc: BaseException | None = None
    for attempt_bs, attempt_st in _iter_smaller_block_configs(start_bs, start_st):
        try:
            out = _build_inner(attempt_bs, attempt_st, start_warps, start_impl)(
                q, k, v, rng, mask_leaves, bias_leaves
            )
            # Force synchronization so async compilation/runtime errors surface
            # inside this try block and can trigger fallback.
            try:
                jax.block_until_ready(out)
            except Exception as sync_exc:
                if _is_smem_resource_error(sync_exc):
                    last_exc = sync_exc
                    continue
                raise
            return out
        except Exception as exc:  # noqa: BLE001 - must inspect for SMEM retry
            if not _is_smem_resource_error(exc):
                raise
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return _build_inner(start_bs, start_st, start_warps)(
        q, k, v, rng, mask_leaves, bias_leaves
    )


def _is_direct_jvp_trace(*args) -> bool:
    """Return True only for active direct forward-mode JVP traces.

    Reverse-mode linearization may introduce tracing as well, but we only want
    to route plain ``jax.jvp(...)`` calls to the fused forward-mode path.
    Requiring a non-symbolic-zero tangent keeps this check narrow.
    """
    flat_args, _ = jax.tree_util.tree_flatten(args)
    for x in flat_args:
        if isinstance(x, ad.JVPTracer) and not isinstance(x.tangent, ad_util.Zero):
            return True
    return False


def _zero_cotangent_tuple(xs):
    return tuple(None for _ in xs)


def _make_mha_runners(
    *,
    mask_treedef,
    bias_treedef,
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
):
    """Build common forward/backward runners shared by custom_vjp/custom_jvp."""

    def _reconstruct_mask_bias(mask_leaves_tuple, bias_leaves_tuple):
        return (
            _unflatten_optional_pytree(mask_treedef, mask_leaves_tuple),
            _unflatten_optional_pytree(bias_treedef, bias_leaves_tuple),
        )

    def _run_forward(
        q,
        k,
        v,
        rng,
        mask_leaves_tuple,
        bias_leaves_tuple,
        *,
        output_activations: bool,
    ):
        mask, bias = _reconstruct_mask_bias(mask_leaves_tuple, bias_leaves_tuple)
        if dropout_rate > 0 and rng is None:
            raise ValueError("dropout_rate > 0 requires a non-None rng.")
        rng_val = rng if rng is not None else jax.random.key(0)
        rng_seed = _rng_seed_from_key(rng_val)

        batch_size, q_seq_len, num_heads, _head_dim = q.shape
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)

        arrays, specs = _extract_fwd_data(
            mask,
            bias,
            rng,
            batch_size=batch_size,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            num_heads=num_heads,
            block_q=block_q,
            block_k=block_k,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )

        static = dict(
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            b_spec=specs["b_spec"],
            q_id_spec=specs["q_id_spec"],
            k_id_spec=specs["k_id_spec"],
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )
        operands = {"q": q, "k": k, "v": v, "rng_seed": rng_seed, **arrays}
        if output_activations:
            out, lse = shardable_kernel(_mha_prims(q, k)[1], operands, **static)
            return out, rng_val, rng_seed, lse
        return shardable_kernel(_mha_prims(q, k)[0], operands, **static)

    def _run_backward(
        do,
        q,
        k,
        v,
        rng,
        rng_seed,
        out,
        lse,
        mask_leaves_tuple,
        bias_leaves_tuple,
    ):
        mask, bias = _reconstruct_mask_bias(mask_leaves_tuple, bias_leaves_tuple)

        batch_size, q_seq_len, num_heads, _head_dim = q.shape
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)
        block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
        block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
        block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
        block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)

        arrays, specs = _extract_bwd_data(
            mask,
            bias,
            rng,
            batch_size=batch_size,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            num_heads=num_heads,
            block_q=block_q,
            block_k=block_k,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_q_dq=block_q_dq,
            block_kv_dq=block_kv_dq,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )

        return shardable_kernel(
            _mha_prims(q, k)[3],
            {
                "do": do,
                "q": q,
                "k": k,
                "v": v,
                "rng_seed": rng_seed,
                "out": out,
                "lse": lse,
                **arrays,
            },
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            bias_fn_grad=specs["bias_fn_grad"],
            b_spec_bwd=specs["b_spec_bwd"],
            q_data_spec=specs["q_data_spec"],
            k_data_spec=specs["k_data_spec"],
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            backward_pass_impl=backward_pass_impl,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )

    return _reconstruct_mask_bias, _run_forward, _run_backward


def _make_mha_custom_vjp(
    *,
    mask_treedef,
    bias_treedef,
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
):
    """Build the reverse-mode path using custom_vjp plus custom_partitioning."""

    _, _run_forward, _run_backward = _make_mha_runners(
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
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

    @jax.custom_vjp
    def _mha_inner(q, k, v, rng, mask_leaves_tuple, bias_leaves_tuple):
        return _run_forward(
            q,
            k,
            v,
            rng,
            mask_leaves_tuple,
            bias_leaves_tuple,
            output_activations=False,
        )

    def _mha_inner_fwd(q, k, v, rng, mask_leaves_tuple, bias_leaves_tuple):
        out, rng_val, rng_seed, lse = _run_forward(
            q,
            k,
            v,
            rng,
            mask_leaves_tuple,
            bias_leaves_tuple,
            output_activations=True,
        )
        residual = (
            q,
            k,
            v,
            rng_val,
            rng_seed,
            out,
            lse,
            mask_leaves_tuple,
            bias_leaves_tuple,
        )
        return out, residual

    def _mha_inner_bwd(res, do):
        (
            q,
            k,
            v,
            rng_val,
            rng_seed,
            out,
            lse,
            mask_leaves_tuple,
            bias_leaves_tuple,
        ) = res

        if isinstance(do, ad_util.Zero):
            return (
                None,
                None,
                None,
                None,
                _zero_cotangent_tuple(mask_leaves_tuple),
                _zero_cotangent_tuple(bias_leaves_tuple),
            )

        dq, dk, dv = _run_backward(
            do,
            q,
            k,
            v,
            rng_val,
            rng_seed,
            out,
            lse,
            mask_leaves_tuple,
            bias_leaves_tuple,
        )
        return (
            dq,
            dk,
            dv,
            None,
            _zero_cotangent_tuple(mask_leaves_tuple),
            _zero_cotangent_tuple(bias_leaves_tuple),
        )

    _mha_inner.defvjp(_mha_inner_fwd, _mha_inner_bwd)
    return _mha_inner


def _make_mha_forward_jvp(
    *,
    mask_treedef,
    bias_treedef,
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
):
    """Build the forward-mode path used for JVP traces."""

    _reconstruct_mask_bias, _run_forward, _ = _make_mha_runners(
        mask_treedef=mask_treedef,
        bias_treedef=bias_treedef,
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

    @jax.custom_jvp
    def _mha_inner(q, k, v, rng, mask_leaves_tuple, bias_leaves_tuple):
        return _run_forward(
            q,
            k,
            v,
            rng,
            mask_leaves_tuple,
            bias_leaves_tuple,
            output_activations=False,
        )

    @_mha_inner.defjvp
    def _jvp_rule(primals, tangents):
        q, k, v, rng, mask_leaves_tuple, bias_leaves_tuple = primals
        dq, dk, dv, _drng, _dmask, _dbias = tangents
        del _drng, _dmask, _dbias

        if any(ad.is_undefined_primal(x) for x in (q, k, v)):
            raise ValueError(
                "diff_mode='forward' does not support reverse-mode "
                "autodiff; use diff_mode='reverse' for grad."
            )

        mask, bias = _reconstruct_mask_bias(mask_leaves_tuple, bias_leaves_tuple)

        def _tangent_or_zero(t, primal):
            if isinstance(t, ad_util.Zero):
                return jnp.zeros_like(primal)
            return t

        dq = _tangent_or_zero(dq, q)
        dk = _tangent_or_zero(dk, k)
        dv = _tangent_or_zero(dv, v)

        out, rng_val, rng_seed, lse_res = _run_forward(
            q,
            k,
            v,
            rng,
            mask_leaves_tuple,
            bias_leaves_tuple,
            output_activations=True,
        )

        batch_size, q_seq_len, num_heads, _head_dim = q.shape
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_k = min(block_sizes.block_k, kv_seq_len)
        arrays, specs = _extract_fwd_data(
            mask,
            bias,
            rng,
            batch_size=batch_size,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            num_heads=num_heads,
            block_q=block_q,
            block_k=block_k,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )
        bias_fn_grad = bias.grad if (bias is not None) else None

        tangent_out = shardable_kernel(
            _mha_prims(q, k)[2],
            {
                "q": q,
                "k": k,
                "v": v,
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "lse": lse_res,
                "rng_seed": rng_seed,
                **arrays,
            },
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            bias_fn_grad=bias_fn_grad,
            b_spec=specs["b_spec"],
            q_id_spec=specs["q_id_spec"],
            k_id_spec=specs["k_id_spec"],
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )
        return out, tangent_out

    return _mha_inner


def _extract_bwd_data(
    mask: AttentionMask | None,
    bias: AttentionBias | None,
    rng: jax.Array | None,
    *,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    block_q: int,
    block_k: int,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    dropout_rate: float,
    dropout_impl: str,
):
    """Extract all arrays and BlockSpecs from mask/bias for backward.

    Returns arrays dict and specs dict.
    """
    # Mask data
    q_data = k_data = None
    if isinstance(mask, AttentionMask):
        q_data, k_data = mask.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)

    # Bias data
    b_data = bias.get_data() if (bias is not None) else None

    # Dropout mask
    dropout_mask = None
    if dropout_rate > 0:
        assert rng is not None
        if dropout_impl == "materialize":
            dropout_mask = get_dropout_mask(
                (batch_size, num_heads, q_seq_len, kv_seq_len),
                prng_key=rng,
                rate=dropout_rate,
            )
        elif dropout_impl == "counter":
            pass
        else:
            raise ValueError(f"Unsupported dropout_impl={dropout_impl!r}")

    # Block-sparse iterators (need different block sizes than forward)
    q_index_offset = q_index_offset_size = None
    kv_index_offset = kv_index_offset_size = None
    if mask is not None:
        q_index_offset, q_index_offset_size = mask.query_iterator_indices(
            q_seq_len, kv_seq_len, block_q_dq, block_kv_dq
        )
        kv_index_offset, kv_index_offset_size = mask.kv_iterator_indices(
            q_seq_len, kv_seq_len, block_q_dkv, block_kv_dkv
        )

    # BlockSpecs (not arrays)
    q_data_spec = k_data_spec = None
    if mask is not None and q_data is not None:
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

    b_spec_bwd = None
    if bias is not None and b_data is not None:
        b_spec_bwd = bias.get_block_spec_backward(
            q_len=q_seq_len,
            kv_len=kv_seq_len,
            block_q=q_seq_len,
            block_kv=kv_seq_len,
        )

    # Callables
    mask_fn = mask.__call__ if mask is not None else None
    bias_fn = bias.__call__ if bias is not None else None
    bias_fn_grad = bias.grad if (bias is not None) else None

    arrays = dict(
        b_data=b_data,
        q_data=q_data,
        k_data=k_data,
        dropout_mask=dropout_mask,
        q_index_offset=q_index_offset,
        q_index_offset_size=q_index_offset_size,
        kv_index_offset=kv_index_offset,
        kv_index_offset_size=kv_index_offset_size,
    )
    specs = dict(
        b_spec_bwd=b_spec_bwd,
        q_data_spec=q_data_spec,
        k_data_spec=k_data_spec,
        mask_fn=mask_fn,
        bias_fn=bias_fn,
        bias_fn_grad=bias_fn_grad,
    )
    return arrays, specs


def _mha_backward_raw(
    do,
    q,
    k,
    v,
    rng_seed,
    out,
    lse,
    b_data,
    q_data,
    k_data,
    dropout_mask,
    q_index_offset,
    q_index_offset_size,
    kv_index_offset,
    kv_index_offset_size,
    *,
    mask_fn,
    bias_fn,
    bias_fn_grad,
    b_spec_bwd,
    q_data_spec,
    k_data_spec,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
):
    """Raw backward taking pre-extracted arrays. No mask/bias objects.

    ``rng_seed`` is a scalar uint32 seed (not a PRNGKey).
    """
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    num_kv_heads = k.shape[2]
    group = _gqa_group(num_heads, num_kv_heads)
    # GQA: the kernels write dK/dV per *query* head (shape [B, Lkv, Hq, D]);
    # the group's contributions are summed below.
    dkv_shape = (batch_size, kv_seq_len, num_heads, k.shape[3])
    block_d = pl.next_power_of_2(head_dim)
    block_q = min(block_sizes.block_q, q_seq_len)
    _block_k = min(block_sizes.block_k, kv_seq_len)
    block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
    block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
    block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
    block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)
    if block_sizes.num_stages_bwd is not None:
        num_stages = block_sizes.num_stages_bwd
    if block_sizes.num_warps_bwd is not None:
        num_warps = block_sizes.num_warps_bwd
    effective_num_stages = _resolve_attention_num_stages(
        num_stages, has_dense_bias=b_data is not None
    )

    # Resolve auto to actual impl
    if backward_pass_impl == "auto":
        if BlockSizes._should_use_split_backward(
            q_seq_len, kv_seq_len, block_q_dq, block_kv_dkv
        ):
            backward_pass_impl = "triton_split"
        else:
            backward_pass_impl = "triton_fused"

    # num_warps
    num_warps_ = num_warps
    if num_warps_ is None:
        if (
            block_q_dkv * block_kv_dkv < 128 * 128
            or block_q_dq * block_kv_dq < 128 * 128
        ):
            num_warps_ = 4
        else:
            num_warps_ = 8

    # Preprocess delta
    delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)

    if backward_pass_impl == "triton_fused":
        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        # Enforce tile count matching for fused backward
        nq = pl.cdiv(q_seq_len, block_q_dq)
        nkv = pl.cdiv(kv_seq_len, block_kv_dkv)
        if nq != nkv:
            n = max(nq, nkv)
            block_q_dq = max((q_seq_len + n - 1) // n, 1)
            block_kv_dkv = max((kv_seq_len + n - 1) // n, 1)

        out_shapes = [
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(dkv_shape, k.dtype),
            jax.ShapeDtypeStruct(dkv_shape, v.dtype),
        ]

        in_specs = [
            # q, k, v
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            # data_q, data_k
            q_data_spec if q_data is not None else None,
            k_data_spec if q_data is not None else None,
            # bias
            b_spec_bwd if b_data is not None else None,
            # dropout mask
            None,
            # rng seed
            pl.BlockSpec((), lambda *_: ()),
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

        if dropout_mask is not None:
            in_specs[6] = pl.BlockSpec(
                (None, None, q_seq_len, kv_seq_len), lambda i, j, _: (i, j, 0, 0)
            )

        # Dynamic iterators for fused backward
        if q_index_offset is not None:
            num_kv_blocks_dq = pl.cdiv(kv_seq_len, block_kv_dq)
            in_specs[-4] = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_kv_blocks_dq)),
            )
            in_specs[-3] = pl.BlockSpec(
                index_map=(lambda i, _, k: k), block_shape=((None,))
            )
        if kv_index_offset is not None:
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            in_specs[-2] = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_q_blocks_dkdv)),
            )
            in_specs[-1] = pl.BlockSpec(
                index_map=(lambda i, _, k: k), block_shape=((None,))
            )

        grid_ = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv))

        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                sm_scale=sm_scale,
                bias_fn=bias_fn,
                mask_fn=mask_fn,
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
            grid=grid_,
            out_specs=[
                pl.BlockSpec(
                    (None, block_q_dq, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim),
                    lambda i, j, k: (i, k, j, 0),
                ),
            ],
            name="mha_backward",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(
                num_warps=num_warps_, num_stages=effective_num_stages
            ),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng_seed,
            out,
            do,
            lse,
            delta,
            q_index_offset,
            q_index_offset_size,
            kv_index_offset,
            kv_index_offset_size,
        )
    elif backward_pass_impl in _ONEPASS_BACKWARD_IMPLS:
        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")
        in_specs = [
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            q_data_spec if q_data is not None else None,
            k_data_spec if q_data is not None else None,
            b_spec_bwd if b_data is not None else None,
            (
                None
                if dropout_mask is None
                else pl.BlockSpec(
                    (None, None, q_seq_len, kv_seq_len),
                    lambda i, j, _: (i, j, 0, 0),
                )
            ),
            pl.BlockSpec((), lambda *_: ()),
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            None,
            None,
        ]
        if kv_index_offset is not None:
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            in_specs[-2] = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_q_blocks_dkdv)),
            )
            in_specs[-1] = pl.BlockSpec(
                index_map=(lambda i, _, k: k), block_shape=((None,))
            )
        dq_spec = pl.BlockSpec(
            (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
        )
        in_specs.append(dq_spec)
        onepass_args = (
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng_seed,
            do,
            lse,
            delta,
            kv_index_offset,
            kv_index_offset_size,
            jnp.zeros(q.shape, jnp.float32),
        )
        # None inputs are empty pytrees: alias by position among the arrays.
        dq_input_index = sum(a is not None for a in onepass_args[:-1])
        dk, dv, dq = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_onepass,
                dq_atomic=not interpret,
                sm_scale=sm_scale,
                mask_fn=mask_fn,
                bias_fn=bias_fn,
                bias_fn_grad=bias_fn_grad,
                dropout_rate=dropout_rate,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_d=block_d,
                head_dim=head_dim,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(dkv_shape, k.dtype),
                jax.ShapeDtypeStruct(dkv_shape, v.dtype),
                jax.ShapeDtypeStruct(q.shape, jnp.float32),
            ],
            in_specs=in_specs,
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
                dq_spec,
            ],
            input_output_aliases={dq_input_index: 2},
            name="mha_backward_onepass",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(
                num_warps=num_warps_, num_stages=effective_num_stages
            ),
        )(*onepass_args)
    else:
        # Split backward into two kernels: first dKdV, then dQ.
        if backward_pass_impl not in ("triton_split", "split", "triton_2pass"):
            raise ValueError(
                f"Invalid backward pass implementation: {backward_pass_impl}"
            )

        if not block_sizes.has_backward_blocks:
            raise ValueError("Backward block sizes must all be set.")

        # Common input specs across both kernels
        common_in_specs = [
            # q, k, v
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d),
                lambda i, j, _: (i, 0, j // group, 0),
            ),
            # mask data
            q_data_spec if q_data is not None else None,
            k_data_spec if q_data is not None else None,
            # bias (dense)
            b_spec_bwd if b_data is not None else None,
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
            pl.BlockSpec((), lambda *_: ()),
            # do, lse, delta
            pl.BlockSpec(
                (None, q_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]

        # dKdV call
        dkdv_in_specs = common_in_specs + [None, None]
        if kv_index_offset is not None:
            num_q_blocks_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
            dkdv_in_specs[-2] = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_q_blocks_dkdv)),
            )
            dkdv_in_specs[-1] = pl.BlockSpec(
                index_map=(lambda i, _, k: k), block_shape=((None,))
            )

        dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_split_dkdv,
                sm_scale=sm_scale,
                mask_fn=mask_fn,
                bias_fn=bias_fn,
                bias_fn_grad=bias_fn_grad,
                dropout_rate=dropout_rate,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_d=block_d,
                head_dim=head_dim,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(dkv_shape, k.dtype),
                jax.ShapeDtypeStruct(dkv_shape, v.dtype),
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
            compiler_params=plgpu.CompilerParams(
                num_warps=num_warps_, num_stages=effective_num_stages
            ),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng_seed,
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
            dq_in_specs[-2] = pl.BlockSpec(
                index_map=(lambda i, _, k: (k, 0)),
                block_shape=((None, num_kv_blocks_dq)),
            )
            dq_in_specs[-1] = pl.BlockSpec(
                index_map=(lambda i, _, k: k), block_shape=((None,))
            )

        dq = pl.pallas_call(
            functools.partial(
                mha_backward_kernel_split_dq,
                sm_scale=sm_scale,
                mask_fn=mask_fn,
                bias_fn=bias_fn,
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
            compiler_params=plgpu.CompilerParams(
                num_warps=num_warps_, num_stages=effective_num_stages
            ),
        )(
            q,
            k,
            v,
            q_data,
            k_data,
            b_data,
            dropout_mask,
            rng_seed,
            do,
            lse,
            delta,
            q_index_offset,
            q_index_offset_size,
        )

    if group > 1:
        dk = _sum_gqa_group(dk, num_kv_heads, group)
        dv = _sum_gqa_group(dv, num_kv_heads, group)
    return dq.astype(q.dtype), dk, dv
