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
from jax.experimental import pallas as pl
from jax._src.pallas import primitives as pallas_primitives
from jax.experimental.pallas import triton as plgpu
from jax.interpreters import ad

from ..attention_mask_bias import (
    AttentionBias,
    AttentionMask,
)
from ..kernel_utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    get_dot_precision,
    get_dropout_mask,
)


def _validate_mha_sharding(sharding, name: str):
    """Validate that a NamedSharding on an MHA operand is supported.

    For q/k/v shaped ``(B, T, H, D)`` only sharding on the batch (dim 0) and
    heads (dim 2) dimensions is allowed.  Sharding on the sequence (dim 1) or
    head_dim (dim 3) dimensions raises an informative ``ValueError``.

    Called inside the ``partition()`` callback of ``custom_partitioning``
    where the sharding object is always available.
    """
    spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) == 0:
        return
    for dim_idx, axis in enumerate(spec):
        if axis is None:
            continue
        if dim_idx == 1:
            raise ValueError(
                f"Pallas flash-attention does not support sharding on the "
                f"sequence dimension (dim 1) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 1 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 2) sharding are supported."
            )
        if dim_idx == 3:
            raise ValueError(
                f"Pallas flash-attention does not support sharding on the "
                f"head_dim dimension (dim 3) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 3 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 2) sharding are supported."
            )


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
    """

    block_q: int = 128
    block_k: int = 128

    block_q_dkv: int = 64
    block_kv_dkv: int = 64
    block_q_dq: int = 64
    block_kv_dq: int = 64

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

        def cdiv(a: int, b: int) -> int:
            return (a + b - 1) // b if b > 0 else 0

        # Calculate what tile counts would be with current block sizes
        nq = max(cdiv(q_len, max(block_q_dq, 1)), 1)
        nkv = max(cdiv(kv_len, max(block_kv_dkv, 1)), 1)

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
        block sizes to satisfy: ceil_div(q_len, block_q_dq) == ceil_div(kv_len, block_kv_dkv).
        This may create inefficient tiny tiles for asymmetric sequence lengths.

        For backward_pass_impl="triton_split", uses split backward with independent
        tile sizes optimized for each pass.
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
            # We retrieve the dynamic indices for the current block if offset is provided.
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        qk = pl.dot(q, k.T, precision=precision)  # [block_q, block_k]

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
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=jnp.nan)
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
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

    lse = pallas_load(lse_ref, (curr_q_slice,))
    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, do_acc):
        if index_offset_ref is not None:
            start_k = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_k, 1),)))
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pallas_load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pallas_load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = pl.dot(q, k.T, precision=precision)
        dqk = pl.dot(dq, k.T, precision=precision) + pl.dot(
            q, dk.T, precision=precision
        )
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
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
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

    pallas_store(
        do_ref, (slice(None), slice(None)), val=do.astype(do_ref.dtype), mask=d_mask
    )


def mha_jvp_simple_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    dq_ref: jax.Array,
    dk_ref: jax.Array,
    dv_ref: jax.Array,
    lse_ref: jax.Array,
    do_ref: Any,
    *,
    sm_scale: float,
    head_dim: int,
    block_q: int,
    block_d: int,
    block_k: int,
):
    """JVP kernel specialized for no mask/bias/dropout."""
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    d_mask = jnp.arange(block_d)[None] < head_dim
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    dq = pallas_load(dq_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    LOG2E = 1.4426950408889634  # log2(e)

    lse = pallas_load(lse_ref, (curr_q_slice,))
    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, do_acc):
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pallas_load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pallas_load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = pl.dot(q, k.T, precision=precision)
        dqk = pl.dot(dq, k.T, precision=precision) + pl.dot(
            q, dk.T, precision=precision
        )
        if sm_scale != 1.0:
            qk *= sm_scale
            dqk *= sm_scale

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        row_sum = jnp.sum(dqk * p, axis=-1)
        dP = p * (dqk - row_sum[:, None])

        do_acc = do_acc + pl.dot(dP.astype(v.dtype), v, precision=precision)
        do_acc = do_acc + pl.dot(p.astype(v.dtype), dv, precision=precision)
        return do_acc

    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)
    do = lax.fori_loop(lower_bound, upper_bound, body_jvp, do)
    pallas_store(
        do_ref, (slice(None), slice(None)), val=do.astype(do_ref.dtype), mask=d_mask
    )


def mha_forward_jvp_simple_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    dq_ref: jax.Array,
    dk_ref: jax.Array,
    dv_ref: jax.Array,
    o_ref: Any,
    do_ref: Any,
    *,
    sm_scale: float,
    head_dim: int,
    block_q: int,
    block_d: int,
    block_k: int,
):
    """Fused forward + JVP for no mask/bias/dropout."""
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)
    precision = get_dot_precision(jax.default_backend(), q_ref.dtype)

    d_mask = jnp.arange(block_d)[None] < head_dim
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pallas_load(q_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    dq = pallas_load(dq_ref, (slice(None), slice(None)), mask=d_mask, other=0.0)
    LOG2E = 1.4426950408889634  # log2(e)

    m_i = jnp.full((block_q,), NEG_INF, dtype=jnp.float32)
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_fwd(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        qk = pl.dot(q, k.T, precision=precision)
        if sm_scale != 1.0:
            qk *= sm_scale
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
        o_curr = pl.dot(s_curr.astype(v.dtype), v, precision=precision)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    upper_bound = pl.cdiv(seq_len, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body_fwd, (o, m_i, l_i))

    l_i = jnp.where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]
    lse = m_i + jnp.log2(l_i)

    do = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    def body_jvp(start_k, do_acc):
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dk = pallas_load(dk_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)
        dv = pallas_load(dv_ref, (curr_k_slice, slice(None)), mask=d_mask, other=0.0)

        qk = pl.dot(q, k.T, precision=precision)
        dqk = pl.dot(dq, k.T, precision=precision) + pl.dot(
            q, dk.T, precision=precision
        )
        if sm_scale != 1.0:
            qk *= sm_scale
            dqk *= sm_scale
        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        row_sum = jnp.sum(dqk * p, axis=-1)
        dP = p * (dqk - row_sum[:, None])

        do_acc = do_acc + pl.dot(dP.astype(v.dtype), v, precision=precision)
        do_acc = do_acc + pl.dot(p.astype(v.dtype), dv, precision=precision)
        return do_acc

    do = lax.fori_loop(0, upper_bound, body_jvp, do)
    pallas_store(
        o_ref, (slice(None), slice(None)), val=o.astype(o_ref.dtype), mask=d_mask
    )
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
        qk = pl.dot(q, k.T, precision=precision)
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

        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=d_mask, other=jnp.nan)
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            s_curr = jnp.where(dmask, 0, s_curr / (1 - dropout_rate))
        o_curr = pl.dot(s_curr.astype(v.dtype), v, precision=precision)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    lower_bound = 0
    upper_bound = pl.cdiv(seq_len, block_k)
    if index_offset_size_ref is not None:
        iters = index_offset_size_ref[...]
        o, m_i, l_i = lax.fori_loop(lower_bound, iters, body_fwd, (o, m_i, l_i))
    else:
        o, m_i, l_i = lax.fori_loop(lower_bound, upper_bound, body_fwd, (o, m_i, l_i))

    l_i = jnp.where(l_i == 0.0, 1, l_i)
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

        qk = pl.dot(q, k.T, precision=precision)
        dqk = pl.dot(dq, k.T, precision=precision) + pl.dot(
            q, dk.T, precision=precision
        )
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
                dmask = pallas_load(dropout_mask_ref, (slice(None), curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
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
        qk = pl.dot(q, k.T)
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
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
        # No built-in causal; pass as mask via mask if needed.

        qk *= LOG2E
        lse = pallas_load(lse_ref, (curr_q_slice,))
        di = pallas_load(delta_ref, (curr_q_slice,))
        do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)

        # Apply dropout scaling for forward consistency
        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # p_drop for dV accumulation (forward consistency)
            p_drop = jnp.where(dmask, 0, p / (1 - dropout_rate))
            # g_drop for softmax Jacobian computation
            g_drop = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            # No dropout case
            p_drop = p
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        # Accumulate dV: use dropout-scaled probabilities
        dv = dv + pl.dot(p_drop.astype(do.dtype).T, do)
        # Softmax Jacobian: use ORIGINAL probabilities (not dropout-scaled)
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
            start_q = jnp.sum(pallas_load(kv_index_offset_ref, (pl.dslice(iter_q, 1),)))
            return inner_loop_dkdv(start_q, carry)

        dv, dk = lax.fori_loop(0, iters, dyn_q, (dv, dk))
    else:
        dv, dk = lax.fori_loop(
            0, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
        )

    dv_ref = pallas_store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=mask_d
    )
    dk_ref = pallas_store(
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

    q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=mask_d, other=0.0)
    # segment ids not used in this kernel
    lse = pallas_load(lse_ref, (curr_q_slice,))
    do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pallas_load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)
        v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=mask_d, other=0.0)

        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale
        qk_pre_mod = qk

        if (bias_fn is not None) or (mask_fn is not None) or (b_ref is not None):
            span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
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
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
        # No built-in causal; pass as mask via mask if needed.

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # g_drop for softmax Jacobian computation
            g_drop = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate))
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
            start_k = jnp.sum(pallas_load(q_index_offset_ref, (pl.dslice(iter_k, 1),)))
            return inner_loop_dq(start_k, dq_c)

        dq = lax.fori_loop(0, iters, dyn_k, dq)
    else:
        dq = lax.fori_loop(0, pl.cdiv(kv_seq_len, block_kv_dq), inner_loop_dq, dq)

    pallas_store(
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

    v = pallas_load(v_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)
    k = pallas_load(k_ref, (curr_k_slice, slice(None)), mask=block_mask, other=0.0)

    if id_k_ref is not None:
        id_k = _load_mask_data(id_k_ref, curr_k_slice)
    elif id_q_ref is not None:
        id_k = _load_mask_data(id_q_ref, curr_k_slice)
    else:
        id_k = None

    LOG2E = 1.4426950408889634

    def inner_loop(start_q, carry):
        if index_offset_ref is not None:
            start_q = jnp.sum(pallas_load(index_offset_ref, (pl.dslice(start_q, 1),)))
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
        dv_acc, dk_acc = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)
        q = pallas_load(q_ref, (curr_q_slice, slice(None)), mask=block_mask, other=0.0)
        qk = pl.dot(q, k.T)
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
                qk = jnp.where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        lse = pallas_load(lse_ref, (curr_q_slice,))
        di = pallas_load(delta_ref, (curr_q_slice,))
        do = pallas_load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # p_drop for dV accumulation (forward consistency)
            p_drop = jnp.where(dmask, 0, p / (1 - dropout_rate))
            # g_drop for softmax Jacobian computation
            g_drop = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate))
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + g_drop
        else:
            # No dropout case
            p_drop = p
            dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
            dp = dp + dp_dropped

        # Accumulate dV: use dropout-scaled probabilities
        dv_acc = dv_acc + pl.dot(p_drop.astype(do.dtype).T, do)
        # Softmax Jacobian: use ORIGINAL probabilities (not dropout-scaled)
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

    pallas_store(
        dv_ref, (slice(None), slice(None)), val=dv.astype(dv_ref.dtype), mask=block_mask
    )
    pallas_store(
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
        qk = pl.dot(q, k.T)
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
                qk = jnp.where(m, qk, DEFAULT_MASK_VALUE)

        qk *= LOG2E
        p = jnp.exp2(qk - lse[:, None])
        dp_dropped = pl.dot(do, v.T)

        if dropout_rate > 0:
            if dropout_mask_ref is not None:
                dmask = pallas_load(dropout_mask_ref, (curr_q_slice, curr_k_slice))
            else:
                rng_seed = pallas_load(rng_ref, ())
                dmask = _dropout_mask_counter(
                    rng_seed, start_b, start_h, span_q, span_k, dropout_rate
                )
            # g_drop for softmax Jacobian computation
            g_drop = jnp.where(dmask, 0, dp_dropped / (1 - dropout_rate))
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
    if b_data is not None:
        b_spec = bias.get_block_spec(
            q_len=q_seq_len, kv_len=kv_seq_len, block_q=block_q, block_kv=block_k
        )
    q_id_spec = k_id_spec = None
    if q_id is not None or k_id is not None:
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

    in_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
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

    in_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
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


def _mha_impl_fused_jvp_simple(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_d = pl.next_power_of_2(head_dim)
    grid_ = grid or (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps_ = num_warps or (4 if block_d <= 64 else 8)

    kernel = functools.partial(
        mha_forward_jvp_simple_kernel,
        sm_scale=sm_scale,
        head_dim=head_dim,
        block_q=block_q,
        block_k=block_k,
        block_d=block_d,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
        pl.BlockSpec((None, kv_seq_len, None, block_d), lambda _, j, k_: (j, 0, k_, 0)),
    ]

    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
    ]
    out_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
    ]

    out, tangent_out = pl.pallas_call(
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
        name="mha_forward_jvp_simple",
    )(q, k, v, dq, dk, dv)

    return out, tangent_out


def _mha_impl_fused_jvp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    mask: AttentionMask | None,
    bias: AttentionBias | None,
    rng: jax.Array | None,
    rng_seed: jax.Array,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    dropout_impl: str,
):
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
    effective_num_stages = _resolve_attention_num_stages(
        num_stages, has_dense_bias=b_data is not None
    )

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
        mha_forward_jvp_kernel,
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

    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
    ]
    out_specs = [
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
        pl.BlockSpec((None, block_q, None, block_d), lambda i, j, k_: (j, i, k_, 0)),
    ]

    out, tangent_out = pl.pallas_call(
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
        name="mha_forward_jvp",
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
        index_offset,
        index_offset_size,
    )

    return out, tangent_out


def _flatten_optional_pytree(value):
    if value is None:
        return None, ()
    leaves, treedef = jax.tree_util.tree_flatten(value)
    return treedef, tuple(leaves)


def _unflatten_optional_pytree(treedef, leaves):
    if treedef is None:
        return None
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _sentinel() -> jax.Array:
    """Dummy scalar used as positional arg placeholder for ``None`` arrays
    in ``custom_partitioning``-wrapped functions (which require all positional
    args to be arrays)."""
    return jnp.zeros((), jnp.float32)


def _or_sentinel(x):
    """Return *x* if it is an array, otherwise a scalar sentinel."""
    return x if x is not None else _sentinel()


def _undo_sentinel(x, present: bool):
    """Return *x* if *present*, otherwise ``None``."""
    return x if present else None


# -- Sharding-rule helpers for extracted-array CP scheme -------------------

# Unique replicated factor names per extracted-array slot.  Every dimension
# of every data array is replicated so that the GSPMD partitioner never
# shards them.  The names are deliberately obscure to avoid collision.
_FWD_DATA_RULES = {
    # name            ndim  rule_when_present       factors_when_present
    "b_data": (4, "eb0 eb1 eb2 eb3", ("eb0", "eb1", "eb2", "eb3")),
    "q_id": (4, "eq0 eq1 eq2 eq3", ("eq0", "eq1", "eq2", "eq3")),
    "k_id": (4, "ek0 ek1 ek2 ek3", ("ek0", "ek1", "ek2", "ek3")),
    "dropout_mask": (4, "ed0 ed1 ed2 ed3", ("ed0", "ed1", "ed2", "ed3")),
    "index_offset": (2, "eo0 eo1", ("eo0", "eo1")),
    "index_offset_size": (1, "es0", ("es0",)),
}

_BWD_DATA_RULES = {
    "b_data": (4, "eb0 eb1 eb2 eb3", ("eb0", "eb1", "eb2", "eb3")),
    "q_data": (4, "eq0 eq1 eq2 eq3", ("eq0", "eq1", "eq2", "eq3")),
    "k_data": (4, "ek0 ek1 ek2 ek3", ("ek0", "ek1", "ek2", "ek3")),
    "dropout_mask": (4, "ed0 ed1 ed2 ed3", ("ed0", "ed1", "ed2", "ed3")),
    "q_index_offset": (2, "eo0 eo1", ("eo0", "eo1")),
    "q_index_offset_size": (1, "es0", ("es0",)),
    "kv_index_offset": (2, "ko0 ko1", ("ko0", "ko1")),
    "kv_index_offset_size": (1, "ks0", ("ks0",)),
}


def _data_rule_parts(
    data_rules: dict,
    present_flags: dict[str, bool],
) -> tuple[list[str], list[str]]:
    """Return (rule_fragments, replication_factors) for extracted data arrays.

    For each array slot: if *present*, use its multi-dim rule and factors;
    if absent (sentinel scalar), use ``""`` and no factors.
    """
    parts: list[str] = []
    factors: list[str] = []
    for name, (_ndim, rule, facs) in data_rules.items():
        if present_flags.get(name, False):
            parts.append(rule)
            factors.extend(facs)
        else:
            parts.append("")  # scalar sentinel
    return parts, factors


def _build_mha_sharding_rule_fwd(
    present_flags: dict[str, bool],
    *,
    include_rng: bool,
    output_activations: bool,
) -> tuple[str, tuple[str, ...]]:
    """Build sharding rule + replication factors for the CP forward function.

    Positional args: q(B,T,H,D) k(B,T,H,D) v(B,T,H,D) rng_seed()
        b_data q_id k_id dropout_mask index_offset index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
    ]
    if include_rng:
        parts.append("")  # rng_seed ()
    data_parts, data_factors = _data_rule_parts(_FWD_DATA_RULES, present_flags)
    parts.extend(data_parts)

    if output_activations:
        out_rule = "batch seq heads head_dim, batch heads seq"
    else:
        out_rule = "batch seq heads head_dim"

    rule = ", ".join(parts) + " -> " + out_rule
    repl = tuple(["seq", "head_dim"] + data_factors)
    return rule, repl


def _build_mha_sharding_rule_jvp(
    present_flags: dict[str, bool],
) -> tuple[str, tuple[str, ...]]:
    """Build sharding rule + replication factors for the CP JVP function.

    Positional args: q k v dq dk dv lse rng_seed
        b_data q_id k_id dropout_mask index_offset index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
        "batch seq heads head_dim",  # dq
        "batch seq heads head_dim",  # dk
        "batch seq heads head_dim",  # dv
        "batch heads seq",  # lse
        "",  # rng_seed ()
    ]
    data_parts, data_factors = _data_rule_parts(_FWD_DATA_RULES, present_flags)
    parts.extend(data_parts)
    rule = ", ".join(parts) + " -> batch seq heads head_dim"
    repl = tuple(["seq", "head_dim"] + data_factors)
    return rule, repl


def _build_mha_sharding_rule_bwd(
    present_flags: dict[str, bool],
) -> tuple[str, tuple[str, ...]]:
    """Build sharding rule + replication factors for the CP backward function.

    Positional args: do q k v rng_seed out lse
        b_data q_data k_data dropout_mask
        q_index_offset q_index_offset_size kv_index_offset kv_index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # do
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
        "",  # rng ()
        "batch seq heads head_dim",  # out
        "batch heads seq",  # lse
    ]
    data_parts, data_factors = _data_rule_parts(_BWD_DATA_RULES, present_flags)
    parts.extend(data_parts)
    rule = (
        ", ".join(parts) + " -> batch seq heads head_dim, "
        "batch seq heads head_dim, "
        "batch seq heads head_dim"
    )
    repl = tuple(["seq", "head_dim"] + data_factors)
    return rule, repl


# -- CP factory functions wrapping raw kernels -----------------------------


def _make_cp_mha_fwd(
    *,
    has_b_data: bool,
    has_q_id: bool,
    has_k_id: bool,
    has_dropout_mask: bool,
    has_index: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    output_activations: bool = False,
    # Keyword-only specs/callables (safe for closures — not arrays)
    mask_fn,
    bias_fn,
    b_spec,
    q_id_spec,
    k_id_spec,
):
    """Build a ``custom_partitioning``-wrapped MHA forward function.

    Returns ``f(q, k, v, rng_seed, b_data, q_id, k_id, dropout_mask,
    index_offset, index_offset_size)`` where absent arrays are scalar
    sentinels.  Calls ``_mha_impl_raw`` per-shard.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    present_flags = dict(
        b_data=has_b_data,
        q_id=has_q_id,
        k_id=has_k_id,
        dropout_mask=has_dropout_mask,
        index_offset=has_index,
        index_offset_size=has_index,
    )
    rule, repl = _build_mha_sharding_rule_fwd(
        present_flags,
        include_rng=True,
        output_activations=output_activations,
    )

    def _call_raw(
        q,
        k,
        v,
        rng_seed,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        index_offset,
        index_offset_size,
    ):
        result = _mha_impl_raw(
            q,
            k,
            v,
            rng_seed,
            _undo_sentinel(b_data, has_b_data),
            _undo_sentinel(q_id, has_q_id),
            _undo_sentinel(k_id, has_k_id),
            _undo_sentinel(dropout_mask, has_dropout_mask),
            _undo_sentinel(index_offset, has_index),
            _undo_sentinel(index_offset_size, has_index),
            mask_fn=mask_fn,
            bias_fn=bias_fn,
            b_spec=b_spec,
            q_id_spec=q_id_spec,
            k_id_spec=k_id_spec,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
            output_activations=output_activations,
        )
        if output_activations:
            out, (_q, _k, _v, _seed, _out_res, lse) = result
            return out, lse
        return result

    @custom_partitioning
    def _fwd(
        q,
        k,
        v,
        rng_seed,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        index_offset,
        index_offset_size,
    ):
        return _call_raw(
            q,
            k,
            v,
            rng_seed,
            b_data,
            q_id,
            k_id,
            dropout_mask,
            index_offset,
            index_offset_size,
        )

    def _partition(mesh, arg_shapes, result_shape):
        flat_arg_shapes = jax.tree.leaves(arg_shapes)
        for shape, name in zip(flat_arg_shapes[:3], ("q", "k", "v")):
            _validate_mha_sharding(shape.sharding, name)
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(
            q,
            k,
            v,
            rng_seed,
            b_data,
            q_id,
            k_id,
            dropout_mask,
            index_offset,
            index_offset_size,
        ):
            return _call_raw(
                q,
                k,
                v,
                rng_seed,
                b_data,
                q_id,
                k_id,
                dropout_mask,
                index_offset,
                index_offset_size,
            )

        return mesh, lower_fn, result_shardings, arg_shardings

    _fwd.def_partition(
        partition=_partition,
        sharding_rule=rule,
        need_replication_factors=repl,
    )
    return _fwd


def _make_cp_mha_jvp(
    *,
    has_b_data: bool,
    has_q_id: bool,
    has_k_id: bool,
    has_dropout_mask: bool,
    has_index: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    # Keyword-only specs/callables
    mask_fn,
    bias_fn,
    bias_fn_grad,
    b_spec,
    q_id_spec,
    k_id_spec,
):
    """Build a ``custom_partitioning``-wrapped MHA JVP-from-LSE function.

    Returns ``f(q, k, v, dq, dk, dv, lse, rng_seed,
    b_data, q_id, k_id, dropout_mask, index_offset, index_offset_size)
    -> tangent_out``.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    present_flags = dict(
        b_data=has_b_data,
        q_id=has_q_id,
        k_id=has_k_id,
        dropout_mask=has_dropout_mask,
        index_offset=has_index,
        index_offset_size=has_index,
    )
    rule, repl = _build_mha_sharding_rule_jvp(present_flags)

    def _call_raw(
        q,
        k,
        v,
        dq,
        dk,
        dv,
        lse,
        rng_seed,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        index_offset,
        index_offset_size,
    ):
        return _mha_impl_jvp_from_lse_raw(
            q,
            k,
            v,
            dq,
            dk,
            dv,
            lse,
            rng_seed,
            _undo_sentinel(b_data, has_b_data),
            _undo_sentinel(q_id, has_q_id),
            _undo_sentinel(k_id, has_k_id),
            _undo_sentinel(dropout_mask, has_dropout_mask),
            _undo_sentinel(index_offset, has_index),
            _undo_sentinel(index_offset_size, has_index),
            mask_fn=mask_fn,
            bias_fn=bias_fn,
            bias_fn_grad=bias_fn_grad,
            b_spec=b_spec,
            q_id_spec=q_id_spec,
            k_id_spec=k_id_spec,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )

    @custom_partitioning
    def _jvp_fn(
        q,
        k,
        v,
        dq,
        dk,
        dv,
        lse,
        rng_seed,
        b_data,
        q_id,
        k_id,
        dropout_mask,
        index_offset,
        index_offset_size,
    ):
        return _call_raw(
            q,
            k,
            v,
            dq,
            dk,
            dv,
            lse,
            rng_seed,
            b_data,
            q_id,
            k_id,
            dropout_mask,
            index_offset,
            index_offset_size,
        )

    def _partition(mesh, arg_shapes, result_shape):
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(
            q,
            k,
            v,
            dq,
            dk,
            dv,
            lse,
            rng_seed,
            b_data,
            q_id,
            k_id,
            dropout_mask,
            index_offset,
            index_offset_size,
        ):
            return _call_raw(
                q,
                k,
                v,
                dq,
                dk,
                dv,
                lse,
                rng_seed,
                b_data,
                q_id,
                k_id,
                dropout_mask,
                index_offset,
                index_offset_size,
            )

        return mesh, lower_fn, result_shardings, arg_shardings

    _jvp_fn.def_partition(
        partition=_partition,
        sharding_rule=rule,
        need_replication_factors=repl,
    )
    return _jvp_fn


def _make_cp_mha_bwd(
    *,
    has_b_data: bool,
    has_q_data: bool,
    has_k_data: bool,
    has_dropout_mask: bool,
    has_q_index: bool,
    has_kv_index: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    interpret: bool,
    debug: bool,
    dropout_rate: float,
    # Keyword-only specs/callables
    mask_fn,
    bias_fn,
    bias_fn_grad,
    b_spec_bwd,
    q_data_spec,
    k_data_spec,
):
    """Build a ``custom_partitioning``-wrapped MHA backward function.

    Returns ``f(do, q, k, v, rng_seed, out, lse,
    b_data, q_data, k_data, dropout_mask,
    q_index_offset, q_index_offset_size,
    kv_index_offset, kv_index_offset_size) -> (dq, dk, dv)``.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    present_flags = dict(
        b_data=has_b_data,
        q_data=has_q_data,
        k_data=has_k_data,
        dropout_mask=has_dropout_mask,
        q_index_offset=has_q_index,
        q_index_offset_size=has_q_index,
        kv_index_offset=has_kv_index,
        kv_index_offset_size=has_kv_index,
    )
    rule, repl = _build_mha_sharding_rule_bwd(present_flags)

    def _call_raw(
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
    ):
        dq, dk, dv = _mha_backward_raw(
            do,
            q,
            k,
            v,
            rng_seed,
            out,
            lse,
            _undo_sentinel(b_data, has_b_data),
            _undo_sentinel(q_data, has_q_data),
            _undo_sentinel(k_data, has_k_data),
            _undo_sentinel(dropout_mask, has_dropout_mask),
            _undo_sentinel(q_index_offset, has_q_index),
            _undo_sentinel(q_index_offset_size, has_q_index),
            _undo_sentinel(kv_index_offset, has_kv_index),
            _undo_sentinel(kv_index_offset_size, has_kv_index),
            mask_fn=mask_fn,
            bias_fn=bias_fn,
            bias_fn_grad=bias_fn_grad,
            b_spec_bwd=b_spec_bwd,
            q_data_spec=q_data_spec,
            k_data_spec=k_data_spec,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            backward_pass_impl=backward_pass_impl,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
        )
        return dq, dk, dv

    @custom_partitioning
    def _bwd(
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
    ):
        return _call_raw(
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
        )

    def _partition(mesh, arg_shapes, result_shape):
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(
            do,
            q,
            k,
            v,
            rng,
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
        ):
            return _call_raw(
                do,
                q,
                k,
                v,
                rng,
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
            )

        return mesh, lower_fn, result_shardings, arg_shardings

    _bwd.def_partition(
        partition=_partition,
        sharding_rule=rule,
        need_replication_factors=repl,
    )
    return _bwd


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
    diff_mode: str = "reverse",
):
    """Multi-Head Attention public API.

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

    # Flatten mask/bias pytrees: array leaves become regular traced args,
    # treedefs are truly static metadata.
    mask_treedef, mask_leaves = _flatten_optional_pytree(mask)
    bias_treedef, bias_leaves = _flatten_optional_pytree(bias)

    force_forward_mode = diff_mode == "forward" or _is_direct_jvp_trace(
        q, k, v, rng, mask_leaves, bias_leaves
    )
    factory = _make_mha_forward_jvp if force_forward_mode else _make_mha_custom_vjp
    inner = factory(
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
    return inner(q, k, v, rng, mask_leaves, bias_leaves)


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

        cp_fwd = _make_cp_mha_fwd(
            has_b_data=arrays["b_data"] is not None,
            has_q_id=arrays["q_id"] is not None,
            has_k_id=arrays["k_id"] is not None,
            has_dropout_mask=arrays["dropout_mask"] is not None,
            has_index=arrays["index_offset"] is not None,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
            output_activations=output_activations,
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            b_spec=specs["b_spec"],
            q_id_spec=specs["q_id_spec"],
            k_id_spec=specs["k_id_spec"],
        )
        try:
            result = cp_fwd(
                q,
                k,
                v,
                rng_seed,
                _or_sentinel(arrays["b_data"]),
                _or_sentinel(arrays["q_id"]),
                _or_sentinel(arrays["k_id"]),
                _or_sentinel(arrays["dropout_mask"]),
                _or_sentinel(arrays["index_offset"]),
                _or_sentinel(arrays["index_offset_size"]),
            )
            if output_activations:
                out, lse = result
                return out, rng_val, rng_seed, lse
            return result
        except AssertionError:
            pass

        result = _mha_impl_raw(
            q,
            k,
            v,
            rng_seed,
            arrays["b_data"],
            arrays["q_id"],
            arrays["k_id"],
            arrays["dropout_mask"],
            arrays["index_offset"],
            arrays["index_offset_size"],
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
            output_activations=output_activations,
        )
        if output_activations:
            out, (_q, _k, _v, _seed, _out_res, lse) = result
            return out, rng_val, rng_seed, lse
        return result

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

        cp_bwd = _make_cp_mha_bwd(
            has_b_data=arrays["b_data"] is not None,
            has_q_data=arrays["q_data"] is not None,
            has_k_data=arrays["k_data"] is not None,
            has_dropout_mask=arrays["dropout_mask"] is not None,
            has_q_index=arrays["q_index_offset"] is not None,
            has_kv_index=arrays["kv_index_offset"] is not None,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            backward_pass_impl=backward_pass_impl,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
            mask_fn=specs["mask_fn"],
            bias_fn=specs["bias_fn"],
            bias_fn_grad=specs["bias_fn_grad"],
            b_spec_bwd=specs["b_spec_bwd"],
            q_data_spec=specs["q_data_spec"],
            k_data_spec=specs["k_data_spec"],
        )
        try:
            return cp_bwd(
                do,
                q,
                k,
                v,
                rng_seed,
                out,
                lse,
                _or_sentinel(arrays["b_data"]),
                _or_sentinel(arrays["q_data"]),
                _or_sentinel(arrays["k_data"]),
                _or_sentinel(arrays["dropout_mask"]),
                _or_sentinel(arrays["q_index_offset"]),
                _or_sentinel(arrays["q_index_offset_size"]),
                _or_sentinel(arrays["kv_index_offset"]),
                _or_sentinel(arrays["kv_index_offset_size"]),
            )
        except AssertionError:
            pass

        return _mha_backward_raw(
            do,
            q,
            k,
            v,
            rng_seed,
            out,
            lse,
            arrays["b_data"],
            arrays["q_data"],
            arrays["k_data"],
            arrays["dropout_mask"],
            arrays["q_index_offset"],
            arrays["q_index_offset_size"],
            arrays["kv_index_offset"],
            arrays["kv_index_offset_size"],
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

        try:
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

            cp_jvp = _make_cp_mha_jvp(
                has_b_data=arrays["b_data"] is not None,
                has_q_id=arrays["q_id"] is not None,
                has_k_id=arrays["k_id"] is not None,
                has_dropout_mask=arrays["dropout_mask"] is not None,
                has_index=arrays["index_offset"] is not None,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
                num_warps=num_warps,
                num_stages=num_stages,
                grid=grid,
                interpret=interpret,
                debug=debug,
                dropout_rate=dropout_rate,
                mask_fn=specs["mask_fn"],
                bias_fn=specs["bias_fn"],
                bias_fn_grad=bias_fn_grad,
                b_spec=specs["b_spec"],
                q_id_spec=specs["q_id_spec"],
                k_id_spec=specs["k_id_spec"],
            )
            tangent_out = cp_jvp(
                q,
                k,
                v,
                dq,
                dk,
                dv,
                lse_res,
                rng_seed,
                _or_sentinel(arrays["b_data"]),
                _or_sentinel(arrays["q_id"]),
                _or_sentinel(arrays["k_id"]),
                _or_sentinel(arrays["dropout_mask"]),
                _or_sentinel(arrays["index_offset"]),
                _or_sentinel(arrays["index_offset_size"]),
            )
            return out, tangent_out
        except AssertionError:
            pass

        if mask is None and bias is None and dropout_rate == 0.0:
            return _mha_impl_fused_jvp_simple(
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
                num_warps=num_warps,
                num_stages=num_stages,
                grid=grid,
                interpret=interpret,
                debug=debug,
            )
        rng_val = rng if rng is not None else jax.random.key(0)
        rng_seed = _rng_seed_from_key(rng_val)
        return _mha_impl_fused_jvp(
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            mask=mask,
            bias=bias,
            rng=rng_val,
            rng_seed=rng_seed,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            num_warps=num_warps,
            num_stages=num_stages,
            grid=grid,
            interpret=interpret,
            debug=debug,
            dropout_rate=dropout_rate,
            dropout_impl=dropout_impl,
        )

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

    b_spec_bwd = None
    if b_data is not None:
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
    block_d = pl.next_power_of_2(head_dim)
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
    block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
    block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
    block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)
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
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]

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
                (None, kv_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
            ),
            pl.BlockSpec(
                (None, kv_seq_len, None, block_d), lambda i, j, _: (i, 0, j, 0)
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

    return dq.astype(q.dtype), dk, dv
