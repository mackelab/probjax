from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, Callable

__all__ = [
    "AttentionMaskBase",
    "AttentionBiasBase",
    "MaskModFn",
    "ScoreModFn",
    "CausalMask",
    "LocalWindowMask",
    "KeyPaddingMask",
    "SameSegmentMask",
    "NoMask",
    "FromMaskBias",
    "ConstantBias",
    # Stateless bias functions
    "bias_identity",
    "bias_causal",
    "bias_alibi",
    "bias_distance_decay",
    # Layer-style bias generators
    "AttentionLogitBiasLayer",
    "CausalAttentionLogitBiasLayer",
    "FullAttentionLogitBiasLayer",
    "ALiBiAttentionLogitBiasLayer",
    "SymmetricALiBiAttentionLogitBiasLayer",
    # Helpers mirroring AXLearn utilities
    "compute_padding_biases",
    "make_segment_bias",
    "apply_attention_logit_biases",
    "alibi_get_slopes",
    # Block sparsity helpers
    "compute_block_mask",
    "compute_block_iterators",
    "compute_kv_iterators",
    "compute_block_bounds",
    "ceil_div",
    # FlashAttention-compatible mask adapters
    "FlashMaskFn",
    "to_flash_mask_fn",
    "flash_causal_mask_fn",
    "flash_local_window_mask_fn",
    "materialize_mask",
    "materialize_bias",
]


# A very negative value used to mask logits safely in float32.
# Keep consistent with pallas_kernels/flash_attention.py
NEG_INF = -1e15
DEFAULT_MASK_VALUE = NEG_INF


# ----------------------------- Utilities ------------------------------------

# Function type aliases following project typing conventions.
MaskModFn = Callable[
    [Array, Array, Array, Array, Optional[Array], Optional[Array]],
    Array,
]
ScoreModFn = Callable[[Array, Array, Array, Array, Array], Array]

# FlashAttention's mask function signature: given absolute row/col indices within
# a tile, return a boolean matrix where True means "keep" (not masked).
FlashMaskFn = Callable[[Array, Array], Array]

def _ensure_tuple_segment_ids(
    segment_ids: Optional[Tuple[Array, Array] | Array],
) -> Tuple[Optional[Array], Optional[Array]]:
    if segment_ids is None:
        return None, None
    if isinstance(segment_ids, tuple):
        seg_q, seg_k = segment_ids
        return seg_q, seg_k
    return segment_ids, segment_ids


# ------------------------ Block Sparsity Helpers -----------------------------


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return -(-int(a) // int(b))


def _block_pair_has_any(
    mask_mod_fn: MaskModFn,
    b_idx: Array,
    h_idx: Array,
    q_start: int,
    k_start: int,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_k: int,
    seg_q: Optional[Array],
    seg_k: Optional[Array],
) -> Array:
    """Returns True if the block tile [q_start:q_start+block_q, k_start:k_start+block_k]
    contains any valid (unmasked) positions for this (b_idx, h_idx).

    This implementation is JIT-friendly: it clips indices to be in-bounds and masks
    out-of-range rows/cols after evaluating the mask function.
    """
    # Build per-block indices, clipped to avoid OOB indexing inside mask functions.
    q_rel = jnp.arange(block_q)
    k_rel = jnp.arange(block_k)
    q_idx = jnp.clip(q_start + q_rel, 0, q_len - 1)
    k_idx = jnp.clip(k_start + k_rel, 0, kv_len - 1)
    base = mask_mod_fn(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k)
    valid_q = (q_start + q_rel) < q_len
    valid_k = (k_start + k_rel) < kv_len
    base = base & valid_q[:, None] & valid_k[None, :]
    return jnp.any(base)


def _fast_blockmask_local_window(
    *, q_len: int, kv_len: int, block_q: int, block_k: int, left_window: int, right_window: int
) -> Array:
    """Fast block mask construction for LocalWindowMask.

    A (q_block, kv_block) pair is non-empty iff the intervals intersect:
        [j0, j1] intersects [i0 - left_window, i1 + right_window]
    where i0,i1 are q block bounds, j0,j1 are kv block bounds.
    """
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    i = jnp.arange(num_q_blocks)
    j = jnp.arange(num_kv_blocks)
    i0 = i * block_q
    i1 = jnp.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    j1 = jnp.minimum((j + 1) * block_k - 1, kv_len - 1)
    # Broadcast to [num_q_blocks, num_kv_blocks]
    cond_left = jnp.expand_dims(j0, 0) <= jnp.expand_dims(i1 + right_window, 1)
    cond_right = jnp.expand_dims(j1, 0) >= jnp.expand_dims(i0 - left_window, 1)
    return cond_left & cond_right


def _fast_blockmask_causal(
    *, q_len: int, kv_len: int, block_q: int, block_k: int
) -> Array:
    """Fast block mask for causal mask (allow k <= q).

    A (q_block, kv_block) pair is non-empty iff the kv block start <= q block end.
    """
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    i = jnp.arange(num_q_blocks)
    j = jnp.arange(num_kv_blocks)
    i1 = jnp.minimum((i + 1) * block_q - 1, q_len - 1)
    j0 = j * block_k
    return (jnp.expand_dims(j0, 0) <= jnp.expand_dims(i1, 1))


def compute_block_mask(
    mask: AttentionMaskBase | MaskModFn,
    *,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    block_q: int,
    block_k: int,
    segment_ids: Optional[Tuple[Array, Array] | Array] = None,
) -> Array:
    """Compute per-(B,H) block masks of shape [B, H, nQB, nKB].

    If `mask` is an `AttentionMaskBase`, delegates to its `block_mask` method.
    Otherwise treats `mask` as a flex-style `MaskModFn` and uses the generic path.
    """
    num_q_blocks = ceil_div(q_len, block_q)
    num_kv_blocks = ceil_div(kv_len, block_k)
    seg_q, seg_k = _ensure_tuple_segment_ids(segment_ids)

    if isinstance(mask, AttentionMaskBase):
        def per_head(bh: Array) -> Array:
            b_idx, h_idx = bh
            sq = None if seg_q is None else seg_q[b_idx]
            sk = None if seg_k is None else seg_k[b_idx]
            return mask.block_mask(
                b_idx,
                h_idx,
                q_len=q_len,
                kv_len=kv_len,
                block_q=block_q,
                block_k=block_k,
                seg_q=sq,
                seg_k=sk,
            )

        bh = jnp.stack(
            jnp.meshgrid(jnp.arange(batch_size), jnp.arange(num_heads), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
        bm = jax.vmap(per_head)(bh).reshape(batch_size, num_heads, num_q_blocks, num_kv_blocks)
        return bm

    # Generic callable path (MaskModFn)
    q_starts = jnp.arange(0, q_len, block_q)
    k_starts = jnp.arange(0, kv_len, block_k)

    def per_head(bh: Array) -> Array:
        b_idx, h_idx = bh
        sq = None if seg_q is None else seg_q[b_idx]
        sk = None if seg_k is None else seg_k[b_idx]

        def per_q(q_start):
            def per_k(k_start):
                return _block_pair_has_any(
                    mask,  # type: ignore[arg-type]
                    b_idx,
                    h_idx,
                    q_start,
                    k_start,
                    q_len,
                    kv_len,
                    block_q,
                    block_k,
                    sq,
                    sk,
                )

            return jax.vmap(per_k)(k_starts)

        return jax.vmap(per_q)(q_starts)

    bh = jnp.stack(
        jnp.meshgrid(jnp.arange(batch_size), jnp.arange(num_heads), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bm = jax.vmap(per_head)(bh).reshape(batch_size, num_heads, num_q_blocks, num_kv_blocks)
    return bm


def _row_iterators(mask_row: Array) -> tuple[Array, Array]:
    """Given a boolean row [N], return (indices, size) with padding.

    - indices: int32[N], listing indices where mask_row is True, padded with 0s.
    - size: int32 scalar, number of valid entries.
    """
    n = mask_row.shape[0]
    idx = jnp.nonzero(mask_row, size=n, fill_value=0)[0]
    size = mask_row.astype(jnp.int32).sum()
    return idx.astype(jnp.int32), size.astype(jnp.int32)


def compute_block_iterators(block_mask: Array) -> tuple[Array, Array]:
    """Per-(B,H,QB) iterators over non-empty KV blocks.

    Args:
        block_mask: bool [B, H, nQB, nKB]

    Returns:
        kv_block_offset: int32 [B, H, nQB, nKB]
        kv_block_offset_size: int32 [B, H, nQB]
    """
    B, H, nQB, nKB = block_mask.shape
    def per_bh(bh_mask):
        # bh_mask: [nQB, nKB]
        idx, sz = jax.vmap(_row_iterators)(bh_mask)
        return idx, sz

    idx, sz = jax.vmap(jax.vmap(per_bh, in_axes=0), in_axes=0)(block_mask)
    # vmap over B then H returns nested tuples; reshape to [B,H,...]
    return idx, sz


def compute_kv_iterators(block_mask: Array) -> tuple[Array, Array]:
    """Per-(B,H,KB) iterators over non-empty Q blocks.

    Returns:
        q_block_offset: int32 [B, H, nKB, nQB]
        q_block_offset_size: int32 [B, H, nKB]
    """
    # Transpose to reuse the same row logic
    bm_t = jnp.swapaxes(block_mask, -1, -2)  # [B,H,nKB,nQB]
    def per_bh(bh_mask):
        idx, sz = jax.vmap(_row_iterators)(bh_mask)
        return idx, sz
    idx, sz = jax.vmap(jax.vmap(per_bh, in_axes=0), in_axes=0)(bm_t)
    return idx, sz


def compute_block_bounds(
    *, q_len: int, kv_len: int, block_q: int, block_k: int
) -> tuple[Array, Array, Array, Array]:
    """Returns (q_start, q_end, kv_start, kv_end) as int32 arrays.

    - q_start/q_end: shape [nQB]
    - kv_start/kv_end: shape [nKB]
    """
    nQB = ceil_div(q_len, block_q)
    nKB = ceil_div(kv_len, block_k)
    q_start = jnp.arange(nQB, dtype=jnp.int32) * int(block_q)
    q_end = jnp.minimum(q_start + int(block_q) - 1, q_len - 1)
    kv_start = jnp.arange(nKB, dtype=jnp.int32) * int(block_k)
    kv_end = jnp.minimum(kv_start + int(block_k) - 1, kv_len - 1)
    return q_start, q_end, kv_start, kv_end


def materialize_mask(
    mask_mod_fn: MaskModFn,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    *,
    segment_ids: Optional[Tuple[Array, Array] | Array] = None,
) -> Array:
    """Materializes a boolean mask array of shape [B, H, Q, K].

    The callable must follow the flex-attention mask signature:
    (b_idx, h_idx, q_idx, k_idx, seg_q, seg_k) -> [Q, K] boolean.
    """
    q_idx = jnp.arange(q_len)
    k_idx = jnp.arange(kv_len)
    seg_q, seg_k = _ensure_tuple_segment_ids(segment_ids)

    def per_head(bh: Array) -> Array:
        b_idx, h_idx = bh
        sq = None if seg_q is None else seg_q[b_idx]
        sk = None if seg_k is None else seg_k[b_idx]
        return mask_mod_fn(b_idx, h_idx, q_idx, k_idx, sq, sk)

    bh = jnp.stack(
        jnp.meshgrid(jnp.arange(batch_size), jnp.arange(num_heads), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    mk: Array = jax.vmap(per_head)(bh).reshape(batch_size, num_heads, q_len, kv_len)
    return mk


def materialize_bias(
    score_mod_fn: ScoreModFn,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
) -> Array:
    """Materializes an additive bias array [B, H, Q, K] from a score modifier.

    The callable must follow the flex-attention score signature:
    (scores, b_idx, h_idx, q_idx, k_idx) -> scores', where scores has shape [Q, K].
    We pass an all-zeros score matrix and return the modified scores.
    """
    q_idx = jnp.arange(q_len)
    k_idx = jnp.arange(kv_len)

    def per_head(bh: Array) -> Array:
        b_idx, h_idx = bh
        base = jnp.zeros((q_len, kv_len))
        return score_mod_fn(base, b_idx, h_idx, q_idx, k_idx)

    bh = jnp.stack(
        jnp.meshgrid(jnp.arange(batch_size), jnp.arange(num_heads), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bi: Array = jax.vmap(per_head)(bh).reshape(batch_size, num_heads, q_len, kv_len)
    return bi


# ---------------------------- Base Classes -----------------------------------


class AttentionMaskBase(ABC):
    """Base class for attention masks.

    Subclasses implement __call__(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k) -> [Q, K] bool.
    """

    @abstractmethod
    def __call__(
        self,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        raise NotImplementedError

    # Logical combinators return composed masks.
    def __and__(self, other: "AttentionMaskBase") -> "AttentionMaskBase":
        if not isinstance(other, AttentionMaskBase):
            return NotImplemented
        return ComposeMask("and", self, other)

    def __or__(self, other: "AttentionMaskBase") -> "AttentionMaskBase":
        if not isinstance(other, AttentionMaskBase):
            return NotImplemented
        return ComposeMask("or", self, other)

    def __xor__(self, other: "AttentionMaskBase") -> "AttentionMaskBase":
        if not isinstance(other, AttentionMaskBase):
            return NotImplemented
        return ComposeMask("xor", self, other)

    def __invert__(self) -> "AttentionMaskBase":
        return NotMask(self)

    # -------- Block-sparse helpers (override for fast paths) --------

    def block_mask(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        """Default generic per-(b,h) block mask [nQB, nKB].

        Subclasses may override for O(1) formulae (e.g., causal, sliding window).
        """
        num_q_blocks = ceil_div(q_len, block_q)
        num_kv_blocks = ceil_div(kv_len, block_k)
        q_starts = jnp.arange(0, q_len, block_q)
        k_starts = jnp.arange(0, kv_len, block_k)

        def per_q(q_start):
            def per_k(k_start):
                return _block_pair_has_any(
                    lambda b, h, q_idx, k_idx, sqa, ska: self(b, h, q_idx, k_idx, sqa, ska),
                    b_idx,
                    h_idx,
                    q_start,
                    k_start,
                    q_len,
                    kv_len,
                    block_q,
                    block_k,
                    seg_q,
                    seg_k,
                )

            return jax.vmap(per_k)(k_starts)

        return jax.vmap(per_q)(q_starts).reshape(num_q_blocks, num_kv_blocks)

    def block_iterators(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Per-(b,h,q_block) iterators over non-empty KV blocks.

        Returns (kv_block_offset [nQB,nKB], kv_block_offset_size [nQB]).
        """
        bm = self.block_mask(
            b_idx,
            h_idx,
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            seg_q=seg_q,
            seg_k=seg_k,
        )
        idx, sz = jax.vmap(_row_iterators)(bm)
        return idx, sz

    def kv_iterators(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Per-(b,h,kv_block) iterators over non-empty Q blocks.

        Returns (q_block_offset [nKB,nQB], q_block_offset_size [nKB]).
        """
        bm = self.block_mask(
            b_idx,
            h_idx,
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            seg_q=seg_q,
            seg_k=seg_k,
        )
        bm_t = jnp.swapaxes(bm, -1, -2)
        idx, sz = jax.vmap(_row_iterators)(bm_t)
        return idx, sz


class AttentionBiasBase(ABC):
    """Base class for attention score biases.

    Subclasses implement __call__(scores, b_idx, h_idx, q_idx, k_idx) -> [Q, K] scores.
    The bias is additive to the input scores (logits).
    """

    @abstractmethod
    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
    ) -> Array:
        raise NotImplementedError

    # Allow additive composition of biases.
    def __add__(self, other: "AttentionBiasBase") -> "AttentionBiasBase":
        if not isinstance(other, AttentionBiasBase):
            return NotImplemented
        return SumBias(self, other)

    def __radd__(self, other: "AttentionBiasBase") -> "AttentionBiasBase":
        if not isinstance(other, AttentionBiasBase):
            return NotImplemented
        return SumBias(other, self)


# -------------------------- Composition helpers ------------------------------


@dataclass(frozen=True)
class ComposeMask(AttentionMaskBase):
    op: str  # one of: 'and', 'or', 'xor'
    lhs: AttentionMaskBase
    rhs: AttentionMaskBase

    def __call__(
        self,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        a = self.lhs(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k)
        c = self.rhs(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k)
        if self.op == "and":
            return jnp.logical_and(a, c)
        if self.op == "or":
            return jnp.logical_or(a, c)
        if self.op == "xor":
            return jnp.logical_xor(a, c)
        raise ValueError(f"Unknown op for ComposeMask: {self.op}")


@dataclass(frozen=True)
class NotMask(AttentionMaskBase):
    inner: AttentionMaskBase

    def __call__(
        self,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        return jnp.logical_not(self.inner(b_idx, h_idx, q_idx, k_idx, seg_q, seg_k))


@dataclass(frozen=True)
class SumBias(AttentionBiasBase):
    lhs: AttentionBiasBase
    rhs: AttentionBiasBase

    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
    ) -> Array:
        s1 = self.lhs(scores, b_idx, h_idx, q_idx, k_idx)
        s2 = self.rhs(scores, b_idx, h_idx, q_idx, k_idx)
        return s1 + s2


# ------------------------------ Masks ----------------------------------------


class NoMask(AttentionMaskBase):
    """Allows all attention (i.e., returns all True)."""

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        return jnp.ones((q_idx.shape[0], k_idx.shape[0]), dtype=bool)

    def block_mask(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        nQB = ceil_div(q_len, block_q)
        nKB = ceil_div(kv_len, block_k)
        return jnp.ones((nQB, nKB), dtype=jnp.bool_)


class CausalMask(AttentionMaskBase):
    """Standard autoregressive mask: allow attending to past and self only."""

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        return q_idx[:, None] >= k_idx[None, :]

    def block_mask(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        return _fast_blockmask_causal(q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k)


class LocalWindowMask(AttentionMaskBase):
    """Local banded mask allowing attention within a window around each query.

    Args:
        left_window: Max distance to the left (past) allowed.
        right_window: Max distance to the right (future) allowed.
    """

    def __init__(self, left_window: int, right_window: Optional[int] = None):
        self.left_window = int(left_window)
        self.right_window = int(left_window if right_window is None else right_window)

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        dqk = q_idx[:, None] - k_idx[None, :]
        return jnp.logical_and(dqk >= -self.right_window, dqk <= self.left_window)

    def block_mask(
        self,
        b_idx: Array,
        h_idx: Array,
        *,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        return _fast_blockmask_local_window(
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            left_window=self.left_window,
            right_window=self.right_window,
        )


class KeyPaddingMask(AttentionMaskBase):
    """Masks out padded key positions based on key lengths or boolean mask.

    Args:
        key_lengths: int array [B] giving number of valid KV tokens per batch; or
            boolean array [B, K] where True indicates a valid KV position.
    """

    def __init__(self, key_lengths: Array):
        self.key_lengths = key_lengths

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        kl = self.key_lengths
        if kl.dtype == jnp.bool_:  # shape [B, K]
            valid_k = kl[b_idx][k_idx]
        else:  # assume lengths [B]
            valid_k = k_idx < kl[b_idx]
        return jnp.broadcast_to(valid_k[None, :], (q_idx.shape[0], k_idx.shape[0]))


class SameSegmentMask(AttentionMaskBase):
    """Allow attention only within the same segment (uses segment_ids)."""

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        if seg_q is None or seg_k is None:
            raise ValueError("SameSegmentMask requires segment_ids for queries and keys.")
        return seg_q[q_idx][:, None] == seg_k[k_idx][None, :]


# ------------------------------ Biases ----------------------------------------


class FromMaskBias(AttentionBiasBase):
    """Converts a mask into additive bias using a large negative value.

    Positions where mask is False receive `mask_value` (e.g., -1e9), others get 0.
    """

    def __init__(self, mask: AttentionMaskBase, mask_value: float = DEFAULT_MASK_VALUE):
        self.mask = mask
        self.mask_value = mask_value

    def __call__(self, scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
        mk = self.mask(b_idx, h_idx, q_idx, k_idx, None, None)
        return jnp.where(mk, scores, scores + self.mask_value)


class ConstantBias(AttentionBiasBase):
    """Adds a constant bias to all logits."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
        return scores + self.value


# --------------------------- Stateless Bias Fns -------------------------------


def bias_identity(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """No-op bias: returns scores unchanged.

    Useful as a default ScoreModFn when wiring APIs.
    """
    return scores


def bias_causal(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """Causal masking expressed as an additive bias.

    Applies a large negative value to disallow attending to future positions.
    """
    mask = q_idx[:, None] >= k_idx[None, :]
    return jnp.where(mask, scores, scores + DEFAULT_MASK_VALUE)


def _alibi_slope_for_head(h_idx: Array) -> Array:
    """Returns a per-head slope for ALiBi-style linear bias.

    This simplified variant uses geometric decay by head index: 2^(-(h+1)).
    It avoids needing the total number of heads and works with the provided API.
    """
    # Cast to float for exponent, ensure Array dtype is float32-compatible.
    h_f = h_idx.astype(jnp.float32)
    return jnp.exp2(-(h_f + 1.0))


def bias_alibi(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """ALiBi-like linear position bias with head-dependent slope.

    Encourages attending to recent tokens. For causal attention, the distance is
    non-negative (i - j), but we guard with relu to keep distances >= 0.
    bias = -slope(h) * relu(i - j)
    """
    slope = _alibi_slope_for_head(h_idx)
    dist = jnp.maximum(q_idx[:, None] - k_idx[None, :], 0)
    return scores - slope * dist


def bias_distance_decay(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """Symmetric distance penalty independent of direction.

    Adds a negative penalty proportional to absolute distance |i - j|.
    Equivalent to a fixed, head-agnostic linear decay.
    """
    alpha = jnp.array(1.0, dtype=jnp.float32)
    dist = jnp.abs(q_idx[:, None] - k_idx[None, :]).astype(jnp.float32)
    return scores - alpha * dist


def apply_attention_logit_biases(bias_a: Array, bias_b: Array | None) -> Array:
    """Combines two bias tensors, handling None and broadcasting.

    Args:
        bias_a: Array of shape [B, H|1, L, L].
        bias_b: Array of shape [B, H|1, L, L] or None.

    Returns:
        Sum with broadcasting, or bias_a if bias_b is None.
    """
    if bias_b is None:
        return bias_a
    return bias_a + bias_b


def make_segment_bias(source_segments: Array, target_segments: Array) -> Array:
    """Produces -inf outside same nonzero segment, zeros otherwise.

    Args:
        source_segments: int Array [B, L], 0 denotes padding.
        target_segments: int Array [B, L], 0 denotes padding.

    Returns:
        bias Array [B, 1, L, L].
    """
    # Allowed only if both nonzero and equal.
    same = (source_segments[:, None, :] == target_segments[:, :, None])
    nonzero = (source_segments[:, None, :] != 0) & (target_segments[:, :, None] != 0)
    allowed = same & nonzero
    bias = jnp.where(allowed, 0.0, NEG_INF).astype(jnp.float32)
    return bias[:, None, :, :]


def compute_padding_biases(input_ids: Array, *, pad_token_id: int | None) -> Array:
    """Compute logits bias to disable attention to/from paddings.

    Args:
        input_ids: Array [B, L].
        pad_token_id: padded token id or None.

    Returns:
        bias Array [B, 1, L, L].
    """
    batch_size, seq_len = input_ids.shape
    if pad_token_id is None:
        return jnp.zeros([batch_size, 1, seq_len, seq_len], dtype=jnp.float32)
    padding_bias = (input_ids == pad_token_id) * NEG_INF
    return padding_bias[:, None, None, :] + padding_bias[:, None, :, None]


def alibi_get_slopes(num_heads: int) -> Array:
    """ALiBi head slopes as in the paper/codebase, shape [H]."""
    import math

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if num_heads <= 0:
        raise ValueError("num_heads must be positive for alibi slopes")
    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 1 << int(math.floor(math.log2(num_heads)))
        slopes = get_slopes_power_of_2(closest_power_of_2) + alibi_get_slopes(2 * closest_power_of_2)[
            0::2
        ][: num_heads - closest_power_of_2]
    return jnp.asarray(slopes, dtype=jnp.float32)


class AttentionLogitBiasLayer:
    """Base attention logit bias layer.

    forward should produce attention logit biases of shape [B, 1|H, L, L].
    """

    def forward(self, *, segment_ids: Array, positions: Array) -> Array:  # pragma: no cover - API
        raise NotImplementedError(type(self))


class CausalAttentionLogitBiasLayer(AttentionLogitBiasLayer):
    """Causal attention logit bias layer (no explicit padding masking)."""

    def forward(self, *, segment_ids: Array, positions: Array) -> Array:
        # positions: [B, L]
        causal = (positions[:, None, :, None] < positions[:, None, None, :]) * NEG_INF
        # Restrict across segments (and padding) using segment bias.
        seg_bias = make_segment_bias(segment_ids, segment_ids)
        return apply_attention_logit_biases(causal, seg_bias)


class FullAttentionLogitBiasLayer(AttentionLogitBiasLayer):
    """Full attention within segments (and nonzero segment IDs)."""

    def forward(self, *, segment_ids: Array, positions: Array) -> Array:
        del positions
        return make_segment_bias(segment_ids, segment_ids)


class ALiBiAttentionLogitBiasLayer(CausalAttentionLogitBiasLayer):
    """ALiBi attention logit bias layer producing [B, H, L, L]."""

    def __init__(self, num_heads: int):
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = int(num_heads)

    def forward(self, *, segment_ids: Array, positions: Array) -> Array:
        slopes = alibi_get_slopes(self.num_heads)  # [H]
        # Relative positions: [B, L, L] = pos[i] - pos[j]
        rel = positions[:, None, :] - positions[:, :, None]
        # Add head dim and apply slopes: [B, H, L, L]
        alibi_bias = rel[:, None, :, :] * slopes[None, :, None, None]
        # Causal + segment constraints
        base = super().forward(segment_ids=segment_ids, positions=positions)
        return apply_attention_logit_biases(alibi_bias, base)


class SymmetricALiBiAttentionLogitBiasLayer(FullAttentionLogitBiasLayer):
    """Symmetric ALiBi bias: -slope * |i - j| within segments."""

    def __init__(self, num_heads: int):
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = int(num_heads)

    def forward(self, *, segment_ids: Array, positions: Array) -> Array:
        slopes = -1.0 * alibi_get_slopes(self.num_heads)  # [H]
        rel_abs = jnp.abs(positions[:, None, :] - positions[:, :, None])
        alibi_bias = rel_abs[:, None, :, :] * slopes[None, :, None, None]
        base = super().forward(segment_ids=segment_ids, positions=positions)
        return apply_attention_logit_biases(alibi_bias, base)


# -------------------- FlashAttention Mask Adapters ---------------------------


def flash_causal_mask_fn() -> FlashMaskFn:
    """Returns a FlashAttention-compatible causal mask function.

    The returned function takes two index tensors, `rows` and `cols`, and returns a
    boolean matrix where True indicates an allowed position.
    """

    def fn(rows: Array, cols: Array) -> Array:
        # rows: [Br, 1], cols: [1, Bc]
        return rows >= cols

    return fn


def flash_local_window_mask_fn(left_window: int, right_window: Optional[int] = None) -> FlashMaskFn:
    """Returns a FlashAttention-compatible local window mask function.

    Args:
        left_window: Max distance allowed to the left (past).
        right_window: Max distance allowed to the right (future). Defaults to left_window.
    """
    lw = int(left_window)
    rw = int(left_window if right_window is None else right_window)

    def fn(rows: Array, cols: Array) -> Array:
        # rows: [Br, 1], cols: [1, Bc]
        diff = rows - cols
        return jnp.logical_and(diff <= lw, diff >= -rw)

    return fn


def to_flash_mask_fn(mask: AttentionMaskBase | MaskModFn) -> FlashMaskFn:
    """Converts a simple mask into a FlashAttention-compatible mask function.

    Supported masks:
        - CausalMask or equivalent callable that enforces j <= i
        - LocalWindowMask or equivalent callable that depends only on (i - j)

    For masks depending on batch- or segment-specific information (e.g., padding,
    SameSegmentMask), prefer using FlashAttention's built-in `segment_ids` or biases
    rather than a mask_fn.
    """
    if isinstance(mask, CausalMask):
        return flash_causal_mask_fn()
    if isinstance(mask, LocalWindowMask):
        return flash_local_window_mask_fn(mask.left_window, mask.right_window)

    # Attempt to adapt a stateless callable of the flex signature. We only use
    # it if the callable doesn't depend on batch/head/segments.
    if callable(mask) and not isinstance(mask, AttentionMaskBase):
        def fn(rows: Array, cols: Array) -> Array:
            q_idx = rows.squeeze(-1)
            k_idx = cols.squeeze(0)
            return mask(jnp.array(0), jnp.array(0), q_idx, k_idx, None, None)  # type: ignore[misc]

        # Probe once to catch obvious shape errors early.
        _ = fn(jnp.arange(2)[:, None], jnp.arange(2)[None, :])
        return fn

    raise NotImplementedError(
        "Unsupported mask for FlashAttention; use CausalMask/LocalWindowMask or pass segment_ids/bias."
    )
