from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, Callable
from jax.experimental import pallas as pl
from .utils import (
    NEG_INF,
    DEFAULT_MASK_VALUE,
    ceil_div,
    ensure_tuple_segment_ids,
    block_pair_has_any,
    fast_blockmask_local_window,
    fast_blockmask_causal,
    row_iterators,
    compute_block_iterators,
    compute_kv_iterators,
    compute_block_bounds,
    materialize_mask,
    materialize_bias,
    apply_attention_logit_biases,
    make_segment_bias,
    compute_padding_biases,
    alibi_get_slopes,
)

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
    "DenseBias",
    "get_bias_grad",
    # Class-based stateless biases
    "IdentityBias",
    "CausalBias",
    "ALiBiBias",
    "DistanceDecayBias",
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
    # Helpers (re-exported from utils)
    "compute_padding_biases",
    "make_segment_bias",
    "apply_attention_logit_biases",
    "alibi_get_slopes",
    # Block sparsity helpers (re-exported + local wrapper)
    "compute_block_mask",
    "compute_block_iterators",
    "compute_kv_iterators",
    "compute_block_bounds",
    "ceil_div",
    "materialize_mask",
    "materialize_bias",
]


# ----------------------------- Utilities ------------------------------------

# Function type aliases following project typing conventions.
MaskModFn = Callable[
    [Array, Array, Array, Array, Optional[Array], Optional[Array]],
    Array,
]
ScoreModFn = Callable[[Array, Array, Array, Array, Array], Array]

## Deprecated Flash mask function types and adapters removed.


# ------------------------ Block Sparsity Helpers -----------------------------

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
    seg_q, seg_k = ensure_tuple_segment_ids(segment_ids)

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
                return block_pair_has_any(
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

    # Optional: Pallas data specs for mask data per side. Default: None.
    def pallas_q_data_spec(self, q_seq_len: int):  # pragma: no cover - API
        return None

    def pallas_k_data_spec(self, kv_seq_len: int):  # pragma: no cover - API
        return None

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

    # (pallas_q_data_spec / pallas_k_data_spec defined above)

    # Optional: returns a tuple (q_data, k_data) arrays required by this mask.
    # Implementations may use q_seq_len/kv_seq_len to construct per-sequence data.
    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:  # pragma: no cover - API surface
        del q_seq_len, kv_seq_len
        return None, None

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
                return block_pair_has_any(
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
        idx, sz = jax.vmap(row_iterators)(bm)
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
        idx, sz = jax.vmap(row_iterators)(bm_t)
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
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        raise NotImplementedError

    # Optional: Pallas bias spec for dense tensor bias.
    def pallas_bias_spec(self, *, block_q: int, kv_seq_len: int):  # pragma: no cover - API
        return None

    # Allow additive composition of biases.
    def __add__(self, other: "AttentionBiasBase") -> "AttentionBiasBase":
        if not isinstance(other, AttentionBiasBase):
            return NotImplemented
        return SumBias(self, other)

    def __radd__(self, other: "AttentionBiasBase") -> "AttentionBiasBase":
        if not isinstance(other, AttentionBiasBase):
            return NotImplemented
        return SumBias(other, self)

    # (pallas_bias_spec defined above)

    # Optional: returns tuple of arrays to be provided to the kernel (e.g., a dense bias tensor)
    def get_data(self) -> tuple[Optional[Array], Optional[Array] | Array]:  # pragma: no cover - API surface
        return ()

    # Optional: gradient of the bias modifier w.r.t. input scores.
    # For purely additive biases f(scores) = scores + g(...), df/dscores = 1.
    # Subclasses that scale or nonlinearly transform scores may override.
    def grad(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:  # pragma: no cover - default behavior
        return jnp.ones_like(scores)


# -------------------------- Composition helpers ------------------------------


@jax.tree_util.register_pytree_node_class
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

    # PyTree: children are lhs/rhs masks; op is static aux.
    def tree_flatten(self):
        return ((self.lhs, self.rhs), {"op": self.op})

    @classmethod
    def tree_unflatten(cls, aux, children):
        lhs, rhs = children
        op = aux["op"] if isinstance(aux, dict) else aux
        return ComposeMask(op, lhs, rhs)


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(self):
        return ((self.inner,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (inner,) = children
        return NotMask(inner)


@jax.tree_util.register_pytree_node_class
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
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        s1 = self.lhs(scores, b_idx, h_idx, q_idx, k_idx, data_q, data_k)
        s2 = self.rhs(scores, b_idx, h_idx, q_idx, k_idx, data_q, data_k)
        return s1 + s2

    def tree_flatten(self):
        return ((self.lhs, self.rhs), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        lhs, rhs = children
        return SumBias(lhs, rhs)


# ------------------------------ Masks ----------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
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

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return NoMask()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
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
        return fast_blockmask_causal(q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k)

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return CausalMask()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalWindowMask(AttentionMaskBase):
    """Local banded mask allowing attention within a window around each query.

    Args:
        left_window: Max distance to the left (past) allowed.
        right_window: Max distance to the right (future) allowed.
    """

    left_window: int
    right_window: Optional[int] = field(default=None)

    def __post_init__(self):
        # Normalize right_window to an int value for consistent behavior
        rw = self.left_window if self.right_window is None else int(self.right_window)
        object.__setattr__(self, "left_window", int(self.left_window))
        object.__setattr__(self, "right_window", rw)

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
        return fast_blockmask_local_window(
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            left_window=self.left_window,
            right_window=self.right_window,
        )

    def tree_flatten(self):
        # Treat window sizes as static metadata; no array children.
        return ((), {"left_window": self.left_window, "right_window": self.right_window})

    @classmethod
    def tree_unflatten(cls, aux, children):
        lw = aux.get("left_window") if isinstance(aux, dict) else aux[0]
        rw = aux.get("right_window") if isinstance(aux, dict) else aux[1]
        return LocalWindowMask(left_window=lw, right_window=rw)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KeyPaddingMask(AttentionMaskBase):
    """Masks out padded key positions based on key lengths or boolean mask.

    Args:
        key_lengths: int array [B] giving number of valid KV tokens per batch; or
            boolean array [B, K] where True indicates a valid KV position.
    """
    # Mark array field as non-comparable/non-hashable to allow static hashing of the mask object
    key_lengths: Optional[Array] = field(compare=False, hash=False, repr=False)

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        if seg_k is None and self.key_lengths is None:
            raise ValueError("KeyPaddingMask requires key_lengths provided either at construction or call time")
        kl = seg_k if seg_k is not None else self.key_lengths
        if (seg_k is not None) and (seg_k.ndim == 1):
            # Per-tile boolean vector [K_tile] provided via kernel load
            valid_k = kl.astype(jnp.bool_)
        elif kl.dtype == jnp.bool_:  # shape [B, K]
            valid_k = kl[b_idx][k_idx]
        else:  # assume lengths [B]
            valid_k = k_idx < kl[b_idx]
        return jnp.broadcast_to(valid_k[None, :], (q_idx.shape[0], k_idx.shape[0]))

    # PyTree registration
    def tree_flatten(self):
        return ((self.key_lengths,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (key_lengths,) = children
        return KeyPaddingMask(key_lengths)

    def pallas_k_data_spec(self, kv_seq_len: int):
        return pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len
        if self.key_lengths is None:
            return None, None
        kl = self.key_lengths
        if kv_seq_len is None:
            # Return stored form as-is
            return None, kl
        # Normalize to boolean [B, K] when necessary
        if kl.dtype == jnp.bool_:
            return None, kl
        # lengths [B] -> boolean [B,K]
        bools = (jnp.arange(kv_seq_len)[None, :] < kl[:, None]).astype(jnp.bool_)
        return None, bools

    # Optional helper to prepare per-sequence data.
    def get_data_with_seq(self, q_seq_len: int, kv_seq_len: int) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len
        if self.key_lengths is None:
            return None, None
        kl = self.key_lengths
        if kl.dtype == jnp.bool_ and kl.ndim == 2:
            return None, kl
        if kl.ndim == 1:
            bools = (jnp.arange(kv_seq_len)[None, :] < kl[:, None]).astype(jnp.bool_)
            return None, bools
        return None, kl

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
        k_starts = jnp.arange(0, kv_len, block_k)
        # Determine validity per KV block
        if seg_k is not None and seg_k.dtype == jnp.bool_:
            kb = seg_k[b_idx]
            def any_in_block(k_start):
                end = jnp.minimum(k_start + block_k, kv_len)
                return jnp.any(kb[k_start:end])
            allowed = jax.vmap(any_in_block)(k_starts)
        else:
            # lengths vector required either from seg_k (1D) or stored
            lengths = seg_k if (seg_k is not None and seg_k.ndim == 1) else self.key_lengths
            if lengths is None:
                raise ValueError("KeyPaddingMask requires key_lengths")
            len_b = lengths[b_idx]
            allowed = k_starts < len_b
        # Broadcast across all query blocks
        return jnp.broadcast_to(allowed[None, :], (nQB, nKB))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SameSegmentMask(AttentionMaskBase):
    """Allow attention only within the same segment (uses segment_ids)."""

    # Exclude arrays from hashing/comparison to keep mask instances hashable
    query_segment_ids: Optional[Array] = field(default=None, compare=False, hash=False, repr=False)
    key_segment_ids: Optional[Array] = field(default=None, compare=False, hash=False, repr=False)

    def __call__(self, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, seg_q: Optional[Array] = None, seg_k: Optional[Array] = None) -> Array:
        # Prefer explicitly provided seg_q/seg_k at call time; else fallback to stored ids.
        s_q = seg_q if seg_q is not None else self.query_segment_ids
        s_k = seg_k if seg_k is not None else self.key_segment_ids
        if s_q is None or s_k is None:
            raise ValueError(
                "SameSegmentMask requires segment ids via call(seg_q, seg_k) or stored in the dataclass."
            )
        # Handle optional leading batch dimension in stored ids.
        if s_q.ndim == 2:  # [B, Q]
            q_seg = s_q[b_idx][q_idx]
        else:  # [Q]
            q_seg = s_q[q_idx]
        if s_k.ndim == 2:  # [B, K]
            k_seg = s_k[b_idx][k_idx]
        else:  # [K]
            k_seg = s_k[k_idx]
        return q_seg[:, None] == k_seg[None, :]

    def pallas_q_data_spec(self, q_seq_len: int):
        return pl.BlockSpec((None, q_seq_len), lambda _, j, k: (j, 0))

    def pallas_k_data_spec(self, kv_seq_len: int):
        return pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len, kv_seq_len
        return self.query_segment_ids, self.key_segment_ids

    # PyTree registration
    def tree_flatten(self):
        return ((self.query_segment_ids, self.key_segment_ids), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        qids, kids = children
        return SameSegmentMask(qids, kids)


# ------------------------------ Biases ----------------------------------------


@jax.tree_util.register_pytree_node_class
class FromMaskBias(AttentionBiasBase):
    """Converts a mask into additive bias using a large negative value.

    Positions where mask is False receive `mask_value` (e.g., -1e9), others get 0.
    """

    def __init__(self, mask: AttentionMaskBase, mask_value: float = DEFAULT_MASK_VALUE):
        self.mask = mask
        self.mask_value = mask_value

    def __call__(self, scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, data_q: Optional[Array] = None, data_k: Optional[Array] = None) -> Array:
        mk = self.mask(b_idx, h_idx, q_idx, k_idx, None, None)
        return jnp.where(mk, scores, scores + self.mask_value)

    def tree_flatten(self):
        return ((self.mask,), {"mask_value": self.mask_value})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mask,) = children
        mv = aux.get("mask_value") if isinstance(aux, dict) else aux
        return FromMaskBias(mask, mv)


@jax.tree_util.register_pytree_node_class
class ConstantBias(AttentionBiasBase):
    """Adds a constant bias to all logits."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, data_q: Optional[Array] = None, data_k: Optional[Array] = None) -> Array:
        return scores + self.value

    def tree_flatten(self):
        return ((), {"value": self.value})

    @classmethod
    def tree_unflatten(cls, aux, children):
        val = aux.get("value") if isinstance(aux, dict) else aux
        return ConstantBias(val)


@jax.tree_util.register_pytree_node_class
class DenseBias(AttentionBiasBase):
    """Adds a precomputed dense bias tensor to the logits.

    Accepts a bias tensor of shape [B|1, H|1, Q, K] and supports broadcasting
    across batch or head dimensions when they are 1.
    """

    def __init__(self, bias: Array):
        assert bias.ndim == 4, "bias must have shape [B|1, H|1, Q, K]"
        self.bias = bias

    def __call__(self, scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array, data_q: Optional[Array] = None, data_k: Optional[Array] = None) -> Array:
        B, H, Q, K = self.bias.shape
        bsel = 0 if B == 1 else int(b_idx)
        hsel = 0 if H == 1 else int(h_idx)
        bh_bias = self.bias[bsel, hsel]
        # Subselect rows/cols
        add = bh_bias[q_idx][:, k_idx]
        return scores + add

    def pallas_bias_spec(self, *, block_q: int, kv_seq_len: int):
        B, H, *_ = self.bias.shape
        return pl.BlockSpec(
            index_map=lambda i, j, k: (
                j if B != 1 else 0,
                k if H != 1 else 0,
                i,
                0,
            ),
            block_shape=(None, None, block_q, kv_seq_len),
        )

    def get_data(self) -> tuple[Array, ...]:
        return (self.bias,)

    # PyTree registration
    def tree_flatten(self):
        return ((self.bias,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (bias,) = children
        return DenseBias(bias)


# ---------------------- Class-based Stateless Biases -------------------------


@jax.tree_util.register_pytree_node_class
class IdentityBias(AttentionBiasBase):
    """No-op bias that returns scores unchanged."""

    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        del b_idx, h_idx, q_idx, k_idx, data_q, data_k
        return scores

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return IdentityBias()


@jax.tree_util.register_pytree_node_class
class CausalBias(AttentionBiasBase):
    """Causal masking expressed as an additive bias."""

    def __init__(self, mask_value: float = DEFAULT_MASK_VALUE):
        self.mask_value = float(mask_value)

    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        del b_idx, h_idx, data_q, data_k
        mask = q_idx[:, None] >= k_idx[None, :]
        return jnp.where(mask, scores, scores + self.mask_value)

    def tree_flatten(self):
        return ((), {"mask_value": self.mask_value})

    @classmethod
    def tree_unflatten(cls, aux, children):
        mv = aux.get("mask_value") if isinstance(aux, dict) else aux
        return CausalBias(mask_value=mv)


def _alibi_slope_for_head(h_idx: Array) -> Array:
    """Per-head slope for ALiBi-style linear bias using 2^(-(h+1))."""
    h_f = h_idx.astype(jnp.float32)
    return jnp.exp2(-(h_f + 1.0))


@jax.tree_util.register_pytree_node_class
class ALiBiBias(AttentionBiasBase):
    """ALiBi-like linear position bias with head-dependent slope.

    bias = -slope(h) * relu(i - j)
    """

    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        del b_idx, data_q, data_k
        slope = _alibi_slope_for_head(h_idx)
        dist = jnp.maximum(q_idx[:, None] - k_idx[None, :], 0)
        return scores - slope * dist

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return ALiBiBias()


@jax.tree_util.register_pytree_node_class
class DistanceDecayBias(AttentionBiasBase):
    """Symmetric distance penalty independent of direction.

    Adds a negative penalty proportional to absolute distance |i - j|.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def __call__(
        self,
        scores: Array,
        b_idx: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        del b_idx, h_idx, data_q, data_k
        dist = jnp.abs(q_idx[:, None] - k_idx[None, :]).astype(jnp.float32)
        return scores - jnp.asarray(self.alpha, dtype=dist.dtype) * dist

    def tree_flatten(self):
        return ((), {"alpha": self.alpha})

    @classmethod
    def tree_unflatten(cls, aux, children):
        a = aux.get("alpha") if isinstance(aux, dict) else aux
        return DistanceDecayBias(alpha=a)


# ------------------------- Bias Grad Utilities -------------------------------

def get_bias_grad(bias: AttentionBiasBase | Callable) -> Optional[Callable]:
    """Return a JAX-compatible grad function for a bias modifier.

    If `bias` is an AttentionBiasBase instance, return its `.grad` method if
    present; otherwise return None.
    """
    if isinstance(bias, AttentionBiasBase):
        return getattr(bias, "grad", None)
    # Callable support removed; only classes should be used.
    return None


# Helper adapters to apply masks/biases with optional data tuples
def apply_mask(mask: AttentionMaskBase,
               b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array,
               data_q: Optional[Tuple[Array, ...]] = None,
               data_k: Optional[Tuple[Array, ...]] = None) -> Array:
    # Accept either tuple or single array (legacy) for per-side data
    sq = None if data_q is None else data_q[0]
    sk = None if data_k is None else data_k[0]
    return mask(b_idx, h_idx, q_idx, k_idx, sq, sk)


def apply_bias(bias: AttentionBiasBase,
               scores: Array,
               b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array,
               data_q: Optional[Tuple[Array, ...]] = None,
               data_k: Optional[Tuple[Array, ...]] = None) -> Array:
    sq = None if data_q is None else data_q[0]
    sk = None if data_k is None else data_k[0]
    return bias(scores, b_idx, h_idx, q_idx, k_idx, sq, sk)


## CallableBias removed; only class-based biases are supported.


# --------------------------- Stateless Bias Fns -------------------------------


def bias_identity(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """No-op bias (wrapper around IdentityBias class)."""
    return IdentityBias()(scores, b_idx, h_idx, q_idx, k_idx)


def bias_causal(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """Causal masking as additive bias (wrapper around CausalBias)."""
    return CausalBias()(scores, b_idx, h_idx, q_idx, k_idx)


def bias_alibi(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """ALiBi-like position bias (wrapper around ALiBiBias)."""
    return ALiBiBias()(scores, b_idx, h_idx, q_idx, k_idx)


def bias_distance_decay(scores: Array, b_idx: Array, h_idx: Array, q_idx: Array, k_idx: Array) -> Array:
    """Symmetric distance penalty (wrapper around DistanceDecayBias)."""
    return DistanceDecayBias()(scores, b_idx, h_idx, q_idx, k_idx)


## Helpers imported from utils are re-exported via __all__ at module import.


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


## FlashAttention mask adapters deprecated and removed.
