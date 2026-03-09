from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from probjax.utils.typing import Array

from .kernel_utils import (
    ceil_div,
    fast_blockmask_causal,
    fast_blockmask_local_window,
    query_iterator_indices,
)

# ---------------------------- Base Classes -----------------------------------


class AttentionMask(ABC):
    """Base class for attention masks.

    Simplified signature (removed batch/head index). Implementations:
        __call__(q_idx, k_idx, seg_q, seg_k) -> [Q, K] bool
    Multi-batch use should be handled by vmapping externally. For legacy code
    that previously passed a batch index, remove it and vmap over batch dim.
    """

    stateful: bool = False  # True if mask uses stored data (e.g., lengths, segment ids)

    @abstractmethod
    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        raise NotImplementedError

    def dense(
        self,
        q_len: int,
        kv_len: int,
        *,
        batch_size: int = 1,
        num_heads: int = 1,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> jax.Array:
        """Materialize a dense boolean mask [B, H, Q, K].

        - For head dimension, masks are head-agnostic and broadcast across H.
        - For batch dimension, optional `seg_q` and `seg_k` can be provided as
          [B, Q] and [B, K] to drive per-batch masking where needed.
        """
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        k_idx = jnp.arange(kv_len, dtype=jnp.int32)

        def per_batch(b: jax.Array) -> jax.Array:
            # Use JAX-friendly dynamic indexing; avoid Python int() on tracers.

            def maybe_take_batch(data: Optional[Array]) -> Optional[Array]:
                if data is None:
                    return None
                ndim = getattr(data, "ndim", 0)
                if ndim >= 2:
                    return data[b]
                if ndim == 1 and getattr(data, "shape", (None,))[0] == batch_size:
                    return data[b]
                return data

            sq = maybe_take_batch(seg_q)
            sk = maybe_take_batch(seg_k)
            mk = self.__call__(q_idx, k_idx, sq, sk)  # [Q, K]
            return jnp.broadcast_to(mk, (num_heads, q_len, kv_len))  # [H, Q, K]

        b_axis = jnp.arange(batch_size, dtype=jnp.int32)
        return jax.vmap(per_batch)(b_axis)  # [B, H, Q, K]

    # Optional: Pallas data specs for mask data per side. Default: None.
    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        return (None, None)

    def get_data_block_spec_backward_pass(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
        block_q_dkv: int | None = None,
        block_kv_dkv: int | None = None,
        block_q_dq: int | None = None,
        block_kv_dq: int | None = None,
    ) -> tuple[None, None]:
        return self.get_data_block_spec(q_len, kv_len, block_q, block_k)

    # Logical combinators return composed masks.
    def __and__(self, other: "AttentionMask") -> "AttentionMask":
        if not isinstance(other, AttentionMask):
            return NotImplemented
        return ComposeMask("and", self, other)

    def __or__(self, other: "AttentionMask") -> "AttentionMask":
        if not isinstance(other, AttentionMask):
            return NotImplemented
        return ComposeMask("or", self, other)

    def __xor__(self, other: "AttentionMask") -> "AttentionMask":
        if not isinstance(other, AttentionMask):
            return NotImplemented
        return ComposeMask("xor", self, other)

    def __invert__(self) -> "AttentionMask":
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
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array:
        """Build block map where True means the block is not fully masked.

        Uses a separate thread to avoid inheriting sharding contexts during compile-time eval.
        """

        def worker():
            num_q_blocks = ceil_div(q_len, block_q)
            num_kv_blocks = ceil_div(kv_len, block_k)
            block_mask_map = np.ones(
                shape=(num_q_blocks, num_kv_blocks), dtype=np.bool_
            )
            for i in range(0, q_len, block_q):
                for j in range(0, kv_len, block_k):
                    rows = np.arange(i, i + block_q, dtype=np.int32)
                    cols = np.arange(j, j + block_k, dtype=np.int32)
                    with jax.ensure_compile_time_eval():
                        if not self.__call__(rows, cols).any():
                            block_mask_map[i // block_q, j // block_k] = False
            return block_mask_map

        with ThreadPoolExecutor(1) as pool:
            return pool.submit(worker).result()

    def query_iterator_indices(
        self,
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
            q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k
        )
        if bm is not None:
            return query_iterator_indices(bm)
        else:
            return None, None

    def kv_iterator_indices(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> tuple[Array, Array]:
        """Per-(b,h,kv_block) iterators over non-empty Q blocks.

        Returns (q_block_offset [nKB,nQB], q_block_offset_size [nKB]).
        """
        bm = self.block_mask(
            q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k
        )
        if bm is not None:
            return query_iterator_indices(bm.T)
        else:
            return None, None


class AttentionBias(ABC):
    """Base class for attention score biases.

    Simplified signature (removed batch index). Implementations:
        __call__(scores, h_idx, q_idx, k_idx, data) -> [Q,K] scores
    """

    @abstractmethod
    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        raise NotImplementedError

    # Optional: Pallas bias spec for dense tensor bias.
    def get_block_spec(
        self, *, q_len: int, kv_len: int, block_q: int, block_kv: int
    ):  # pragma: no cover - API
        return None

    # Optional: backward-specific Pallas bias spec for dense tensor bias.
    # Default to forward mapping for compatibility.
    def get_block_spec_backward(
        self, *, q_len: int, kv_len: int, block_q: int, block_kv: int
    ):  # pragma: no cover - API
        return self.get_block_spec(
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_kv=block_kv,
        )

    # Allow additive composition of biases.
    def __add__(self, other: "AttentionBias") -> "AttentionBias":
        if not isinstance(other, AttentionBias):
            return NotImplemented
        return SumBias(self, other)

    def __radd__(self, other: "AttentionBias") -> "AttentionBias":
        if not isinstance(other, AttentionBias):
            return NotImplemented
        return SumBias(other, self)

    def get_data(self) -> Optional[Array]:  # pragma: no cover - API
        return None

    # Optional: gradient of the bias modifier w.r.t. input scores.
    # For purely additive biases f(scores) = scores + g(...), df/dscores = 1.
    # Subclasses that scale or nonlinearly transform scores may override.
    def grad(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:  # pragma: no cover - default behavior
        return jnp.ones_like(scores)

    def dense(
        self,
        q_len: int,
        kv_len: int,
        *,
        batch_size: int = 1,
        num_heads: int = 1,
    ) -> jax.Array:
        """Materialize a dense additive bias tensor [B, H, Q, K].

        Default implementation evaluates the bias per head and broadcasts across batch.
        Subclasses with stored dense data (e.g., DenseBias) may override for efficiency.
        """
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        k_idx = jnp.arange(kv_len, dtype=jnp.int32)

        def per_head(h: jax.Array) -> jax.Array:
            base = jnp.zeros((q_len, kv_len), dtype=jnp.float32)
            return self(base, h, q_idx, k_idx)  # [Q, K]

        h_axis = jnp.arange(num_heads, dtype=jnp.int32)
        per_h = jax.vmap(per_head)(h_axis)  # [H, Q, K]
        return jnp.broadcast_to(per_h, (batch_size, num_heads, q_len, kv_len))


# -------------------------- Composition helpers ------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ComposeMask(AttentionMask):
    op: str  # one of: 'and', 'or', 'xor'
    lhs: AttentionMask
    rhs: AttentionMask
    stateful: bool = field(init=False, default=False)

    def __post_init__(self):
        object.__setattr__(self, "stateful", self.lhs.stateful or self.rhs.stateful)

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        a = self.lhs(q_idx, k_idx, seg_q, seg_k)
        c = self.rhs(q_idx, k_idx, seg_q, seg_k)
        if self.op == "and":
            return jnp.logical_and(a, c)
        if self.op == "or":
            return jnp.logical_or(a, c)
        if self.op == "xor":
            return jnp.logical_xor(a, c)
        raise ValueError(f"Unknown op for ComposeMask: {self.op}")

    def get_data(
        self, *, q_seq_len: int | None = None, kv_seq_len: int | None = None
    ) -> tuple[Array | None, Array | None]:
        if self.lhs.stateful and self.rhs.stateful:
            raise ValueError(
                "Cannot compose two stateful masks; ambiguous data requirements."
            )
        if self.lhs.stateful:
            return self.lhs.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
        if self.rhs.stateful:
            return self.rhs.get_data(q_seq_len=q_seq_len, kv_seq_len=kv_seq_len)
        return (None, None)

    def dense(
        self,
        q_len: int,
        kv_len: int,
        *,
        batch_size: int = 1,
        num_heads: int = 1,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> jax.Array:
        if self.stateful and seg_q is None and seg_k is None:
            seg_q, seg_k = self.get_data(q_seq_len=q_len, kv_seq_len=kv_len)
        return super().dense(
            q_len,
            kv_len,
            batch_size=batch_size,
            num_heads=num_heads,
            seg_q=seg_q,
            seg_k=seg_k,
        )

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        if self.lhs.stateful and self.rhs.stateful:
            raise ValueError(
                "Cannot compose two stateful masks; ambiguous data requirements."
            )
        if self.lhs.stateful:
            return self.lhs.get_data_block_spec(
                q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k
            )
        if self.rhs.stateful:
            return self.rhs.get_data_block_spec(
                q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k
            )
        return (None, None)

    # PyTree: children are lhs/rhs masks; op is static aux.
    def tree_flatten(self):
        flat_arrays, tree = jax.tree_util.tree_flatten((self.lhs, self.rhs))
        return (flat_arrays, {"tree": tree, "op": self.op})

    @classmethod
    def tree_unflatten(cls, aux, children):
        tree = aux["tree"]
        lhs, rhs = jax.tree_util.tree_unflatten(tree, children)
        return ComposeMask(op=aux["op"], lhs=lhs, rhs=rhs)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NotMask(AttentionMask):
    inner: AttentionMask

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        return jnp.logical_not(self.inner(q_idx, k_idx, seg_q, seg_k))

    def tree_flatten(self):
        return ((self.inner,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (inner,) = children
        return NotMask(inner)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SumBias(AttentionBias):
    lhs: AttentionBias
    rhs: AttentionBias

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        s1 = self.lhs(scores, h_idx, q_idx, k_idx, data=data)
        s2 = self.rhs(scores, h_idx, q_idx, k_idx, data=data)
        # Each bias returns modified scores. Convert to additive deltas so
        # composition is scores + (delta_lhs + delta_rhs).
        return scores + (s1 - scores) + (s2 - scores)

    def tree_flatten(self):
        return ((self.lhs, self.rhs), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        lhs, rhs = children
        return SumBias(lhs, rhs)


# ------------------------------ Masks ----------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NoMask(AttentionMask):
    """Allows all attention (i.e., returns all True)."""

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        return jnp.ones((q_idx.shape[0], k_idx.shape[0]), dtype=bool)

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array | None:
        return None

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return NoMask()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CausalMask(AttentionMask):
    """Standard autoregressive mask: allow attending to past and self only."""

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        return q_idx[:, None] >= k_idx[None, :]

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array:
        return fast_blockmask_causal(
            q_len=q_len, kv_len=kv_len, block_q=block_q, block_k=block_k
        )

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return CausalMask()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalWindowMask(AttentionMask):
    """Local banded mask allowing attention within a window around each query.

    Args:
        left_window: Max distance to the left (past) allowed.
        right_window: Max distance to the right (future) allowed.
    """

    left_window: int
    right_window: Optional[int] = field(default=None)

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        dqk = q_idx[:, None] - k_idx[None, :]
        if self.right_window is None:
            right_window = 0
        else:
            right_window = int(self.right_window)
        return jnp.logical_and(dqk >= -right_window, dqk <= self.left_window)

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array:
        if self.right_window is None:
            rw = 0
        else:
            rw = int(self.right_window)
        return fast_blockmask_local_window(
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            left_window=self.left_window,
            right_window=rw,
        )

    def tree_flatten(self):
        # Treat window sizes as static metadata; no array children.
        return (
            (),
            {"left_window": self.left_window, "right_window": self.right_window},
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        lw = aux.get("left_window") if isinstance(aux, dict) else aux[0]
        rw = aux.get("right_window") if isinstance(aux, dict) else aux[1]
        return LocalWindowMask(left_window=lw, right_window=rw)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CausalLocalWindowMask(AttentionMask):
    """Causal local attention with a finite left context window.

    Query i can attend to keys k in [i - left_window, i].
    """

    left_window: int

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        dqk = q_idx[:, None] - k_idx[None, :]
        return jnp.logical_and(dqk >= 0, dqk <= self.left_window)

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array:
        return fast_blockmask_local_window(
            q_len=q_len,
            kv_len=kv_len,
            block_q=block_q,
            block_k=block_k,
            left_window=self.left_window,
            right_window=0,
        )

    def tree_flatten(self):
        return ((), {"left_window": self.left_window})

    @classmethod
    def tree_unflatten(cls, aux, children):
        lw = aux.get("left_window") if isinstance(aux, dict) else aux
        return CausalLocalWindowMask(left_window=lw)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CausalFromBottomRightMask(AttentionMask):
    """Causal mask aligned from the bottom-right for q_len != kv_len.

    Allow where q_idx + (kv_length - q_length) >= k_idx.
    """

    q_length: int
    kv_length: int

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        shift = int(self.kv_length) - int(self.q_length)
        return (q_idx[:, None] + shift) >= k_idx[None, :]

    def tree_flatten(self):
        return ((), {"q_length": self.q_length, "kv_length": self.kv_length})

    @classmethod
    def tree_unflatten(cls, aux, children):
        ql = aux.get("q_length") if isinstance(aux, dict) else aux[0]
        kl = aux.get("kv_length") if isinstance(aux, dict) else aux[1]
        return CausalFromBottomRightMask(q_length=ql, kv_length=kl)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PrefixLMMask(AttentionMask):
    """Prefix-LM mask: bidirectional on prefix, causal on suffix.

    Keys in [0, prefix_length) are visible to all queries. For keys outside
    prefix, standard causal masking is used.
    """

    prefix_lengths: Array
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_k
        p = seg_q if seg_q is not None else self.prefix_lengths
        if p is None:
            raise ValueError("PrefixLMMask requires prefix lengths")
        p = jnp.asarray(p)
        if p.ndim > 0:
            p = p.reshape(-1)[0]
        return (k_idx[None, :] < p) | (q_idx[:, None] >= k_idx[None, :])

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        del kv_len, block_q, block_k
        if getattr(self.prefix_lengths, "ndim", 0) == 0:
            return (None, None)
        q_spec = pl.BlockSpec((None, q_len), lambda _, j, k: (j, 0))
        return (q_spec, None)

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del kv_seq_len
        p = jnp.asarray(self.prefix_lengths)
        if p.ndim == 0:
            return (p, None)
        if q_seq_len is None:
            raise ValueError(
                "PrefixLMMask.get_data requires q_seq_len for vector input"
            )
        p_bq = jnp.broadcast_to(p[:, None], (p.shape[0], q_seq_len))
        return (p_bq, None)

    def tree_flatten(self):
        return ((self.prefix_lengths,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (prefix_lengths,) = children
        return PrefixLMMask(prefix_lengths)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockDiagonalMask(AttentionMask):
    """Allow attention only for token pairs with the same block id."""

    query_block_ids: Array
    key_block_ids: Optional[Array] = None
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        qids = seg_q if seg_q is not None else self.query_block_ids
        kids = (
            seg_k
            if seg_k is not None
            else (self.key_block_ids if self.key_block_ids is not None else qids)
        )
        if qids is None or kids is None:
            raise ValueError("BlockDiagonalMask requires block ids")
        return qids[..., :, None] == kids[..., None, :]

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        if self.query_block_ids is None:
            q_spec = None
        elif self.query_block_ids.ndim == 2:
            q_spec = pl.BlockSpec((None, q_len), lambda _, j, k: (j, 0))
        elif self.query_block_ids.ndim == 1:
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (0,))
        else:
            q_spec = None
        if self.key_block_ids is None:
            return (q_spec, None)
        if self.key_block_ids.ndim == 2:
            k_spec = pl.BlockSpec((None, kv_len), lambda _, j, k: (j, 0))
        elif self.key_block_ids.ndim == 1:
            k_spec = pl.BlockSpec((kv_len,), lambda _, j, k: (0,))
        else:
            k_spec = None
        return (q_spec, k_spec)

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len, kv_seq_len
        return self.query_block_ids, self.key_block_ids

    def tree_flatten(self):
        return ((self.query_block_ids, self.key_block_ids), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        qids, kids = children
        return BlockDiagonalMask(qids, kids)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockDiagonalCausalMask(AttentionMask):
    """Causal mask applied independently within each block id."""

    query_block_ids: Array
    key_block_ids: Optional[Array] = None
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        qids = seg_q if seg_q is not None else self.query_block_ids
        kids = (
            seg_k
            if seg_k is not None
            else (self.key_block_ids if self.key_block_ids is not None else qids)
        )
        if qids is None or kids is None:
            raise ValueError("BlockDiagonalCausalMask requires block ids")
        same = qids[..., :, None] == kids[..., None, :]
        causal = q_idx[:, None] >= k_idx[None, :]
        return same & causal

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        return BlockDiagonalMask(
            self.query_block_ids, self.key_block_ids
        ).get_data_block_spec(q_len, kv_len, block_q, block_k)

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len, kv_seq_len
        return self.query_block_ids, self.key_block_ids

    def tree_flatten(self):
        return ((self.query_block_ids, self.key_block_ids), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        qids, kids = children
        return BlockDiagonalCausalMask(qids, kids)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QKVLengthMask(AttentionMask):
    """This is a useful mask Q,K,V length are not power of 2."""

    q_length: int
    kv_length: int
    block_sparse: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        del seg_q, seg_k
        return jnp.logical_and(
            q_idx[:, None] < self.q_length, k_idx[None, :] < self.kv_length
        )

    def tree_flatten(self):
        # Treat window sizes as static metadata; no array children.
        return (
            (),
            {"q_length": self.q_length, "kv_length": self.kv_length},
        )

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array:
        if not self.block_sparse:
            return None
        nQB = ceil_div(q_len, block_q)
        nKB = ceil_div(kv_len, block_k)
        bm = np.ones((nQB, nKB), dtype=np.bool_)
        last_q_block = (self.q_length - 1) // block_q
        last_kv_block = (self.kv_length - 1) // block_k
        bm[last_q_block + 1 :, :] = False
        bm[:, last_kv_block + 1 :] = False
        return bm

    @classmethod
    def tree_unflatten(cls, aux, children):
        q_len = aux.get("q_length") if isinstance(aux, dict) else aux[0]
        kv_len = aux.get("kv_length") if isinstance(aux, dict) else aux[1]
        return QKVLengthMask(q_length=q_len, kv_length=kv_len)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KeyPaddingMask(AttentionMask):
    """Masks out padded key positions based on key lengths or boolean mask.

    Args:
        key_lengths: int array [B] giving number of valid KV tokens per batch; or
            boolean array [B, K] where True indicates a valid KV position.
    """

    key_lengths: Array  # Per query key lengths [B] or bool mask [B, K]
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        if seg_q is None and self.key_lengths is None:
            raise ValueError(
                "KeyPaddingMask requires key_lengths provided either at construction or call time"
            )
        kl = seg_q if seg_q is not None else self.key_lengths
        if kl is None:
            raise ValueError("KeyPaddingMask could not resolve key lengths")
        # lengths vector [B] or scalar
        valid_k = k_idx[None, :] < kl[:, None]  # [B, K]
        return valid_k[: q_idx.shape[0], :]

    # PyTree registration
    def tree_flatten(self):
        return ((self.key_lengths,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (key_lengths,) = children
        return KeyPaddingMask(key_lengths)

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        if self.key_lengths.ndim == 2:
            q_spec = pl.BlockSpec((None, q_len), lambda _, j, k: (j, 0))
        elif self.key_lengths.ndim == 1:
            # Shared 1D vector (no batch axis): always index from 0.
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (0,))
        else:
            q_spec = None
        return q_spec, None

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[None, Optional[Array]]:
        del q_seq_len, kv_seq_len
        kl = self.key_lengths
        return kl, None

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array | None:
        return None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SeqLenMask(AttentionMask):
    """Different seqlen per batch element.

    Args:
        seq_lengths: int array [B] giving number of valid KV tokens per batch;
    """

    seq_lengths: Array  # Per query key lengths [B]
    stateful: bool = True

    def dense(
        self,
        q_len: int,
        kv_len: int,
        *,
        batch_size: int = 1,
        num_heads: int = 1,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> jax.Array:
        """Materialize [B, H, Q, K] respecting per-batch seq lengths."""
        if seg_q is not None or seg_k is not None:
            return super().dense(
                q_len,
                kv_len,
                batch_size=batch_size,
                num_heads=num_heads,
                seg_q=seg_q,
                seg_k=seg_k,
            )

        L = jnp.asarray(self.seq_lengths).reshape(-1)
        b = L.shape[0]
        if batch_size != b:
            raise ValueError(
                f"SeqLenMask.dense batch_size={batch_size} does not match stored lengths {b}"
            )

        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        k_idx = jnp.arange(kv_len, dtype=jnp.int32)
        valid_q = q_idx[None, :] < L[:, None]
        valid_k = k_idx[None, :] < L[:, None]
        rect = valid_q[:, :, None] & valid_k[:, None, :]
        diag = jnp.equal(q_idx[:, None], k_idx[None, :])[None, :, :]
        mask = rect | diag
        mask = mask[:, None, :, :]
        return jnp.broadcast_to(mask, (b, num_heads, q_len, kv_len))

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        """Mask for per-batch variable sequence lengths.

        - Within the valid sequence [0, L) for the batch, allow full attention.
        - Outside the valid sequence, allow only self-attention (diagonal).

        seg_q/seg_k may be provided as vectors matching the current block (each
        entry typically equal to L for that batch). If not provided, falls back
        to the stored per-batch lengths and is expected to be vmapped outside.
        """
        if seg_q is None and self.seq_lengths is None:
            raise ValueError(
                "SeqLenMask requires seq_lengths provided either at construction or call time"
            )

        if seg_q is not None:
            # seg_q and seg_k are per-position vectors (block-sized), each entry equal to L.
            # Build rectangular validity from q and k indices separately.
            seg_q = jnp.asarray(seg_q)
            seg_k = seg_k if seg_k is not None else seg_q
            seg_k = jnp.asarray(seg_k)
            q_idx = jnp.asarray(q_idx)
            k_idx = jnp.asarray(k_idx)
            valid_q = q_idx < seg_q  # [Q]
            # If seg_k not provided, default to seg_q (self-attention case).
            valid_k = k_idx < seg_k  # [K]
            rect = valid_q[:, None] & valid_k[None, :]
            diag = q_idx[:, None] == k_idx[None, :]
            return rect | diag
        else:
            # Fallback for cases where we call without seg_* (should be vmapped over batch).
            # self.seq_lengths shape [B]; compare against q_idx/k_idx assuming single batch use.
            # Construct per-batch boolean matrices [B, Q, K]. Outside-L diagonal is kept.
            L = jnp.asarray(self.seq_lengths)
            q_idx = jnp.asarray(q_idx)
            k_idx = jnp.asarray(k_idx)
            # Broadcast batch lengths to index domain; these branches are less commonly used in-kernel.
            valid_q = q_idx[None, :] < L[:, None]
            valid_k = k_idx[None, :] < L[:, None]
            rect = valid_q[:, :, None] & valid_k[:, None, :]
            # Diagonal guard is independent of batch; broadcast across batches.
            diag = jnp.equal(q_idx[:, None], k_idx[None, :])[None, :, :]
            return rect | diag

    # PyTree registration
    def tree_flatten(self):
        return ((self.seq_lengths,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (seq_lengths,) = children
        return SeqLenMask(seq_lengths)

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        # Provide a per-batch row view of length q_len; kernel will slice with curr_q_slice.
        q_spec = pl.BlockSpec((None, q_len), lambda _, j, k_: (j, 0))
        kv_extent = kv_len if kv_len is not None else q_len
        k_spec = pl.BlockSpec((None, kv_extent), lambda _, j, k_: (j, 0))
        return q_spec, k_spec

    def get_data_block_spec_backward_pass(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
        block_q_dkv: int | None = None,
        block_kv_dkv: int | None = None,
        block_q_dq: int | None = None,
        block_kv_dq: int | None = None,
    ) -> tuple[None, None]:
        # Backward grids use (B, H, tile) ordering; we still expose a row view per batch.
        q_spec = pl.BlockSpec((None, q_len), lambda i, j, k_: (i, 0))
        kv_extent = kv_len if kv_len is not None else q_len
        k_spec = pl.BlockSpec((None, kv_extent), lambda i, j, k_: (i, 0))
        return q_spec, k_spec

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        # Broadcast per-batch lengths to a (B, Q) array whose rows are all the batch length.
        if q_seq_len is None:
            raise ValueError("SeqLenMask.get_data requires q_seq_len")
        L = jnp.asarray(self.seq_lengths)
        Lbq = jnp.broadcast_to(L[:, None], (L.shape[0], q_seq_len))
        kv_extent = kv_seq_len if kv_seq_len is not None else q_seq_len
        Lbk = jnp.broadcast_to(L[:, None], (L.shape[0], kv_extent))
        return Lbq, Lbk

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
    ) -> Array | None:
        return None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SameSegmentMask(AttentionMask):
    """Allow attention only within the same segment (uses segment_ids)."""

    # Exclude arrays from hashing/comparison to keep mask instances hashable
    query_segment_ids: Array
    key_segment_ids: Optional[Array]
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        # Prefer explicitly provided seg_q/seg_k at call time; else fallback to stored ids.
        s_q = seg_q if seg_q is not None else self.query_segment_ids
        if self.key_segment_ids is None:
            s_k = s_q
        else:
            s_k = seg_k if seg_k is not None else self.key_segment_ids
        if s_q is None or s_k is None:
            raise ValueError(
                "SameSegmentMask requires segment ids via call(seg_q, seg_k) or stored in the dataclass."
            )
        # Handle optional leading batch dimension in stored ids.
        return s_q[..., :, None] == s_k[..., None, :]

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        if self.query_segment_ids is None:
            q_spec = None
        elif self.query_segment_ids.ndim == 2:
            q_spec = pl.BlockSpec((None, q_len), lambda _, j, k: (j, 0))
        elif self.query_segment_ids.ndim == 1:
            # Shared 1D vector (no batch axis): always index from 0.
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (0,))
        else:
            raise ValueError()
        if self.key_segment_ids is None:
            return (q_spec, None)
        elif self.key_segment_ids.ndim == 2:
            k_spec = pl.BlockSpec((None, kv_len), lambda _, j, k: (j, 0))
        elif self.key_segment_ids.ndim == 1:
            # Shared 1D vector (no batch axis): always index from 0.
            k_spec = pl.BlockSpec((kv_len,), lambda _, j, k: (0,))
        else:
            k_spec = None
        return (q_spec, k_spec)

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len, kv_seq_len
        return self.query_segment_ids, self.key_segment_ids

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array | None:
        # Dynamic tensors...
        return None

    # PyTree registration
    def tree_flatten(self):
        return ((self.query_segment_ids, self.key_segment_ids), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        qids, kids = children
        return SameSegmentMask(qids, kids)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MarginalizationMask(AttentionMask):
    """Mask tokens by a boolean keep-mask with diagonal passthrough.

    Positions where both query and key mask are True are allowed. The diagonal
    (self-attention) is always kept to avoid fully-masked rows.
    """

    # Exclude arrays from hashing/comparison to keep mask instances hashable

    mask: Array
    stateful: bool = True

    def __call__(
        self,
        q_idx: Array,
        k_idx: Array,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array:
        # Prefer explicitly provided seg_q/seg_k at call time; else fallback to stored mask.
        mask_q = seg_q if seg_q is not None else self.mask
        mask_k = seg_k if seg_k is not None else self.mask
        # If seg_* are already per-block vectors, use them directly; otherwise
        # index the stored masks by q_idx/k_idx so shapes agree with indices.
        q_mask_sel = mask_q if seg_q is not None else jnp.take(mask_q, q_idx, axis=-1)
        k_mask_sel = mask_k if seg_k is not None else jnp.take(mask_k, k_idx, axis=-1)
        return (q_mask_sel[..., :, None] & k_mask_sel[..., None, :]) | (
            q_idx[..., :, None] == k_idx[..., None, :]
        )

    def get_data_block_spec(
        self,
        q_len: int,
        kv_len: int | None = None,
        block_q: int | None = None,
        block_k: int | None = None,
    ):
        if self.mask is None:
            q_spec = None
        elif self.mask.ndim == 2:
            q_spec = pl.BlockSpec((None, q_len), lambda _, j, k: (j, 0))
        elif self.mask.ndim == 1:
            # Shared 1D vector (no batch axis): always index from 0.
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (0,))
        else:
            raise ValueError()
        return q_spec, None

    def get_data(
        self,
        *,
        q_seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> tuple[Optional[Array], Optional[Array]]:
        del q_seq_len, kv_seq_len
        return self.mask, None

    def block_mask(
        self,
        q_len: int,
        kv_len: int,
        block_q: int,
        block_k: int,
        seg_q: Optional[Array] = None,
        seg_k: Optional[Array] = None,
    ) -> Array | None:
        # Dynamic tensors...
        return None

    # PyTree registration
    def tree_flatten(self):
        return (self.mask,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mask = children[0]
        return MarginalizationMask(mask)


# ------------------------------ Biases ----------------------------------------
@jax.tree_util.register_pytree_node_class
class ConstantBias(AttentionBias):
    """Adds a constant bias to all logits."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del h_idx, q_idx, k_idx, data
        return scores + self.value

    def tree_flatten(self):
        return ((), {"value": self.value})

    @classmethod
    def tree_unflatten(cls, aux, children):
        val = aux.get("value") if isinstance(aux, dict) else aux
        return ConstantBias(val)


@jax.tree_util.register_pytree_node_class
class DenseBias(AttentionBias):
    """Adds a precomputed dense bias tensor to the logits.

    Accepts a bias tensor of shape [B|1, H|1, Q, K] and supports broadcasting
    across batch or head dimensions when they are 1.
    """

    def __init__(self, bias: Array):
        # During JAX transformations (e.g., autodiff) pytree children for
        # non-differentiable leaves may be replaced with a sentinel
        # `<object object at ...>`, which does not have array attributes
        # like `ndim`. Avoid asserting in that case so tree_unflatten can
        # reconstruct a placeholder instance safely.
        if hasattr(bias, "ndim"):
            assert bias.ndim == 4, "bias must have shape [B|1, H|1, Q, K]"
        self.bias = bias

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        # If chunked data is provided (e.g., via Pallas b_ref), prefer it.
        if data is not None:
            # Expect data shaped like (block_q, kv_len) or (block_q, block_kv).
            # Add matching columns to current score tile.
            return scores + data
        # Fallback: slice from stored dense tensor using indices.
        B, H, *_ = self.bias.shape
        hsel = 0 if H == 1 else int(h_idx)
        if B != 1:
            raise ValueError(
                "DenseBias.__call__ without block data requires bias batch dimension to be 1"
            )
        bh_bias = self.bias[0, hsel]
        add = bh_bias[q_idx][:, k_idx]
        return scores + add

    def get_block_spec(self, *, q_len: int, kv_len: int, block_q: int, block_kv: int):
        return pl.BlockSpec(
            index_map=lambda i, j, k: (
                j if self.bias.shape[0] != 1 else 0,
                k if self.bias.shape[1] != 1 else 0,
                i,
                0,
            ),
            block_shape=(None, None, block_q, kv_len),
        )

    def get_block_spec_backward(
        self, *, q_len: int, kv_len: int, block_q: int, block_kv: int
    ):
        # Backward kernels use grid ordering (batch, head, tile), unlike forward
        # (q_tile, batch, head). Map B/H from the first two program IDs.
        return pl.BlockSpec(
            index_map=lambda b, h, _: (
                b if self.bias.shape[0] != 1 else 0,
                h if self.bias.shape[1] != 1 else 0,
                0,
                0,
            ),
            block_shape=(None, None, block_q, block_kv),
        )

    def get_data(self) -> Array:
        return self.bias

    def dense(
        self,
        q_len: int,
        kv_len: int,
        *,
        batch_size: int = 1,
        num_heads: int = 1,
    ) -> jax.Array:
        b = self.bias
        if hasattr(b, "shape") and b.shape[-2:] != (q_len, kv_len):
            raise ValueError(
                f"DenseBias shape mismatch: bias[...,Q,K]={b.shape[-2:]} vs ({q_len},{kv_len})"
            )
        B_src, H_src = int(b.shape[0]), int(b.shape[1])
        # Validate broadcastability
        if B_src not in (1, batch_size):
            raise ValueError(f"Cannot broadcast bias batch dim {B_src} to {batch_size}")
        if H_src not in (1, num_heads):
            raise ValueError(f"Cannot broadcast bias head dim {H_src} to {num_heads}")
        target_shape = (
            batch_size,
            num_heads,
            q_len,
            kv_len,
        )
        if B_src == batch_size and H_src == num_heads:
            return b
        return jnp.broadcast_to(b, target_shape)

    # PyTree registration
    def tree_flatten(self):
        return ((self.bias,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (bias,) = children
        # If `bias` is a JAX sentinel (e.g., `<object object ...>`), avoid
        # calling `__init__` which asserts on `ndim`. Instead, construct a
        # bare instance and attach the sentinel; it will only be used for
        # structural purposes during transformation and not at runtime.
        if not hasattr(bias, "ndim"):
            obj = object.__new__(DenseBias)
            obj.bias = bias
            return obj
        return DenseBias(bias)


# ---------------------- Class-based Stateless Biases -------------------------


@jax.tree_util.register_pytree_node_class
class IdentityBias(AttentionBias):
    """No-op bias that returns scores unchanged."""

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del h_idx, q_idx, k_idx, data
        return scores

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return IdentityBias()


def _alibi_slope_for_head(h_idx: Array) -> Array:
    """Per-head slope for ALiBi-style linear bias using 2^(-(h+1))."""
    h_f = h_idx.astype(jnp.float32)
    return jnp.exp2(-(h_f + 1.0))


@jax.tree_util.register_pytree_node_class
class CausalAlibiBias(AttentionBias):
    """Causal ALiBi-style linear position bias with head-dependent slope.

    bias = -slope(h) * relu(i - j)
    """

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del data
        slope = _alibi_slope_for_head(h_idx)
        dist = jnp.maximum(q_idx[:, None] - k_idx[None, :], 0)
        return scores - slope * dist

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return CausalAlibiBias()


@jax.tree_util.register_pytree_node_class
class SymmetricAlibiBias(AttentionBias):
    """Symmetric ALiBi-style bias using absolute distance.

    bias = -slope(h) * |i - j|
    """

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del data
        slope = _alibi_slope_for_head(h_idx)
        dist = jnp.abs(q_idx[:, None] - k_idx[None, :])
        return scores - slope * dist

    def tree_flatten(self):
        return ((), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return SymmetricAlibiBias()


# Backward compatible alias; prefer CausalAlibiBias directly.
ALiBiBias = CausalAlibiBias


@jax.tree_util.register_pytree_node_class
class DistanceDecayBias(AttentionBias):
    """Symmetric distance penalty independent of direction.

    Adds a negative penalty proportional to absolute distance |i - j|.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del h_idx, data
        dist = jnp.abs(q_idx[:, None] - k_idx[None, :]).astype(jnp.float32)
        return scores - jnp.asarray(self.alpha, dtype=dist.dtype) * dist

    def tree_flatten(self):
        return ((), {"alpha": self.alpha})

    @classmethod
    def tree_unflatten(cls, aux, children):
        a = aux.get("alpha") if isinstance(aux, dict) else aux
        return DistanceDecayBias(alpha=a)


def _relative_position_bucket(
    relative_position: Array,
    *,
    num_buckets: int,
    max_distance: int,
    bidirectional: bool,
) -> Array:
    """T5-style relative position bucketing."""
    rp = -relative_position
    if bidirectional:
        half = num_buckets // 2
        sign_bucket = (rp < 0).astype(jnp.int32) * half
        rp = jnp.abs(rp)
        nb = half
    else:
        sign_bucket = 0
        rp = jnp.maximum(rp, 0)
        nb = num_buckets

    max_exact = nb // 2
    is_small = rp < max_exact
    rp_float = rp.astype(jnp.float32)
    max_exact_f = jnp.asarray(max_exact, dtype=jnp.float32)
    nb_f = jnp.asarray(nb, dtype=jnp.float32)
    max_dist_f = jnp.asarray(max_distance, dtype=jnp.float32)
    val_large = max_exact_f + (
        jnp.log(jnp.maximum(rp_float, 1.0) / max_exact_f)
        / jnp.log(max_dist_f / max_exact_f)
        * (nb_f - max_exact_f)
    )
    val_large = jnp.minimum(nb - 1, val_large.astype(jnp.int32))
    bucket = jnp.where(is_small, rp.astype(jnp.int32), val_large)
    return bucket + sign_bucket


@jax.tree_util.register_pytree_node_class
class T5RelativePositionBias(AttentionBias):
    """T5-style bucketed relative position bias.

    `bias_table` has shape [H|1, num_buckets] (or [num_buckets]).
    """

    def __init__(
        self,
        bias_table: Array,
        *,
        num_buckets: int,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        bt = jnp.asarray(bias_table)
        if bt.ndim == 1:
            bt = bt[None, :]
        if bt.ndim != 2:
            raise ValueError("bias_table must have shape [H|1, num_buckets]")
        if int(bt.shape[-1]) != int(num_buckets):
            raise ValueError("bias_table last dim must equal num_buckets")
        self.bias_table = bt
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del data
        rel = k_idx[None, :] - q_idx[:, None]
        buckets = _relative_position_bucket(
            rel,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            bidirectional=self.bidirectional,
        )
        hsel = 0 if self.bias_table.shape[0] == 1 else int(h_idx)
        add = self.bias_table[hsel][buckets]
        return scores + add

    def tree_flatten(self):
        return (
            (self.bias_table,),
            {
                "num_buckets": self.num_buckets,
                "max_distance": self.max_distance,
                "bidirectional": self.bidirectional,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (bias_table,) = children
        return T5RelativePositionBias(
            bias_table,
            num_buckets=aux["num_buckets"],
            max_distance=aux["max_distance"],
            bidirectional=aux["bidirectional"],
        )


@jax.tree_util.register_pytree_node_class
class LearnedRelativePositionBias(AttentionBias):
    """Learned clipped-distance bias table indexed by (k - q).

    `table` has shape [H|1, 2*max_distance + 1] (or [2*max_distance + 1]).
    """

    def __init__(self, table: Array, *, max_distance: int):
        t = jnp.asarray(table)
        if t.ndim == 1:
            t = t[None, :]
        if t.ndim != 2:
            raise ValueError("table must have shape [H|1, 2*max_distance+1]")
        expected = 2 * int(max_distance) + 1
        if int(t.shape[-1]) != expected:
            raise ValueError("table last dim must be 2*max_distance+1")
        self.table = t
        self.max_distance = int(max_distance)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del data
        rel = k_idx[None, :] - q_idx[:, None]
        rel = jnp.clip(rel, -self.max_distance, self.max_distance) + self.max_distance
        hsel = 0 if self.table.shape[0] == 1 else int(h_idx)
        add = self.table[hsel][rel]
        return scores + add

    def tree_flatten(self):
        return ((self.table,), {"max_distance": self.max_distance})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (table,) = children
        return LearnedRelativePositionBias(table, max_distance=aux["max_distance"])


@jax.tree_util.register_pytree_node_class
class SoftCappingBias(AttentionBias):
    """Apply tanh soft-capping to logits: softcap * tanh(scores / softcap)."""

    def __init__(self, softcap: float = 20.0):
        if softcap <= 0:
            raise ValueError("softcap must be > 0")
        self.softcap = float(softcap)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del h_idx, q_idx, k_idx, data
        s = jnp.asarray(self.softcap, dtype=scores.dtype)
        return s * jnp.tanh(scores / s)

    def grad(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del h_idx, q_idx, k_idx, data
        s = jnp.asarray(self.softcap, dtype=scores.dtype)
        t = jnp.tanh(scores / s)
        return 1.0 - t * t

    def tree_flatten(self):
        return ((), {"softcap": self.softcap})

    @classmethod
    def tree_unflatten(cls, aux, children):
        return SoftCappingBias(softcap=aux["softcap"])


@jax.tree_util.register_pytree_node_class
class PerHeadScaleBias(AttentionBias):
    """Multiply attention scores by a per-head scale."""

    def __init__(self, scales: Array):
        s = jnp.asarray(scales)
        if s.ndim == 0:
            s = s[None]
        if s.ndim != 1:
            raise ValueError("scales must be shape [H|1]")
        self.scales = s.astype(jnp.float32)

    def __call__(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del q_idx, k_idx, data
        hsel = 0 if self.scales.shape[0] == 1 else int(h_idx)
        return scores * jnp.asarray(self.scales[hsel], dtype=scores.dtype)

    def grad(
        self,
        scores: Array,
        h_idx: Array,
        q_idx: Array,
        k_idx: Array,
        data: Optional[Array] = None,
    ) -> Array:
        del scores, q_idx, k_idx, data
        hsel = 0 if self.scales.shape[0] == 1 else int(h_idx)
        return jnp.asarray(self.scales[hsel], dtype=jnp.float32)

    def tree_flatten(self):
        return ((self.scales,), {})

    @classmethod
    def tree_unflatten(cls, aux, children):
        (scales,) = children
        return PerHeadScaleBias(scales)
