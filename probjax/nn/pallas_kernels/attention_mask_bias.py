from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from probjax.utils.typing import Array, Callable

from .utils import (
    DEFAULT_MASK_VALUE,
    NEG_INF,
    alibi_get_slopes,
    apply_attention_logit_biases,
    ceil_div,
    compute_block_bounds,
    compute_block_iterators,
    compute_kv_iterators,
    compute_padding_biases,
    fast_blockmask_causal,
    fast_blockmask_local_window,
    make_segment_bias,
    materialize_bias,
    materialize_mask,
    query_iterator_indices,
)

__all__ = [
    "AttentionMask",
    "AttentionBias",
    "CausalMask",
    "LocalWindowMask",
    "KeyPaddingMask",
    "SameSegmentMask",
    "NoMask",
    "FromMaskBias",
    "ConstantBias",
    "DenseBias",
    "get_bias_grad",
    "IdentityBias",
    "CausalBias",
    "ALiBiBias",
    "DistanceDecayBias",
    "bias_identity",
    "bias_causal",
    "bias_alibi",
    "bias_distance_decay",
    "AttentionLogitBiasLayer",
    "CausalAttentionLogitBiasLayer",
    "FullAttentionLogitBiasLayer",
    "ALiBiAttentionLogitBiasLayer",
    "SymmetricALiBiAttentionLogitBiasLayer",
    "compute_padding_biases",
    "make_segment_bias",
    "apply_attention_logit_biases",
    "alibi_get_slopes",
    "compute_block_iterators",
    "compute_kv_iterators",
    "compute_block_bounds",
    "ceil_div",
    "materialize_mask",
    "materialize_bias",
]


# ---------------------------- Base Classes -----------------------------------


class AttentionMask(ABC):
    """Base class for attention masks.

    Simplified signature (removed batch index). Implementations:
        __call__(h_idx, q_idx, k_idx, seg_q, seg_k) -> [Q, K] bool
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


# -------------------------- Composition helpers ------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ComposeMask(AttentionMask):
    op: str  # one of: 'and', 'or', 'xor'
    lhs: AttentionMask
    rhs: AttentionMask

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
        if len(flat_arrays) > 2:
            raise ValueError("Naive composition not supported between stateful masks")
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
        data_q: Optional[Array] = None,
        data_k: Optional[Array] = None,
    ) -> Array:
        s1 = self.lhs(scores, h_idx, q_idx, k_idx, data_q, data_k)
        s2 = self.rhs(scores, h_idx, q_idx, k_idx, data_q, data_k)
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
        num_heads: int | None = None,
    ) -> Array:
        del num_heads
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
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (j,))
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
            valid_q = q_idx < seg_q  # [Q]
            # If seg_k not provided, default to seg_q (self-attention case).
            seg_k = seg_k if seg_k is not None else seg_q
            valid_k = k_idx < seg_k  # [K]
            rect = valid_q[:, None] & valid_k[None, :]
            diag = q_idx[:, None] == k_idx[None, :]
            return rect | diag
        else:
            # Fallback for cases where we call without seg_* (should be vmapped over batch).
            # self.seq_lengths shape [B]; compare against q_idx/k_idx assuming single batch use.
            # Construct per-batch boolean matrices [B, Q, K]. Outside-L diagonal is kept.
            L = self.seq_lengths
            # Broadcast batch lengths to index domain; these branches are less commonly used in-kernel.
            valid_q = q_idx[None, :] < L[:, None]
            valid_k = k_idx[None, :] < L[:, None]
            rect = valid_q[:, :, None] & valid_k[:, None, :]
            eye = jnp.eye(rect.shape[-1], dtype=rect.dtype)[None, :, :]
            return rect | eye

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
        return q_spec, None

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
        return q_spec, None

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
        return Lbq, None

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
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (j,))
        else:
            raise ValueError()
        if self.key_segment_ids is None:
            return None
        elif self.key_segment_ids.ndim == 2:
            k_spec = pl.BlockSpec((None, kv_len), lambda _, j, k: (j, 0))
        elif self.key_segment_ids.ndim == 1:
            k_spec = pl.BlockSpec((kv_len,), lambda _, j, k: (j,))
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
    """Allow attention only within the same segment (uses segment_ids)."""

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
            q_spec = pl.BlockSpec((q_len,), lambda _, j, k: (j,))
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
        bh_bias = self.bias[0 if B != 0 else 0, hsel]
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

    def get_data(self) -> Array:
        return self.bias

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
class ALiBiBias(AttentionBias):
    """ALiBi-like linear position bias with head-dependent slope.

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
        return ALiBiBias()


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
