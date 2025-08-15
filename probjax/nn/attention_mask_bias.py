from abc import ABC, abstractmethod
from typing import Callable


class AttentionMask(ABC):
    def __init__(self, q_seq_len=None, kv_seq_len=None):
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len

    @abstractmethod
    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        pass

    def __jax_array__(self):
        if self.q_seq_len is None or self.kv_seq_len is None:
            raise ValueError("Mask not initialized with q_N and kv_N")
        return self.mask_mod_fn(1, 1, self.q_seq_len, self.kv_seq_len).squeeze()

    def __and__(self, other):
        if not isinstance(other, AttentionMask):
            return NotImplemented

        return GeneralMask(
            lambda b, h, q_idx, kv_idx: self.mask_mod_fn(b, h, q_idx, kv_idx)
            & other.mask_mod_fn(b, h, q_idx, kv_idx),
            self.q_seq_len,
            self.kv_seq_len,
        )

    def __or__(self, other):
        if not isinstance(other, AttentionMask):
            return NotImplemented

        return GeneralMask(
            lambda b, h, q_idx, kv_idx: self.mask_mod_fn(b, h, q_idx, kv_idx)
            | other.mask_mod_fn(b, h, q_idx, kv_idx),
            self.q_seq_len,
            self.kv_seq_len,
        )

    def __xor__(self, other):
        if not isinstance(other, AttentionMask):
            return NotImplemented

        return GeneralMask(
            lambda b, h, q_idx, kv_idx: self.mask_mod_fn(b, h, q_idx, kv_idx)
            ^ other.mask_mod_fn(b, h, q_idx, kv_idx),
            self.q_seq_len,
            self.kv_seq_len,
        )

    def __invert__(self):
        return GeneralMask(
            lambda b, h, q_idx, kv_idx: ~self.mask_mod_fn(b, h, q_idx, kv_idx),
            self.q_seq_len,
            self.kv_seq_len,
        )


class GeneralMask(AttentionMask):
    def __init__(self, mask_mod: Callable, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)
        self.mask_mod_fn = mask_mod

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return self.mask_mod_fn(b, h, q_idx, kv_idx)


class FullMask(AttentionMask):
    def __init__(self, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return True


class NoMask(AttentionMask):
    def __init__(self, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return False


class DiagonalMask(AttentionMask):
    def __init__(self, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return q_idx[:, None] == kv_idx[None, :]


class CausalMask(AttentionMask):
    def __init__(self, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return q_idx[:, None] >= kv_idx[None, :]


class SlidingWindowMask(AttentionMask):
    def __init__(self, window_size, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)
        self.window_size = window_size

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= self.window_size


class BlockMask(AttentionMask):
    def __init__(self, block_specs, q_N=None, kv_N=None):
        super().__init__(q_N, kv_N)
        self.block_specs = block_specs

    def mask_mod_fn(self, b, h, q_idx, kv_idx):
        mask = False
        for block_size, mask_instance in self.block_specs:
            mask |= mask_instance.mask_mod_fn(
                b, h, q_idx // block_size, kv_idx // block_size
            )
        return mask
