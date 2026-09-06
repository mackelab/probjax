"""Dispatcher protocol: routing/expert execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from probjax.utils.typing import Array


@dataclass(frozen=True)
class DispatchedTokens:
    """Result of ``dispatch``: packed tokens plus routing metadata."""

    tokens: Array  # [E, C, D] packed inputs
    token_ids: Array  # [M] originating token index per assignment (M = N*K)
    expert_ids: Array  # [M] sorted expert index per assignment
    route_weights: Array  # [M] weight per assignment
    valid: Array  # [M] bool — False for capacity-overflow (dropped) assignments
    safe_slots: Array  # [M] slot index clipped into [0, C)
    num_tokens: int  # N (flat token count before top-k expansion)
    capacity: int  # C


class Dispatcher(Protocol):
    def dispatch(
        self,
        x: Array,
        expert_ids: Array,
        route_weights: Array,
    ) -> DispatchedTokens:
        """Pack ``[N, D]`` tokens into ``[E, C, D]`` for expert execution."""
        ...

    def combine(self, expert_y: Array, routed: DispatchedTokens) -> Array:
        """Weight, mask and accumulate ``[E, C, D]`` outputs back to ``[N, D]``."""
        ...
