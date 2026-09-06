"""Single-device sparse dispatcher with fixed expert capacity."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from probjax.nn.moe.dispatch.base import DispatchedTokens
from probjax.utils.typing import Array


def compute_capacity(
    n_tokens: int, top_k: int, num_experts: int, capacity_factor: float
) -> int:
    """Static per-expert capacity ``ceil(capacity_factor * N * K / E)`` (>= 1)."""
    return max(1, math.ceil(capacity_factor * n_tokens * top_k / num_experts))


class LocalDispatcher:
    """Sort-by-expert + fixed-capacity pack/unpack on a single device.

    The ``MoELayer`` and expert code only see ``[E, C, D]`` tensors, so this
    can later be swapped for an expert-parallel dispatcher without changing
    the model code.
    """

    def __init__(self, num_experts: int, capacity_factor: float = 1.25):
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

    def dispatch(
        self,
        x: Array,
        expert_ids: Array,
        route_weights: Array,
    ) -> DispatchedTokens:
        n_tokens, d_model = x.shape
        top_k = expert_ids.shape[1]
        capacity = compute_capacity(
            n_tokens, top_k, self.num_experts, self.capacity_factor
        )

        # Flatten [N, K] routing decisions into assignment vectors.
        flat_experts = expert_ids.reshape(-1)  # [M]
        flat_weights = route_weights.reshape(-1)  # [M]
        token_ids = jnp.repeat(jnp.arange(n_tokens), top_k)  # [M]

        # Sort by expert so each expert's assignments are contiguous.
        order = jnp.argsort(flat_experts, stable=True)
        s_experts = flat_experts[order]
        s_tokens = token_ids[order]
        s_weights = flat_weights[order]

        # Per-expert counts and slot-within-expert for every assignment.
        counts = jax.ops.segment_sum(
            jnp.ones_like(s_experts),
            s_experts,
            num_segments=self.num_experts,
            indices_are_sorted=True,
        )
        offsets = jnp.cumsum(counts) - counts
        slots = jnp.arange(n_tokens * top_k) - offsets[s_experts]

        valid = slots < capacity
        safe_slots = jnp.minimum(slots, capacity - 1)

        packed = jnp.zeros((self.num_experts, capacity, d_model), dtype=x.dtype)
        packed = packed.at[s_experts, safe_slots].add(
            x[s_tokens] * valid[:, None].astype(x.dtype)
        )
        return DispatchedTokens(
            tokens=packed,
            token_ids=s_tokens,
            expert_ids=s_experts,
            route_weights=s_weights,
            valid=valid,
            safe_slots=safe_slots,
            num_tokens=n_tokens,
            capacity=capacity,
        )

    def combine(self, expert_y: Array, routed: DispatchedTokens) -> Array:
        assignment_y = expert_y[routed.expert_ids, routed.safe_slots]  # [M, D]
        assignment_y = assignment_y * routed.valid[:, None].astype(assignment_y.dtype)
        assignment_y = assignment_y * routed.route_weights[:, None].astype(
            assignment_y.dtype
        )
        return jax.ops.segment_sum(
            assignment_y,
            routed.token_ids,
            num_segments=routed.num_tokens,
        )
