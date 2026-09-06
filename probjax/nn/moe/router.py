"""Top-k token router with Switch-style load-balancing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer

from probjax.utils.typing import Array


@dataclass(frozen=True)
class RouterOutput:
    """Top-k routing decisions. Registered as a pytree (all-array fields)."""

    expert_ids: Array  # [N, K] int32 global expert ids
    route_weights: Array  # [N, K] softmax over selected experts (rows sum to 1)
    router_probs: Array  # [N, E] full softmax over experts (for balancing stats)
    logits: Array  # [N, E]


jax.tree_util.register_dataclass(
    RouterOutput,
    data_fields=["expert_ids", "route_weights", "router_probs", "logits"],
    meta_fields=[],
)


def _default_router_init() -> Initializer:
    return nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")


class TopKRouter(nnx.Module):
    """Replicated linear router: ``logits = x @ W_router`` + top-k + softmax."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        *,
        kernel_init: Initializer | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if d_model <= 0 or num_experts <= 0:
            raise ValueError(
                f"d_model/num_experts must be positive, got {d_model}/{num_experts}"
            )
        self.d_model = d_model
        self.num_experts = num_experts
        init = kernel_init or _default_router_init()
        self.w_router = nnx.Param(
            init(rngs.params(), (d_model, num_experts), param_dtype)
        )

    def __call__(self, x: Array, *, top_k: int) -> RouterOutput:
        """Route flattened tokens.

        Args:
            x: ``[N, D]`` token matrix.
            top_k: number of experts per token (``1 <= top_k <= E``).

        Returns:
            RouterOutput with ``[N, K]`` ids/weights and ``[N, E]`` probs.
        """
        if not 1 <= top_k <= self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {top_k}")
        logits = x @ self.w_router[...]  # [N, E]
        top_logits, expert_ids = jax.lax.top_k(logits, top_k)  # [N, K]
        route_weights = jax.nn.softmax(top_logits, axis=-1)
        router_probs = jax.nn.softmax(logits, axis=-1)
        return RouterOutput(
            expert_ids=expert_ids,
            route_weights=route_weights,
            router_probs=router_probs,
            logits=logits,
        )


def load_balance_loss(
    router_probs: Array, expert_ids: Array, num_experts: int
) -> Array:
    """Switch-style balancing objective: ``E * sum_e p_e * f_e``.

    Args:
        router_probs: ``[N, E]`` full softmax probabilities.
        expert_ids: ``[N, K]`` selected expert ids.
        num_experts: ``E``.
    """
    n_tokens = router_probs.shape[0]
    p_e = jnp.mean(router_probs, axis=0)  # [E]
    one_hot = jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.float32)  # [N, K, E]
    f_e = jnp.sum(one_hot, axis=(0, 1)) / (n_tokens * expert_ids.shape[1])  # [E]
    return jnp.asarray(num_experts, dtype=jnp.float32) * jnp.sum(p_e * f_e)
