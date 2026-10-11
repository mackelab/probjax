"""Top-k Mixture-of-Experts MLP with SwiGLU experts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer

from probjax.nn.sharding import BATCH, EXPERT, constrain, param_metadata
from probjax.utils.typing import Array

# Fan-in over the per-expert input dim only; without ``batch_axis`` the expert
# axis would be folded into the fan and shrink the init scale by ``1/sqrt(E)``.
_expert_init = nnx.initializers.variance_scaling(
    1.0, "fan_in", "truncated_normal", batch_axis=(0,)
)


@dataclass(frozen=True)
class MoEAux:
    """Router diagnostics returned when ``return_aux=True``.

    Registered as a JAX pytree (``capacity`` is static) so it can be
    returned from ``jit``-traced functions.
    """

    load_balance_loss: Array  # scalar Switch-style balancing objective
    router_z_loss: Array  # scalar ST-MoE penalty mean(logsumexp(logits)^2)
    expert_counts: Array  # [E] routed assignments per expert (pre-capacity)
    expert_utilization: Array  # [E] expert_counts / (N * K)
    router_probabilities: Array  # [E] mean full-softmax probability per expert
    overflow_fraction: Array  # scalar fraction of assignments dropped by capacity
    capacity_utilization: Array  # [E] kept assignments / capacity
    capacity: int  # per-expert capacity C used by the sparse path


jax.tree_util.register_dataclass(
    MoEAux,
    data_fields=[
        "load_balance_loss",
        "router_z_loss",
        "expert_counts",
        "expert_utilization",
        "router_probabilities",
        "overflow_fraction",
        "capacity_utilization",
    ],
    meta_fields=["capacity"],
)


def load_balance_loss(
    router_probs: Array, expert_ids: Array, num_experts: int
) -> Array:
    """Switch-style balancing objective ``E * sum_e p_e * f_e``.

    Args:
        router_probs: ``[N, E]`` full softmax probabilities.
        expert_ids: ``[N, K]`` selected expert ids.
        num_experts: ``E``.
    """
    p_e = jnp.mean(router_probs, axis=0)
    f_e = jnp.bincount(expert_ids.reshape(-1), length=num_experts) / expert_ids.size
    return num_experts * jnp.sum(p_e * f_e)


def compute_capacity(
    n_tokens: int, top_k: int, num_experts: int, capacity_factor: float
) -> int:
    """Static per-expert capacity ``ceil(capacity_factor * N * K / E)`` (>= 1)."""
    return max(1, math.ceil(capacity_factor * n_tokens * top_k / num_experts))


class ExpertSwiGLU(nnx.Module):
    """``E`` independent SwiGLU MLPs applied in parallel: ``[E, C, D] -> [E, C, D]``.

    Kernels carry the ``EXPERT`` logical axis on their leading dim only, so
    under a mesh the experts are sharded (expert parallelism) while the inner
    dims stay replicated.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        *,
        kernel_init: Initializer = _expert_init,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.d_model = d_model

        def einsum(spec: str, shape: tuple[int, ...]) -> nnx.Einsum:
            return nnx.Einsum(
                spec,
                shape,
                kernel_init=kernel_init,
                kernel_metadata=param_metadata(EXPERT, None, None),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.gate = einsum("ecd,edh->ech", (num_experts, d_model, d_hidden))
        self.up = einsum("ecd,edh->ech", (num_experts, d_model, d_hidden))
        self.down = einsum("ech,ehd->ecd", (num_experts, d_hidden, d_model))

    def __call__(self, x: Array) -> Array:
        if x.ndim != 3 or x.shape[0] != self.num_experts or x.shape[2] != self.d_model:
            raise ValueError(
                f"Expected [E={self.num_experts}, C, D={self.d_model}], got {x.shape}"
            )
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class MoELayer(nnx.Module):
    """Top-k MoE layer: linear router + :class:`ExpertSwiGLU` experts.

    The router always runs in float32 (top-k selection is precision
    sensitive), independent of ``dtype``.

    ``sparse=True`` sorts token assignments by expert and packs them into a
    fixed ``[E, C, D]`` buffer (assignments beyond capacity ``C`` are
    dropped). ``sparse=False`` runs every expert on every token — the exact
    reference, and often faster for small ``E``.

    Under a mesh, expert kernels shard over the ``EXPERT`` logical axis, the
    router stays replicated, and the output is constrained to ``BATCH``.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        *,
        sparse: bool = True,
        kernel_init: Initializer = _expert_init,
        router_init: Initializer = nnx.initializers.lecun_normal(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.sparse = sparse

        self.router = nnx.Linear(
            d_model,
            num_experts,
            use_bias=False,
            kernel_init=router_init,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.experts = ExpertSwiGLU(
            num_experts,
            d_model,
            d_hidden,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _dense(self, x: Array, expert_ids: Array, weights: Array) -> Array:
        n_tokens = x.shape[0]
        expert_y = self.experts(jnp.broadcast_to(x, (self.num_experts, *x.shape)))
        # [N, E] mixing weights, zero for unselected experts.
        gates = jnp.zeros((n_tokens, self.num_experts), weights.dtype)
        gates = gates.at[jnp.arange(n_tokens)[:, None], expert_ids].set(weights)
        return jnp.einsum("ne,end->nd", gates, expert_y.astype(gates.dtype))

    def _sparse(
        self, x: Array, expert_ids: Array, weights: Array, counts: Array, capacity: int
    ) -> Array:
        n_tokens, d_model = x.shape
        # Sort the N*K assignments by expert; each gets a slot within its expert.
        flat_ids = expert_ids.reshape(-1)
        order = jnp.argsort(flat_ids, stable=True)
        experts = flat_ids[order]
        tokens = order // self.top_k
        offsets = jnp.cumsum(counts) - counts
        slots = jnp.arange(flat_ids.size) - offsets[experts]

        # Out-of-capacity slots are dropped on scatter and read as 0 on gather.
        packed = jnp.zeros((self.num_experts, capacity, d_model), x.dtype)
        packed = packed.at[experts, slots].set(x[tokens], mode="drop")
        expert_y = self.experts(packed)
        y = expert_y.at[experts, slots].get(mode="fill", fill_value=0)
        y = y * weights.reshape(-1)[order, None].astype(y.dtype)
        return jax.ops.segment_sum(y, tokens, num_segments=n_tokens)

    def __call__(
        self, x: Array, *, return_aux: bool = False
    ) -> Array | tuple[Array, MoEAux]:
        """Map ``[..., d_model] -> [..., d_model]``, optionally with :class:`MoEAux`."""
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Last dim must be d_model={self.d_model}, got {x.shape}")
        x_flat = x.reshape(-1, self.d_model)
        n_tokens = x_flat.shape[0]

        logits = self.router(x_flat)  # [N, E]
        top_logits, expert_ids = jax.lax.top_k(logits, self.top_k)  # [N, K]
        weights = jax.nn.softmax(top_logits, axis=-1)
        counts = jnp.bincount(expert_ids.reshape(-1), length=self.num_experts)
        capacity = compute_capacity(
            n_tokens, self.top_k, self.num_experts, self.capacity_factor
        )

        if self.sparse:
            y = self._sparse(x_flat, expert_ids, weights, counts, capacity)
            kept = jnp.minimum(counts, capacity)
        else:
            y = self._dense(x_flat, expert_ids, weights)
            kept = counts
        # Keep the leading batch axis on "data" (no-op without a mesh); the
        # sparse path's sort/scatter otherwise leaves it replicated.
        y = constrain(y.reshape(x.shape), BATCH)
        if not return_aux:
            return y

        router_probs = jax.nn.softmax(logits, axis=-1)
        total = n_tokens * self.top_k
        aux = MoEAux(
            load_balance_loss=load_balance_loss(
                router_probs, expert_ids, self.num_experts
            ),
            router_z_loss=jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2),
            expert_counts=counts.astype(jnp.float32),
            expert_utilization=counts / total,
            router_probabilities=jnp.mean(router_probs, axis=0),
            overflow_fraction=jnp.sum(counts - kept) / total,
            capacity_utilization=kept / capacity,
            capacity=capacity,
        )
        return y, aux
