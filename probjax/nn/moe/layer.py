"""Mixture-of-Experts MLP layer (single-device, JIT-friendly)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer

from probjax.nn.moe.dispatch.local import LocalDispatcher, compute_capacity
from probjax.nn.moe.experts import ExpertSwiGLU
from probjax.nn.moe.router import RouterOutput, TopKRouter, load_balance_loss
from probjax.nn.sharding import BATCH, constrain
from probjax.utils.typing import Array


@dataclass(frozen=True)
class MoEAux:
    """Router diagnostics returned when ``return_aux=True``.

    Registered as a JAX pytree (``capacity`` is static) so it can be
    returned from ``jit``-traced functions.
    """

    load_balance_loss: Array  # scalar Switch-style balancing objective
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
        "expert_counts",
        "expert_utilization",
        "router_probabilities",
        "overflow_fraction",
        "capacity_utilization",
    ],
    meta_fields=["capacity"],
)


class MoELayer(nnx.Module):
    """Top-k MoE layer with SwiGLU experts.

    Supports a correctness-first dense path (every expert evaluated) and a
    sparse single-device path (sorted fixed-capacity dispatch). Set
    ``sparse=False`` for the dense reference; small ``E`` may also be faster
    dense due to better GEMM utilization.

    Sharding (see :mod:`probjax.nn.sharding`): expert kernels are annotated
    with the ``EXPERT`` logical axis (expert axis only; inner dims stay
    replicated) and the router stays replicated, so constructing under a
    ``("data", "model")`` mesh yields expert-parallel parameters with no
    further configuration. The dense path keeps data sharding end to end;
    the sparse path's sort/scatter replicates intermediates and the output
    is re-constrained to ``BATCH``.
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
        kernel_init: Initializer | None = None,
        router_init: Initializer | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if d_model <= 0 or d_hidden <= 0:
            raise ValueError(
                f"d_model/d_hidden must be positive, got {d_model}/{d_hidden}"
            )
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.sparse = sparse

        self.experts = ExpertSwiGLU(
            num_experts,
            d_model,
            d_hidden,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.router = TopKRouter(
            d_model,
            num_experts,
            kernel_init=router_init,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dispatcher = LocalDispatcher(num_experts, capacity_factor)

    def _aux_stats(
        self, routed: RouterOutput, n_tokens: int, capacity: int, n_dropped: Array
    ) -> MoEAux:
        counts = jnp.sum(
            jax.nn.one_hot(routed.expert_ids, self.num_experts, dtype=jnp.float32),
            axis=(0, 1),
        )  # [E]
        total = n_tokens * self.top_k
        return MoEAux(
            load_balance_loss=load_balance_loss(
                routed.router_probs, routed.expert_ids, self.num_experts
            ),
            expert_counts=counts,
            expert_utilization=counts / total,
            router_probabilities=jnp.mean(routed.router_probs, axis=0),
            overflow_fraction=n_dropped / total,
            capacity_utilization=counts / capacity,
            capacity=capacity,
        )

    def _dense_forward(self, x_flat: Array, routed: RouterOutput) -> Array:
        expert_y = self.experts.dense_forward(x_flat)  # [N, E, D]
        selected = jnp.take_along_axis(
            expert_y, routed.expert_ids[..., None], axis=1
        )  # [N, K, D]
        return jnp.sum(selected * routed.route_weights[..., None], axis=1)

    def _sparse_forward(
        self, x_flat: Array, routed: RouterOutput
    ) -> tuple[Array, MoEAux]:
        dispatched = self.dispatcher.dispatch(
            x_flat, routed.expert_ids, routed.route_weights
        )
        expert_y = self.experts(dispatched.tokens)  # [E, C, D]
        y_flat = self.dispatcher.combine(expert_y, dispatched)
        n_dropped = jnp.sum((~dispatched.valid).astype(jnp.float32))
        # Recompute kept-per-expert counts for capacity utilization.
        kept_counts = jax.ops.segment_sum(
            dispatched.valid.astype(jnp.float32),
            dispatched.expert_ids,
            num_segments=self.num_experts,
            indices_are_sorted=True,
        )
        counts = jnp.sum(
            jax.nn.one_hot(routed.expert_ids, self.num_experts, dtype=jnp.float32),
            axis=(0, 1),
        )
        total = x_flat.shape[0] * self.top_k
        aux = MoEAux(
            load_balance_loss=load_balance_loss(
                routed.router_probs, routed.expert_ids, self.num_experts
            ),
            expert_counts=counts,
            expert_utilization=counts / total,
            router_probabilities=jnp.mean(routed.router_probs, axis=0),
            overflow_fraction=n_dropped / total,
            capacity_utilization=kept_counts / dispatched.capacity,
            capacity=dispatched.capacity,
        )
        return y_flat, aux

    def __call__(
        self, x: Array, *, return_aux: bool = False
    ) -> Array | tuple[Array, MoEAux]:
        """Forward pass.

        Args:
            x: input of shape ``[..., d_model]``.
            return_aux: if True, also return :class:`MoEAux` diagnostics.

        Returns:
            Output of shape ``[..., d_model]``, or ``(output, aux)``.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Last dim must be d_model={self.d_model}, got {x.shape}")
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # [N, D]
        routed = self.router(x_flat, top_k=self.top_k)

        if self.sparse:
            y_flat, aux = self._sparse_forward(x_flat, routed)
        else:
            y_flat = self._dense_forward(x_flat, routed)
            capacity = compute_capacity(
                x_flat.shape[0], self.top_k, self.num_experts, self.capacity_factor
            )
            aux = self._aux_stats(routed, x_flat.shape[0], capacity, jnp.zeros(()))

        y = y_flat.reshape(original_shape)
        # Keep the leading batch axis on "data" (no-op without a mesh).
        # Matters most for the sparse path, whose sort/scatter otherwise
        # leaves the output replicated under GSPMD.
        y = constrain(y, BATCH)
        return (y, aux) if return_aux else y
