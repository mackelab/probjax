"""Top-k Mixture-of-Experts MLP with SwiGLU experts."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer

from probjax.nn.layers.grouped_matmul import (
    grouped_backend,
    grouped_swiglu,
    ragged_matmul_hopper,
)
from probjax.nn.sharding import BATCH, EXPERT, constrain, param_metadata
from probjax.utils.typing import Array, DTypeLike

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


@jax.custom_vjp
def _permute(x: Array, perm: Array, inverse: Array) -> Array:
    """``x[perm]`` whose backward is the gather ``g[inverse]``.

    A plain gather differentiates to a scatter-add, and putting the expert
    outputs back with ``zeros.at[order].set(...)`` is a scatter too.  Around
    the grouped-GEMM kernel those scatters kept every layer's buffers alive
    under the caller's remat: a stack of remat'd dropless layers at 262k
    tokens grew 5.5 GiB per layer (21.3 GiB at 4) while the expert block
    alone grew ~1 GiB per layer.  A permutation needs neither.
    """
    return x[perm]


def _permute_fwd(x, perm, inverse):
    return x[perm], (perm, inverse)


def _permute_bwd(res, g):
    perm, inverse = res
    return g[inverse], None, None


_permute.defvjp(_permute_fwd, _permute_bwd)


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
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_experts = num_experts
        self.d_model = d_model
        self.dtype = dtype

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

    def _check_rows(self, x: Array) -> None:
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"Expected [M, D={self.d_model}], got {x.shape}")

    def ragged_forward(self, x: Array, group_sizes: Array) -> Array:
        """Apply expert ``e`` to its contiguous group of rows: ``[M, D] -> [M, D]``.

        ``x`` holds the routed rows sorted by expert and ``group_sizes[e]`` how
        many belong to expert ``e`` (summing to ``M``).  One grouped matmul per
        weight via :func:`jax.lax.ragged_dot`: each row costs exactly one
        expert, with no capacity padding and nothing dropped.
        """
        self._check_rows(x)
        w_gate, w_up, w_down = (m.kernel[...] for m in (self.gate, self.up, self.down))
        # Same promotion as the einsum paths, so all three agree numerically.
        cdt = jnp.result_type(x.dtype, w_gate.dtype)
        x = x.astype(cdt)
        group_sizes = group_sizes.astype(jnp.int32)
        gate = jax.lax.ragged_dot(x, w_gate.astype(cdt), group_sizes)
        up = jax.lax.ragged_dot(x, w_up.astype(cdt), group_sizes)
        y = jax.lax.ragged_dot(jax.nn.silu(gate) * up, w_down.astype(cdt), group_sizes)
        return y if self.dtype is None else y.astype(self.dtype)

    def grouped_forward(
        self,
        x: Array,
        group_sizes: Array,
        *,
        block_m: int = 64,
        interpret: bool | None = None,
    ) -> Array:
        """:meth:`ragged_forward` through a Pallas grouped-GEMM kernel: jax's
        Mosaic-GPU kernel on Hopper, probjax's Triton kernel on other GPUs.

        Computes in ``self.dtype`` (or ``x.dtype``) with fp32 accumulation, so
        a bf16 model runs bf16 tensor-core matmuls.
        """
        self._check_rows(x)
        w_gate, w_up, w_down = (m.kernel[...] for m in (self.gate, self.up, self.down))
        cdt = self.dtype if self.dtype is not None else x.dtype
        if grouped_backend() == "mosaic" and not interpret:
            # Hopper: jax's Mosaic-GPU kernels take the sorted rows directly.
            mm = functools.partial(ragged_matmul_hopper, group_sizes=group_sizes)
            xs = x.astype(cdt)
            h = jax.nn.silu(mm(xs, w_gate)) * mm(xs, w_up)
            return mm(h, w_down).astype(cdt)
        # One VJP for the whole block, saving only inputs: see grouped_swiglu.
        y = grouped_swiglu(
            x.astype(cdt),
            w_gate.astype(cdt),
            w_up.astype(cdt),
            w_down.astype(cdt),
            group_sizes,
            block_m,
            interpret,
        )
        return y.astype(cdt)


class MoELayer(nnx.Module):
    """Top-k MoE layer: linear router + :class:`ExpertSwiGLU` experts.

    The router always runs in float32 (top-k selection is precision
    sensitive), independent of ``dtype``.

    ``routing`` picks how tokens reach their experts:

    * ``"capacity"`` (``sparse=True``): sort assignments by expert into a
      fixed ``[E, C, D]`` buffer; assignments beyond capacity ``C`` are
      dropped, so a token's output depends on the other tokens in the call.
    * ``"dense"`` (``sparse=False``): every expert on every token — exact,
      ``E / top_k`` times the compute, often fastest for small ``E``.
    * ``"dropless"``: rows sorted by expert and one grouped matmul — exact
      like dense at ``top_k`` experts per token. ``dropless_impl="auto"``
      uses the Pallas grouped-GEMM kernel on GPU and ``ragged_dot``
      elsewhere (on an A100 ``ragged_dot`` took 134 ms fwd+bwd per layer at
      65k tokens, slower than dense at 74 ms; see
      :mod:`probjax.nn.layers.grouped_matmul`).

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
        routing: Literal["capacity", "dense", "dropless"] | None = None,
        dropless_impl: Literal["auto", "pallas", "ragged_dot"] = "auto",
        kernel_init: Initializer = _expert_init,
        router_init: Initializer = nnx.initializers.lecun_normal(),
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
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
        if routing is None:
            routing = "capacity" if sparse else "dense"
        if routing not in ("capacity", "dense", "dropless"):
            raise ValueError(
                f"routing must be 'capacity', 'dense' or 'dropless', got {routing!r}"
            )
        if dropless_impl not in ("auto", "pallas", "ragged_dot"):
            raise ValueError(f"unknown dropless_impl {dropless_impl!r}")
        self.routing = routing
        self.dropless_impl = dropless_impl

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

    def _dropless(
        self, x: Array, expert_ids: Array, weights: Array, counts: Array
    ) -> Array:
        n_tokens, d_model = x.shape
        order = jnp.argsort(expert_ids.reshape(-1), stable=True)
        inverse = jnp.argsort(order)
        # Each token repeated per assignment, then permuted into expert order.
        x_rep = jnp.broadcast_to(x[:, None, :], (n_tokens, self.top_k, d_model))
        x_sorted = _permute(x_rep.reshape(-1, d_model), order, inverse)
        impl = self.dropless_impl
        if impl == "auto":
            impl = "pallas" if jax.default_backend() == "gpu" else "ragged_dot"
        if impl == "pallas":
            y_sorted = self.experts.grouped_forward(x_sorted, counts)
        else:
            y_sorted = self.experts.ragged_forward(x_sorted, counts)
        y = _permute(y_sorted, inverse, order).reshape(n_tokens, self.top_k, -1)
        return jnp.sum(y * weights[..., None].astype(y.dtype), axis=1)

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

        if self.routing == "capacity":
            y = self._sparse(x_flat, expert_ids, weights, counts, capacity)
            kept = jnp.minimum(counts, capacity)
        elif self.routing == "dense":
            y = self._dense(x_flat, expert_ids, weights)
            kept = counts
        else:
            y = self._dropless(x_flat, expert_ids, weights, counts)
            # No capacity: an expert may take every token, none are dropped.
            capacity, kept = n_tokens, counts
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
