"""Batched SwiGLU experts with an explicit expert axis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer

from probjax.nn.sharding import EXPERT, param_metadata
from probjax.utils.typing import Array


def _default_kernel_init() -> Initializer:
    return nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")


class ExpertSwiGLU(nnx.Module):
    """SwiGLU feed-forward block applied to every expert in parallel.

    Contract: ``[E, C, D] -> [E, C, D]`` where ``E`` is the expert axis and
    ``C`` the (padded) per-expert capacity.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        *,
        kernel_init: Initializer | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if d_model <= 0 or d_hidden <= 0:
            raise ValueError(
                f"d_model/d_hidden must be positive, got {d_model}/{d_hidden}"
            )
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_hidden = d_hidden

        init = kernel_init or _default_kernel_init()
        k1, k2, k3 = jax.random.split(rngs.params(), 3)

        # Expert-parallel sharding: only the leading expert axis is named
        # (inner dims stay replicated). Sharding HIDDEN over "model" as well
        # would silently mix tensor parallelism in; that stays opt-in.
        # No-op without an active mesh.
        def _mk(key: Array, shape: tuple[int, ...]) -> nnx.Param:
            return nnx.Param(
                init(key, shape, param_dtype),
                **param_metadata(EXPERT, None, None),
            )

        self.w_gate = _mk(k1, (num_experts, d_model, d_hidden))
        self.w_up = _mk(k2, (num_experts, d_model, d_hidden))
        self.w_down = _mk(k3, (num_experts, d_hidden, d_model))
        self.dtype = dtype

    def __call__(self, x: Array) -> Array:
        """Apply all experts to packed tokens.

        Args:
            x: packed tokens of shape ``[E, C, D]``.

        Returns:
            Expert outputs of shape ``[E, C, D]``.
        """
        if x.ndim != 3 or x.shape[0] != self.num_experts or x.shape[2] != self.d_model:
            raise ValueError(
                f"Expected [E={self.num_experts}, C, D={self.d_model}], got {x.shape}"
            )
        gate = jnp.einsum("ecd,edh->ech", x, self.w_gate[...])
        up = jnp.einsum("ecd,edh->ech", x, self.w_up[...])
        h = jax.nn.silu(gate) * up
        y = jnp.einsum("ech,ehd->ecd", h, self.w_down[...])
        if self.dtype is not None:
            y = y.astype(self.dtype)
        return y

    def dense_forward(self, x: Array) -> Array:
        """Evaluate every expert on every token: ``[N, D] -> [N, E, D]``.

        Correctness-first reference used by the dense MoE path and tests.
        """
        gate = jnp.einsum("nd,edh->neh", x, self.w_gate[...])
        up = jnp.einsum("nd,edh->neh", x, self.w_up[...])
        h = jax.nn.silu(gate) * up
        return jnp.einsum("neh,ehd->ned", h, self.w_down[...])
