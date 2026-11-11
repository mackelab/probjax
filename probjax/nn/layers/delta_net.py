"""Gated DeltaNet sequence mixer (Yang et al., 2024).

``GatedDeltaNetCell`` maps ``[B, L, D] -> [B, L, D]`` and can be used as the
``recurrent_cls`` of :class:`probjax.nn.nets.ssm.SSMModel`.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from probjax.nn.layers.ssm import RecurrentCell, ShortCausalConv
from probjax.nn.pallas_kernels.kernels.delta_rule import (
    gated_delta_rule,
    l2_normalize,
)
from probjax.nn.sharding import BATCH, constrain
from probjax.nn.utils import filter_precision_kwargs
from probjax.utils.typing import DTypeLike, PrecisionLike

__all__ = ["GatedDeltaNetCell", "ShortCausalConv"]


class GatedDeltaNetCell(RecurrentCell):
    """Gated DeltaNet token mixer mapping ``[B, L, D] -> [B, L, D]``.

    Per head (``Dk = head_dim``, ``Dv = expand_v * head_dim``)::

        q, k, v = SiLU(ShortConv(W x))           # per projection
        q, k    = L2Norm(q), L2Norm(k);  q *= Dk ** -0.5
        log_alpha = -exp(A_log) * softplus(W_a x + dt_bias)    # gate alpha
        beta      = sigmoid(W_b x)                              # write strength
        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t
        y   = W_o (RMSNorm(o) * SiLU(W_g x))

    With ``use_gate=False`` the decay is dropped (``alpha == 1``: DeltaNet).

    Reference: Yang, Kautz & Hatamizadeh, 2024 -- Gated Delta Networks:
    Improving Mamba2 with Delta Rule; Yang et al., 2024 -- Parallelizing Linear
    Transformers with the Delta Rule over Sequence Length.
    """

    def __init__(
        self,
        model_dim: int,
        rngs: nnx.Rngs,
        *,
        num_heads: int = 4,
        head_dim: int | None = None,
        expand_v: int = 1,
        conv_size: int = 4,
        use_gate: bool = True,
        chunk_size: int = 64,
        implementation: str = "auto",
        norm_eps: float = 1e-6,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
    ):
        num_heads = int(num_heads)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is None:
            head_dim = (
                model_dim // num_heads if model_dim % num_heads == 0 else model_dim
            )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = int(head_dim)
        self.value_dim = int(expand_v) * self.head_dim
        self.use_gate = use_gate
        self.chunk_size = chunk_size
        self.implementation = implementation
        self.norm_eps = norm_eps

        precision_kwargs = filter_precision_kwargs(
            nnx.Linear,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )
        hk = num_heads * self.head_dim
        hv = num_heads * self.value_dim
        lin = lambda i, o, **kw: nnx.Linear(  # noqa: E731
            i, o, rngs=rngs, **precision_kwargs, **kw
        )
        self.to_q = lin(model_dim, hk, use_bias=False)
        self.to_k = lin(model_dim, hk, use_bias=False)
        self.to_v = lin(model_dim, hv, use_bias=False)
        self.to_beta = lin(model_dim, num_heads)
        self.to_gate = lin(model_dim, hv, use_bias=False)
        self.to_out = lin(hv, model_dim, use_bias=False)
        self.conv_q = ShortCausalConv(hk, conv_size, rngs)
        self.conv_k = ShortCausalConv(hk, conv_size, rngs)
        self.conv_v = ShortCausalConv(hv, conv_size, rngs)
        self.norm_scale = nnx.Param(jnp.ones((self.value_dim,), jnp.float32))
        if use_gate:
            self.to_alpha = lin(model_dim, num_heads, use_bias=False)
            # Mamba-2 style decay: A ~ U[1, 16], dt ~ logU[dt_min, dt_max].
            key_a, key_dt = jax.random.split(rngs.params())
            a = jax.random.uniform(key_a, (num_heads,), minval=1.0, maxval=16.0)
            self.a_log = nnx.Param(jnp.log(a))
            dt = jnp.exp(
                jax.random.uniform(key_dt, (num_heads,))
                * (jnp.log(dt_max) - jnp.log(dt_min))
                + jnp.log(dt_min)
            )
            dt = jnp.maximum(dt, dt_init_floor)
            self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))
        else:
            self.to_alpha = None
            self.a_log = None
            self.dt_bias = None

    def _heads(self, x: jax.Array, dim: int) -> jax.Array:
        b, length, _ = x.shape
        return x.reshape(b, length, self.num_heads, dim).transpose(0, 2, 1, 3)

    def __call__(self, inputs: jax.Array, *, rng: jax.Array | None = None) -> jax.Array:
        del rng
        added_batch = inputs.ndim == 2
        if added_batch:
            inputs = inputs[None]
        b, length, _ = inputs.shape
        q = jax.nn.silu(self.conv_q(self.to_q(inputs)))
        k = jax.nn.silu(self.conv_k(self.to_k(inputs)))
        v = jax.nn.silu(self.conv_v(self.to_v(inputs)))
        q = l2_normalize(self._heads(q, self.head_dim)) * (self.head_dim**-0.5)
        k = l2_normalize(self._heads(k, self.head_dim))
        v = self._heads(v, self.value_dim)
        beta = jax.nn.sigmoid(self.to_beta(inputs).astype(jnp.float32))
        beta = beta.transpose(0, 2, 1)
        if self.use_gate:
            dt = jax.nn.softplus(
                self.to_alpha(inputs).astype(jnp.float32) + self.dt_bias[...]
            )
            log_alpha = (-jnp.exp(self.a_log[...]) * dt).transpose(0, 2, 1)
        else:
            log_alpha = None
        o, _ = gated_delta_rule(
            q,
            k,
            v,
            log_alpha,
            beta,
            chunk_size=self.chunk_size,
            implementation=self.implementation,
        )
        # Per-head RMSNorm, then the SiLU output gate.
        of = o.astype(jnp.float32)
        of = of * jax.lax.rsqrt(
            jnp.mean(of * of, axis=-1, keepdims=True) + self.norm_eps
        )
        of = of * self.norm_scale[...]
        o = of.transpose(0, 2, 1, 3).reshape(b, length, -1).astype(inputs.dtype)
        y = self.to_out(o * jax.nn.silu(self.to_gate(inputs)))
        if added_batch:
            y = y[0]
        return constrain(y, BATCH)
