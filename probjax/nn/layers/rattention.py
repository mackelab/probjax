# Some of the code in this file is adapted from:
# Apple/axlearn
# https://github.com/apple/axlearn/tree/main/axlearn/common/rattention

"""Local-global RAttention built on probjax's multi-head attention API."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from

from probjax.nn.layers.attention import MultiHeadAttention, dot_product_attention
from probjax.nn.pallas_kernels import (
    AttentionBias,
    AttentionMask,
    CausalLocalWindowMask,
)
from probjax.nn.pallas_kernels.kernels.linear_attention import (
    FeatureMap,
    get_feature_map,
    residual_linear_attention,
)
from probjax.nn.sharding import BATCH, constrain
from probjax.utils.typing import Array, ArrayLike, DTypeLike


class GroupRMSNorm(nnx.Module):
    """Per-head RMS normalization for tensors shaped ``[B, T, H, D]``."""

    def __init__(
        self,
        num_groups: int,
        group_dim: int,
        *,
        epsilon: float = 1e-8,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: rnglib.Rngs,
    ):
        del rngs
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.epsilon = epsilon
        self.dtype = dtype
        self.scale = nnx.Param(jnp.ones((num_groups, group_dim), dtype=param_dtype))

    def __call__(self, x: Array) -> Array:
        if x.shape[-2:] != (self.num_groups, self.group_dim):
            raise ValueError(
                "Expected trailing dimensions "
                f"{(self.num_groups, self.group_dim)}, got {x.shape[-2:]}"
            )
        original_dtype = x.dtype
        x = x.astype(jnp.float32)
        mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(mean_square + self.epsilon)
        x = x * self.scale[...].astype(x.dtype)
        return x.astype(self.dtype or original_dtype)


class ResidualLinearAttention(nnx.Module):
    """Parameter-light linear branch for tokens outside a local window."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        window_size: int,
        feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
        chunk_size: int = 128,
        implementation: str = "auto",
        use_learned_init: bool = False,
        use_qk_scale: bool = False,
        num_meta_tokens: int = 128,
        param_dtype: DTypeLike = jnp.float32,
        rngs: rnglib.Rngs,
    ):
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if window_size < -1:
            raise ValueError("window_size must be at least -1")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.feature_map = FeatureMap(feature_map)
        self.chunk_size = chunk_size
        self.implementation = implementation

        if use_learned_init:
            scale = head_dim**-0.5
            self.k_tokens = nnx.Param(
                scale
                * jax.random.normal(
                    rngs.params(),
                    (num_heads, num_meta_tokens, head_dim),
                    dtype=param_dtype,
                )
            )
            self.v_tokens = nnx.Param(
                scale
                * jax.random.normal(
                    rngs.params(),
                    (num_heads, num_meta_tokens, head_dim),
                    dtype=param_dtype,
                )
            )
        else:
            self.k_tokens = None
            self.v_tokens = None

        if use_qk_scale:
            self.q_scale = nnx.Param(jnp.ones((head_dim,), dtype=param_dtype))
            self.k_scale = nnx.Param(jnp.ones((head_dim,), dtype=param_dtype))
        else:
            self.q_scale = None
            self.k_scale = None

    def _initial_state(self, batch_size: int, dtype) -> Array:
        if self.k_tokens is None:
            return jnp.zeros(
                (batch_size, self.num_heads, 2 * self.head_dim, self.head_dim),
                dtype=jnp.float32,
            )
        assert self.v_tokens is not None
        k = get_feature_map(self.feature_map).fwd(self.k_tokens[...])
        state = jnp.einsum(
            "htk,htv->hkv",
            k,
            self.v_tokens[...],
            preferred_element_type=jnp.float32,
        )
        return jnp.broadcast_to(state[None], (batch_size,) + state.shape).astype(dtype)

    def __call__(self, q: Array, k: Array, v: Array) -> Array:
        if self.q_scale is not None:
            assert self.k_scale is not None
            q = q * self.q_scale[...]
            k = k * self.k_scale[...]
        h0 = self._initial_state(q.shape[0], jnp.float32)
        output, _ = residual_linear_attention(
            jnp.swapaxes(q, 1, 2),
            jnp.swapaxes(k, 1, 2),
            jnp.swapaxes(v, 1, 2),
            h0,
            window_size=self.window_size,
            feature_map=self.feature_map,
            chunk_size=self.chunk_size,
            implementation=self.implementation,
        )
        return jnp.swapaxes(output, 1, 2)


def _apply_rope(x: Array, theta: float) -> Array:
    """Apply RoFormer-style rotary embeddings to ``[B, T, H, D]``."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RAttention requires an even head dimension for RoPE")
    dtype = jnp.float32
    inv_freq = theta ** (-jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim)
    angles = jnp.arange(x.shape[1], dtype=dtype)[:, None] * inv_freq[None, :]
    cos = jnp.concatenate((jnp.cos(angles), jnp.cos(angles)), axis=-1)[None, :, None, :]
    sin = jnp.concatenate((jnp.sin(angles), jnp.sin(angles)), axis=-1)[None, :, None, :]
    first, second = jnp.split(x.astype(dtype), 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return (x.astype(dtype) * cos + rotated * sin).astype(x.dtype)


class RAttention(MultiHeadAttention):
    """Residual linear attention combined with causal sliding-window attention.

    Set ``residual_la=False`` for local-only attention, or set
    ``sliding_window_size=-1`` for pure global linear attention.
    """

    def __init__(
        self,
        *args,
        sliding_window_size: int = 512,
        feature_map: FeatureMap | str = FeatureMap.SOFTMAX,
        chunk_size: int = 128,
        linear_attention_implementation: str = "auto",
        residual_la: bool = True,
        use_learned_init: bool = False,
        use_qk_scale: bool = False,
        rope_theta: float = 500_000.0,
        **kwargs,
    ):
        if sliding_window_size < -1:
            raise ValueError("sliding_window_size must be at least -1")
        if not residual_la and sliding_window_size == -1:
            raise ValueError("At least one attention branch must be enabled")
        rngs: rnglib.Rngs = kwargs["rngs"]
        kwargs.setdefault("attention_fn", dot_product_attention)
        super().__init__(*args, **kwargs)
        self.sliding_window_size = sliding_window_size
        self.rope_theta = rope_theta
        self.residual_la_enabled = residual_la

        if residual_la:
            self.residual_linear_attention = ResidualLinearAttention(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                window_size=sliding_window_size,
                feature_map=feature_map,
                chunk_size=chunk_size,
                implementation=linear_attention_implementation,
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.rla_norm = GroupRMSNorm(
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.residual_linear_attention = None
            self.rla_norm = None

        if sliding_window_size >= 0 and residual_la:
            self.swa_norm = GroupRMSNorm(
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.swa_norm = None

    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: AttentionMask | ArrayLike | None = None,
        bias: AttentionBias | ArrayLike | None = None,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        sow_weights: bool = False,
        decode: bool | None = False,
        kv_len: int | Array | None = None,
    ) -> Array:
        del kv_len
        if inputs_k is not None or inputs_v is not None:
            raise ValueError("RAttention only supports self-attention")
        decode = first_from(
            decode,
            self.decode,
            error_msg="No decode argument was provided to RAttention.",
        )
        if decode:
            raise NotImplementedError("RAttention decoding is not implemented yet")
        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input dimension {self.in_features}, got {inputs_q.shape[-1]}"
            )

        query = self.query(inputs_q)
        key = self.key(inputs_q)
        value = self.value(inputs_q)
        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            query = self.query_ln(query)
            key = self.key_ln(key)

        rla_output = None
        if self.residual_linear_attention is not None:
            rla_output = self.residual_linear_attention(query, key, value)

        swa_output = None
        if self.sliding_window_size >= 0:
            local_mask = CausalLocalWindowMask(self.sliding_window_size)
            if mask is None:
                mask = local_mask
            elif isinstance(mask, AttentionMask):
                mask = mask & local_mask
            else:
                dense_local = local_mask.dense(
                    query.shape[1],
                    key.shape[1],
                    batch_size=query.shape[0],
                    num_heads=self.num_heads,
                )
                mask = jnp.logical_and(jnp.asarray(mask), dense_local)

            if rng is not None and rngs is None:
                rngs = cast(rnglib.RngStream, lambda: rng)
            if rngs is None:
                rngs = self.rngs
            elif isinstance(rngs, rnglib.Rngs):
                rngs = rngs.dropout
            if self.dropout_rate > 0:
                deterministic = first_from(
                    deterministic,
                    self.deterministic,
                    error_msg="deterministic is required when dropout is enabled",
                )
                if not deterministic and rngs is None:
                    raise ValueError("rngs is required when dropout is enabled")
                dropout_rng = None if deterministic else rngs()
            else:
                deterministic = True
                dropout_rng = None

            swa_output = self.attention_fn(
                _apply_rope(query, self.rope_theta),
                _apply_rope(key, self.rope_theta),
                value,
                mask=mask,
                bias=bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=self.precision,
                module=self if sow_weights else None,
            )

        if swa_output is None:
            assert rla_output is not None and self.rla_norm is not None
            context = self.rla_norm(rla_output)
        elif rla_output is None:
            context = swa_output
        else:
            assert self.swa_norm is not None and self.rla_norm is not None
            context = self.swa_norm(swa_output) + self.rla_norm(rla_output)
        return constrain(self.out(context), BATCH)


__all__ = ["GroupRMSNorm", "RAttention", "ResidualLinearAttention"]
