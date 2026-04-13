from __future__ import annotations

import math
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import MultiHeadAttention as FlaxMultiHeadAttention
from flax.nnx import combine_masks, rnglib
from flax.nnx import dot_product_attention as flax_dot_product_attention
from flax.nnx.module import first_from
from flax.typing import Dtype
from jax import lax

from probjax.nn.pallas_kernels import (
    AttentionBias,
    AttentionMask,
    BlockSizes,
    QKVLengthMask,
    mha,
)
from probjax.nn.sharding import LinearShardingCfg, ShardingCfg
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
    pad_to_power_of_2,
)
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLikeType,
    PrecisionLike,
)


# ---------------------------------------------------------------------------
# Query scaling modules (pre-kernel Q scaling for SSMax / QASSMax / etc.)
# ---------------------------------------------------------------------------


def _zero_init_last_layer(mlp: Any, bias_value: float | None = None) -> None:
    """Zero the last layer's kernel of an MLP so it starts as a no-op.

    Optionally set the last layer's bias to a constant (e.g. 1.0 so the
    MLP initially outputs that constant everywhere).
    """
    last = mlp.layers[-1]
    last.kernel[...] = jnp.zeros_like(last.kernel[...])
    if bias_value is not None and last.bias is not None:
        last.bias[...] = jnp.full_like(last.bias[...], bias_value)


class SSMaxQueryScale(nnx.Module):
    """Scalable-Softmax (SSMax) per-head query scaling.

    Scales each query by ``s_h * log(n)`` where *s* is a learnable per-head
    scalar and *n* is the number of keys (KV sequence length).  This
    compensates for "attention fading" as the context grows.

    At initialisation ``s = 1`` so the effective scaling is just ``log(n)``.

    Reference: *Scalable Softmax* (https://arxiv.org/abs/2501.14222).

    The module is designed to be passed to :class:`MultiHeadAttention` via
    ``q_scale_cls`` (or the legacy ``query_scale`` argument)::

        mha = MultiHeadAttention(
            ...,
            q_scale_cls=SSMaxQueryScale,
        )

    Args:
        num_heads: number of attention heads.
        min_scale: minimum allowed per-head scale.
        max_scale: maximum allowed per-head scale.
        param_dtype: dtype for the learnable scalar.
        rngs: random number generators.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int | None = None,
        *,
        min_scale: float = 0.0,
        max_scale: float = 4.0,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: rnglib.Rngs,
    ):
        del head_dim, dtype
        if min_scale > max_scale:
            raise ValueError("`min_scale` must be <= `max_scale`.")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.s = nnx.Param(jnp.ones((num_heads,), dtype=param_dtype))

    def __call__(self, query: Array, *, kv_len: int | Array) -> Array:
        """Scale *query* by ``s * log(kv_len)``.

        Args:
            query: projected queries, shape ``[batch, length, num_heads, head_dim]``.
            kv_len: number of keys the query will attend over.  Can be a
                scalar (same for all examples) or a per-batch array of shape
                ``[batch]`` / ``[batch, 1]`` for variable-length sequences.

        Returns:
            Scaled queries (same shape).
        """
        kv_len = jnp.asarray(kv_len, dtype=jnp.float32)
        log_n = jnp.log(kv_len + 1.0)
        # Reshape log_n so it broadcasts with query [B, L, H, D].
        # scalar → works as-is; [B] → [B, 1, 1, 1]; [B, 1] → [B, 1, 1, 1]
        if log_n.ndim >= 1:
            log_n = log_n.reshape(-1, *([1] * (query.ndim - 1)))
        # s: [H] → [1, 1, H, 1]
        s = jnp.clip(self.s[...], self.min_scale, self.max_scale)
        scale = s[None, None, :, None] * log_n
        return query * scale


class PerHeadQueryScale(nnx.Module):
    """Simple learnable per-head query scaling.

    Applies a learnable scalar ``s_h`` to each attention head:

    ``q'[:, :, h, :] = s_h * q[:, :, h, :]``.

    This module follows the same call signature as SSMax/QASSMax and can be
    passed to :class:`MultiHeadAttention` via ``q_scale_cls``.

    Args:
        num_heads: number of attention heads.
        init_value: initial value for each head scale.
        min_scale: minimum allowed per-head scale.
        max_scale: maximum allowed per-head scale.
        param_dtype: dtype for the learnable scale parameters.
        rngs: random number generators (accepted for API consistency).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int | None = None,
        *,
        init_value: float = 1.0,
        min_scale: float = 0.0,
        max_scale: float = 4.0,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: rnglib.Rngs,
    ):
        del head_dim, dtype, rngs
        if min_scale > max_scale:
            raise ValueError("`min_scale` must be <= `max_scale`.")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.s = nnx.Param(jnp.full((num_heads,), init_value, dtype=param_dtype))

    def __call__(self, query: Array, *, kv_len: int | Array) -> Array:
        del kv_len
        s = jnp.clip(self.s[...], self.min_scale, self.max_scale)
        return query * s[None, None, :, None]


class QASSMaxQueryScale(nnx.Module):
    """Query-Aware Scalable Softmax (QASSMax) per-head query scaling.

    A richer variant of SSMax where the scaling is both *per-element* and
    *query-dependent*:

    .. math::

        \\tilde q_{h,i} = q_{h,i} \\odot \\text{base}_h(\\log n)
                          \\odot (1 + \\tanh(\\text{gate}_h(q_{h,i})))

    * ``base_h(log n)``: a per-head 2-layer MLP that maps the scalar
      ``log(n)`` to a ``[num_heads * head_dim]`` vector (reshaped to
      ``[num_heads, head_dim]``).  Zero-init output with bias = 1 so it
      starts as identity scaling.
    * ``gate_h(q)``: a 2-layer MLP operating on the last axis (``head_dim``)
      of the query tensor.  Because query has shape
      ``[batch, len, num_heads, head_dim]``, the MLP naturally broadcasts
      over batch/length/heads and produces head-specific outputs (different
      query → different gate).  Zero-init output so ``1 + tanh(0) = 1``
      at initialisation.

    Combined, at init: ``q' = q * 1 * 1 = q`` (identity).

    Reference: *TabICLv2* (https://arxiv.org/abs/2506.05196).

    Args:
        num_heads: number of attention heads.
        head_dim: dimension of each head.
        hidden_dim: hidden dimension for both MLPs.
        param_dtype: dtype for learnable parameters.
        dtype: computation dtype (forwarded to MLP).
        rngs: random number generators.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        hidden_dim: int = 64,
        param_dtype: DTypeLike | None = None,
        dtype: DTypeLike | None = None,
        rngs: rnglib.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Lazy import to avoid circular dependency
        # (attention → nets.simple → nets.__init__ → autoregressive → attention).
        from probjax.nn.nets.simple import MLP

        mlp_kwargs: dict[str, Any] = {}
        if param_dtype is not None:
            mlp_kwargs["param_dtype"] = param_dtype
        if dtype is not None:
            mlp_kwargs["dtype"] = dtype

        # Base MLP: scalar log(n) → [num_heads * head_dim].
        # Starts outputting 1.0 everywhere (identity scaling).
        self.base_mlp = MLP(
            feature_dims=[1, hidden_dim, num_heads * head_dim],
            rngs=rngs,
            **mlp_kwargs,
        )
        _zero_init_last_layer(self.base_mlp, bias_value=1.0)

        # Gate MLP: [head_dim] → [head_dim], shared weights across heads.
        # Applied to query of shape [..., num_heads, head_dim] — the Linear
        # operates on the last axis and broadcasts over all leading dims,
        # so each head gets a different output (different query vector in).
        # Starts outputting 0.0 → 1 + tanh(0) = 1.0 (identity).
        self.gate_mlp = MLP(
            feature_dims=[head_dim, hidden_dim, head_dim],
            rngs=rngs,
            **mlp_kwargs,
        )
        _zero_init_last_layer(self.gate_mlp)

    def __call__(self, query: Array, *, kv_len: int | Array) -> Array:
        """Scale *query* using base and gate MLPs.

        Args:
            query: projected queries, shape ``[batch, length, num_heads, head_dim]``.
            kv_len: number of keys the query will attend over.  Can be a
                scalar (same for all examples) or a per-batch array of shape
                ``[batch]`` / ``[batch, 1]`` for variable-length sequences.

        Returns:
            Scaled queries (same shape).
        """
        kv_len = jnp.asarray(kv_len, dtype=jnp.float32)
        log_n = jnp.log(kv_len + 1.0)

        # Base MLP: f(log_n) → per-head per-dim scale.
        # Input: scalar → [1, 1] or per-batch [B] → [B, 1].
        log_n_flat = log_n.reshape(-1, 1)  # [B, 1] or [1, 1]
        base_flat = self.base_mlp(log_n_flat)  # [B, H*D] or [1, H*D]
        # Reshape to [B, 1, H, D] (or [1, 1, H, D] for scalar kv_len)
        # so it broadcasts with query [B, L, H, D].
        base = base_flat.reshape(-1, 1, self.num_heads, self.head_dim)

        # Gate: g(query) → [B, L, H, D], bounded (0, 2)
        gate = 1.0 + jnp.tanh(self.gate_mlp(query))

        return query * base * gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_qk_norm(
    *,
    cls: ModuleLikeType | None,
    head_dim: int,
    dtype: Any,
    param_dtype: Any,
    promote_dtype: Any,
    scale_metadata: dict,
    rngs: rnglib.Rngs,
) -> Any:
    """Instantiate a normalization layer for query or key projections.

    When *cls* is None, falls back to ``nnx.LayerNorm`` (the Flax default).
    For ``nnx.LayerNorm`` specifically, ``use_bias=False`` and
    ``scale_metadata`` are forwarded.
    """
    if cls is None:
        return nnx.LayerNorm(
            head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            promote_dtype=promote_dtype,
            rngs=rngs,
            scale_metadata=scale_metadata,
        )

    return cls(
        head_dim,
        dtype=dtype,
        param_dtype=param_dtype,
        promote_dtype=promote_dtype,
        rngs=rngs,
    )


def _build_q_scale(
    *,
    cls: ModuleLikeType,
    num_heads: int,
    head_dim: int,
    dtype: Any,
    param_dtype: Any,
    rngs: rnglib.Rngs,
) -> Any:
    """Instantiate a query-scaling module from a class.

    The constructor is called with ``num_heads``, ``head_dim``, ``dtype``,
    ``param_dtype``, and ``rngs``.
    """
    return cls(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )


class MultiHeadAttention(FlaxMultiHeadAttention):
    """Multi-head attention with optional QK normalization and query scaling.

    Extends Flax's ``MultiHeadAttention`` with sharding support, pluggable
    QK normalization layers, and a pre-kernel *query scaling* hook.

    **QK normalization** (``normalize_qk`` + ``normalize_{q,kv}_cls``)
    lets you swap in any norm layer (e.g. ``LpNorm`` for cosine-similarity
    attention).

    **Query scaling** (``q_scale_cls``) is applied *after* QK normalisation
    and *before* the attention kernel.  Because attention logits are linear
    in Q, multiplying Q is equivalent to multiplying the logits — which is
    exactly how SSMax and QASSMax are meant to be implemented.

    Args:
        normalize_qk: if True, normalize query and key projections before
            computing attention weights.
        normalize_q_cls: optional module *class* (constructor) used to build
            the query normalizer.  It is instantiated inside MHA with
            defaults (``rngs``, ``dtype``, ``param_dtype``, ``promote_dtype``).
            When
            ``None`` (default) and ``normalize_qk`` is True, falls back to
            ``nnx.LayerNorm``.
        normalize_k_cls: same as ``normalize_q_cls`` but for keys only.
            Values are not normalized.
        q_scale_cls: optional module class used to build the query scaling
            module. If provided, it is instantiated inside MHA with ``rngs``,
            ``num_heads``, ``head_dim``, ``dtype``, and ``param_dtype``.
    """

    def __init__(
        self,
        *args,
        sharding_cfg: ShardingCfg | None = None,
        sharding_spec=None,
        normalize_qk: bool = False,
        normalize_q_cls: ModuleLikeType | None = None,
        normalize_k_cls: ModuleLikeType | None = None,
        normalize_kv_cls: ModuleLikeType | None = None,
        q_scale_cls: ModuleLikeType | None = None,
        **kwargs,
    ):
        if normalize_k_cls is not None and normalize_kv_cls is not None:
            raise ValueError(
                "Pass either `normalize_k_cls` or `normalize_kv_cls`, not both."
            )
        key_norm_cls = (
            normalize_k_cls if normalize_k_cls is not None else normalize_kv_cls
        )

        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)
        cfg = self.sharding_cfg.as_type(LinearShardingCfg)
        spec = (
            sharding_spec
            if sharding_spec is not None
            else (cfg.mha_spec() if isinstance(cfg, LinearShardingCfg) else None)
        )
        if spec is not None:
            if spec.kernel is not None:
                init_fn = kwargs.get("kernel_init", nnx.initializers.lecun_normal())
                kwargs["kernel_init"] = self.sharding_cfg.partitioned_init(
                    init_fn, spec.kernel
                )
            if spec.bias is not None:
                init_fn = kwargs.get("bias_init", nnx.initializers.zeros)
                kwargs["bias_init"] = self.sharding_cfg.partitioned_init(
                    init_fn, spec.bias
                )
            self._activation_spec = spec.activation
        else:
            self._activation_spec = None

        # Grab rngs and metadata before passing kwargs to the parent.
        rngs: rnglib.Rngs = kwargs["rngs"]
        query_ln_scale_metadata = kwargs.get("query_ln_scale_metadata", {})
        key_ln_scale_metadata = kwargs.get("key_ln_scale_metadata", {})

        super().__init__(*args, normalize_qk=False, **kwargs)

        # Build QK normalization layers ourselves so we can swap in any class.
        if normalize_qk:
            self.normalize_qk = True
            self.query_ln = _build_qk_norm(  # type: ignore[assignment]
                cls=normalize_q_cls,
                head_dim=self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                promote_dtype=self.ln_promote_dtype,
                scale_metadata=query_ln_scale_metadata,
                rngs=rngs,
            )
            self.key_ln = _build_qk_norm(  # type: ignore[assignment]
                cls=key_norm_cls,
                head_dim=self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                promote_dtype=self.ln_promote_dtype,
                scale_metadata=key_ln_scale_metadata,
                rngs=rngs,
            )

        # Pre-kernel query scaling (SSMax, QASSMax, etc.).
        if q_scale_cls is not None:
            self._query_scale = _build_q_scale(
                cls=q_scale_cls,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self._query_scale = None

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
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
        inputs_q: input queries of shape `[batch_sizes..., length, features]`.
        inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
        inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
        mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
        deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic. The
            ``deterministic`` flag passed into the call method will take precedence
            over the ``deterministic`` flag passed into the constructor.
        rngs: rng key. The rng key passed into the call method will take
            precedence over the rng key passed into the constructor.
        sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection.
        decode: whether to prepare and use an autoregressive cache. The ``decode``
            flag passed into the call method will take precedence over the ``decode``
            flag passed into the constructor.
        kv_len: effective number of keys each query attends over.  Used by
            ``query_scale`` (SSMax / QASSMax) for the ``log(n)`` term.  Can be:

            - ``None`` (default): inferred from the key tensor shape, or from
              the cache index during autoregressive decoding.
            - A scalar ``int`` or 0-d array: same length for every example.
            - A 1-d array of shape ``[batch]``: per-example lengths for
              variable-length sequences (e.g. in a padded batch).

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
        if rng is not None and rngs is None:
            rngs = cast(rnglib.RngStream, lambda: rng)
        if rngs is None:
            rngs = self.rngs
        elif isinstance(rngs, rnglib.Rngs):
            rngs = rngs.dropout

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    '`inputs_k` cannot be None if `inputs_v` is not None. '
                    'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
                    'value to `inputs_k` and leave `inputs_v` as None.'
                )
            inputs_k = inputs_q
            if inputs_v is None:
                inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f'Incompatible input dimension, got {inputs_q.shape[-1]} '
                f'but module expects {self.in_features}.'
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        assert inputs_v is not None
        value = self.value(inputs_v)

        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
                as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            # Only supported with Array based attention masks for now.
            if mask is not None and not isinstance(mask, jax.Array):
                raise ValueError(
                    "Autoregressive caching with MultiHeadAttention only supports "
                    "Array based attention masks for now."
                )
            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    'Autoregressive cache not initialized, call ``init_cache`` first.'
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key[...].shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    'Autoregressive cache shape error, '
                    'expected query shape %s instead got %s.'
                    % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index[...]
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key[...], key, indices)
            value = lax.dynamic_update_slice(self.cached_value[...], value, indices)
            self.cached_key[...] = key
            self.cached_value[...] = value
            self.cache_index[...] += 1
            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
            )

        # Per-head query scaling (SSMax, QASSMax, etc.).
        # Applied after QK norm and after cache update so that kv_len
        # reflects the actual number of keys being attended to.
        # Because logits = Q @ K^T, scaling Q is equivalent to scaling
        # the logits — this is the "pre-kernel" trick from the SSMax paper.
        if self._query_scale is not None:
            if kv_len is not None:
                # Caller provided an explicit kv_len (scalar or [batch]).
                effective_kv_len = kv_len
            elif decode and self.cache_index is not None:
                # During autoregressive decoding, use the number of keys
                # actually filled in the cache (cur_index was incremented
                # above, so cache_index already equals the count).
                effective_kv_len = self.cache_index[...]
            else:
                effective_kv_len = key.shape[-3]
            query = self._query_scale(query, kv_len=effective_kv_len)

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""No `deterministic` argument was provided to MultiHeadAttention
                    as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided to __call__ method if "
                        "MultiHeadAttention instance is defined with keep_rngs=False."
                    )
                dropout_rng = rngs()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
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
        # back to the original inputs dimensions
        out = self.out(x)
        return self.sharding_cfg.constrain(out, self._activation_spec)


class InducedSelfAttention(nnx.Module):
    """Two-stage self-attention with learned inducing points (ISAB).

    Implements the Set Transformer's Induced Self-Attention Block using two
    Multihead Attention Blocks (MABs).  Each MAB is a full transformer-style
    block consisting of multi-head attention followed by a row-wise
    feedforward network, both with residual connections and layer
    normalization (pre-norm convention):

    .. code-block:: text

        H = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X, Y) = LayerNorm(H + rFF(H))

        ISAB(X) = MAB(X, MAB(I, X))

    where *I* are the learned inducing points and *rFF* is a row-wise
    feedforward (MLP) applied independently to each token.

    Args:
        in_features: input / output feature dimension.
        num_inducing_points: number of learned inducing points.
        num_heads: number of attention heads.
        attn_size: per-head dimension (defaults to ``in_features // num_heads``).
        widening_factor: expansion factor for the feedforward hidden dim.
        mlp_cls: feedforward MLP class (default: ``probjax.nn.nets.simple.MLP``).
        dropout_rate: dropout rate for MHA.
        q_scale_cls: optional query-scaling class for the inducing MHA.
        output_q_scale_cls: optional query-scaling class for the output MHA.
        norm_cls: normalization layer class (``None`` to disable norms).
        mha_cls: multi-head attention class.
        dtype: computation dtype.
        param_dtype: parameter dtype.
        precision: computation precision.
        preferred_element_type: output dtype cast.
        sharding_cfg: sharding configuration.
    """

    def __init__(
        self,
        in_features: int,
        *,
        num_inducing_points: int,
        rngs: nnx.Rngs,
        num_heads: int = 8,
        attn_size: int | None = None,
        widening_factor: int = 4,
        mlp_cls: ModuleLikeType | None = None,
        dropout_rate: float = 0.0,
        q_scale_cls: ModuleLikeType | None = None,
        output_q_scale_cls: ModuleLikeType | None = None,
        norm_cls: ModuleLikeType | None = nnx.LayerNorm,
        mha_cls: ModuleLikeType = MultiHeadAttention,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        sharding_cfg: ShardingCfg | None = None,
    ):
        if in_features <= 0:
            raise ValueError(f"`in_features` must be positive, got {in_features}.")
        if num_inducing_points <= 0:
            raise ValueError(
                f"`num_inducing_points` must be positive, got {num_inducing_points}."
            )

        self.in_features = in_features
        self.num_heads = num_heads
        self.num_inducing_points = num_inducing_points
        self.preferred_element_type = preferred_element_type
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

        qkv_features = (
            in_features // num_heads if attn_size is None else attn_size * num_heads
        )

        # Precision and dtype settings.
        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # --- MHA sub-layers ---
        self.inducing_attn = mha_cls(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=in_features,
            dropout_rate=dropout_rate,
            sharding_cfg=self.sharding_cfg,
            q_scale_cls=q_scale_cls,
            rngs=rngs,
            **filter_precision_kwargs(mha_cls, **precision_kwargs),
        )
        self.output_attn = mha_cls(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=in_features,
            dropout_rate=dropout_rate,
            sharding_cfg=self.sharding_cfg,
            q_scale_cls=output_q_scale_cls,
            rngs=rngs,
            **filter_precision_kwargs(mha_cls, **precision_kwargs),
        )

        # --- Row-wise feedforward (rFF) sub-layers ---
        if mlp_cls is None:
            from probjax.nn.nets.simple import MLP as _MLP

            mlp_cls = _MLP

        ff_dims = [in_features, widening_factor * in_features, in_features]
        self.inducing_ff = mlp_cls(
            ff_dims,
            rngs=rngs,
            **filter_precision_kwargs(mlp_cls, **precision_kwargs),
        )
        self.output_ff = mlp_cls(
            ff_dims,
            rngs=rngs,
            **filter_precision_kwargs(mlp_cls, **precision_kwargs),
        )

        # --- Normalization layers ---
        # Each MAB needs: pre-attn norm (on Q), pre-attn norm (on KV),
        # post-attn norm (on H before rFF).
        # MAB1 (inducing): inducing_norm (Q), input_norm (KV), inducing_ff_norm (pre-FF)
        # MAB2 (output):   output_norm (Q), hidden_norm (KV), output_ff_norm (pre-FF)
        norm_kwargs = (
            self.sharding_cfg.norm_kwargs(norm_cls) if norm_cls is not None else {}
        )
        if norm_cls is not None:
            self.inducing_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
            self.input_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
            self.inducing_ff_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
            self.output_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
            self.hidden_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
            self.output_ff_norm = norm_cls(in_features, rngs=rngs, **norm_kwargs)
        else:
            self.inducing_norm = None
            self.input_norm = None
            self.inducing_ff_norm = None
            self.output_norm = None
            self.hidden_norm = None
            self.output_ff_norm = None

        init_dtype = jnp.float32 if param_dtype is None else param_dtype
        inducing_init = nnx.initializers.normal(stddev=0.02)(
            rngs.params(),
            (num_inducing_points, in_features),
            init_dtype,
        )
        self.inducing_points = nnx.Param(inducing_init)

    @staticmethod
    def _maybe_norm(norm: nnx.Module | None, x: Array) -> Array:
        return x if norm is None else norm(x)

    def _mab(
        self,
        x: Array,
        y: Array,
        *,
        attn: Any,
        x_norm: Any,
        y_norm: Any,
        ff: Any,
        ff_norm: Any,
        deterministic: bool,
        rng: jax.Array | None,
        kv_len: int | Array | None = None,
    ) -> Array:
        """Single Multihead Attention Block (MAB).

        H = X + MHA(norm(X), norm(Y), norm(Y))   # attention + residual
        MAB(X, Y) = H + rFF(norm(H))              # feedforward + residual
        """
        # Attention sub-block (pre-norm residual).
        y_n = self._maybe_norm(y_norm, y)
        h = x + attn(
            self._maybe_norm(x_norm, x),
            y_n,
            y_n,
            deterministic=deterministic,
            rng=rng,
            kv_len=kv_len,
        )
        # Feedforward sub-block (pre-norm residual).
        out = h + ff(self._maybe_norm(ff_norm, h))
        return out

    def __call__(
        self,
        x: Array,
        *,
        x_kv: Array | None = None,
        deterministic: bool = True,
        rng: jax.Array | None = None,
        kv_len: int | Array | None = None,
    ) -> Array:
        """Apply induced self-attention.

        Args:
            x: input of shape ``[..., seq_len, features]``.
            x_kv: optional separate KV input for the inducing stage (MAB 1).
                When provided, ``MAB1(I, x_kv)`` compresses only ``x_kv``
                into the inducing hidden state, while ``MAB2(x, H)`` still
                queries with the full ``x``.  This is useful when you want
                the inducing bottleneck to capture information from a subset
                (e.g. train rows) while broadcasting back to all rows
                (e.g. test + train).  When ``None``, ``x`` is used for both
                stages as in the standard ISAB.
            deterministic: if ``True``, disable dropout.
            rng: optional PRNG key for dropout.
            kv_len: effective number of keys for query scaling
                (SSMax / QASSMax).  Forwarded to the inducing MHA call.
                Can be ``None`` (inferred from key shape), a scalar ``int``
                or 0-d array, or a per-batch array of shape ``[batch]``.

        Returns:
            Output of same shape as *x*.
        """
        x = jnp.asarray(x)
        if x.ndim < 2:
            raise ValueError(
                f"`x` must have shape [..., seq_len, features], got ndim={x.ndim}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible input dimension, got {x.shape[-1]} "
                f"but module expects {self.in_features}."
            )

        kv_input = x if x_kv is None else jnp.asarray(x_kv)

        inducing_points = self.inducing_points[...]
        inducing_points = jnp.broadcast_to(
            inducing_points,
            kv_input.shape[:-2] + inducing_points.shape,
        )

        # MAB 1: inducing points attend to kv_input  ->  H = MAB(I, X_kv)
        inducing_hidden = self._mab(
            inducing_points,
            kv_input,
            attn=self.inducing_attn,
            x_norm=self.inducing_norm,
            y_norm=self.input_norm,
            ff=self.inducing_ff,
            ff_norm=self.inducing_ff_norm,
            deterministic=deterministic,
            rng=rng,
            kv_len=kv_len,
        )

        # MAB 2: input attends to induced representation  ->  ISAB(X) = MAB(X, H)
        out = self._mab(
            x,
            inducing_hidden,
            attn=self.output_attn,
            x_norm=self.output_norm,
            y_norm=self.hidden_norm,
            ff=self.output_ff,
            ff_norm=self.output_ff_norm,
            deterministic=deterministic,
            rng=rng,
        )

        if self.preferred_element_type is not None:
            out = out.astype(self.preferred_element_type)
        return out


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: AttentionMask | Array | None = None,
    bias: AttentionBias | Array | None = None,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = False,
    deterministic=True,
    dtype=None,
    precision=None,
    module=None,  # Required arguments by Flax
    sm_scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    batch_size, q_len, num_heads, _ = query.shape
    kv_len = key.shape[-3]
    if isinstance(mask, AttentionMask):
        mask = mask.dense(q_len, kv_len, batch_size=batch_size, num_heads=num_heads)
    if isinstance(bias, AttentionBias):
        bias = bias.dense(q_len, kv_len, batch_size=batch_size, num_heads=num_heads)
    return flax_dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        bias=bias,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        broadcast_dropout=broadcast_dropout,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )


def flex_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: AttentionMask | None = None,
    bias: AttentionBias | None = None,
    dropout_rng=None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = False,
    deterministic=True,
    dtype=None,
    precision=None,
    module=None,  # Required arguments by Flax
    sm_scale: Optional[float] = None,
    enable_gqa: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    block_q_dkv: int = 64,
    block_kv_dkv: int = 64,
    block_q_dq: int = 64,
    block_kv_dq: int = 64,
    backward_pass_impl: str = "auto",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    dropout_impl: str = "counter",
    diff_mode: str = "reverse",
):
    # These can not be used by the pallas backend
    del (
        module,
        precision,
        broadcast_dropout,
    )

    if dtype is not None:
        query = query.astype(dtype)
        key = key.astype(dtype)
        value = value.astype(dtype)

    if (query.dtype != key.dtype) or (query.dtype != value.dtype):
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )

    if (query.ndim < 3) or (key.ndim < 3) or (value.ndim < 3):
        raise ValueError(
            f"Expected query, key, and value to all be at least 3 dimensional, but got query.ndim: "
            f"{query.ndim}, key.ndim: {key.ndim}, and value.ndim: {value.ndim} instead."
        )

    if (not enable_gqa) and query.shape[-2] != key.shape[-2]:
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.shape[-2]} and Hkv={key.shape[-2]}. "
            f"Try setting enable_gqa=True for GQA."
        )

    if enable_gqa:
        Hq = query.shape[2]
        Hkv = key.shape[2]
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={Hq} and Hkv={Hkv}."
            )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(query.shape[-1])

    query = query[None] if query.ndim == 3 else query

    *_, l_q, h, n = query.shape
    *_, l_kv, _, _ = key.shape
    query = pad_to_power_of_2(query, axis=(-3, -1))
    key = pad_to_power_of_2(key, axis=(-3, -1))
    value = pad_to_power_of_2(value, axis=(-3, -1))

    if query.shape[1] != l_q or key.shape[1] != l_kv:
        # Non power-of-2 sequence lengths, hence padding was applied.
        # But this will bias the results, if we don't mask out the padded
        # positions. So we create a mask for the padded positions.
        mask = (
            QKVLengthMask(
                q_length=l_q,
                kv_length=l_kv,
                block_sparse=False,
            )
            if mask is None
            else mask & QKVLengthMask(q_length=l_q, kv_length=l_kv, block_sparse=False)
        )

    mask = jax.tree_util.tree_map(pad_to_power_of_2, mask) if mask is not None else None

    # Compute backward-compatible block sizes (no external BlockSizes input)
    block_sizes = BlockSizes.init_default(
        query.shape[-3],
        key.shape[-3],
        block_q,
        block_k,
        block_q_dkv,
        block_kv_dkv,
        block_q_dq,
        block_kv_dq,
    )
    # Score modifier gradient is handled via bias classes in pallas kernels.

    # If compiling for CPU, enforce interpret mode
    if jax.default_backend() == "cpu" or (
        not isinstance(query, jax.Array) and query.device.platform == "cpu"
    ):
        interpret = True
    elif (
        block_sizes.block_q_dkv < 16
        or block_sizes.block_kv_dkv < 16
        or block_sizes.block_q_dq < 16
        or block_sizes.block_kv_dq < 16
    ):
        # Triton lowering requires matmul inner dims >= 16 for these kernels.
        # Fall back to interpret mode for very small backward tiles.
        interpret = True

    if deterministic:
        dropout_rate = 0.0
        dropout_rng = None

    output = mha(
        q=query,
        k=key,
        v=value,
        mask=mask,  # AttentionMaskBase or None
        bias=bias,  # AttentionBiasBase or None
        rng=dropout_rng,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        dropout_rate=dropout_rate,
        dropout_impl=dropout_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        diff_mode=diff_mode,
    )

    output = output[:, :l_q, :h, :n]

    return output
