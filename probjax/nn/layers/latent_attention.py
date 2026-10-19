# Adapted from DeepSeek-V3/V3.2 inference/model.py (MIT, deepseek-ai).
"""Multi-head latent attention (MLA): one shared KV cache, distinct K/V roles.

Conventional attention learns two projections of the hidden state

    K = f_K(H),  V = f_V(H),

so the cache stores ``2 * heads * head_dim`` numbers per token.  MLA instead
computes a single latent per token

    C = f_KV(H),

and derives the per-head keys and values from that latent:

    K_h = C @ W_K[h],   V_h = C @ W_V[h].

Two execution forms exist, and they differ only in *where* the per-head
projections are applied -- the cache is identical either way:

``materialized``
    Keys and values are formed explicitly (``heads * (qk_head_dim + v_head_dim)``
    numbers per token) and attended normally.  This is what the reference
    inference code calls the ``naive`` path.

``absorbed``
    Keys and values are the latent itself (``K_h = V_h = C``); the key
    projection is folded into the queries and the value projection into the
    output.  The attention head dimension then equals ``kv_lora_rank``.  This is
    the MQA/decoding form and is the cheapest to cache.

The distinction the two forms make explicit is that *sharing the cache* does
not require *sharing the functional roles* of keys and values: in
``materialized`` mode a head's key and value remain different linear maps of the
latent.  Subclasses can override :meth:`MultiHeadLatentAttention._kv_from_latent`
to change how the latent becomes a key/value pair without touching the cache,
the query projection, or the output projection.

RoPE is intentionally not applied: the motivating use (tabular in-context
attention) has no token order, so position must not enter the memory.

Bank sharing
------------
A layer built with ``kv_encoder=False`` omits the encoder and treats its
``inputs_kv`` argument as the bank itself.  One encoder can then serve many
attention layers, each with its own queries and outputs, so the expensive
``H -> C`` map is computed once per forward pass rather than once per layer.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib

from probjax.nn.layers.attention import _build_q_scale, flex_attention
from probjax.nn.layers.lowrank import LowRankLinear
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes
from probjax.nn.pallas_kernels.kernels.latent_attention import mla
from probjax.nn.sharding import BATCH, constrain
from probjax.utils.typing import (
    Array,
    DTypeLike,
    ModuleLikeType,
    PrecisionLike,
)

MODES = ("materialized", "absorbed")


def _linear(
    in_features: int,
    out_features: int,
    *,
    rngs: rnglib.Rngs,
    dtype: DTypeLike | None,
    param_dtype: DTypeLike,
    precision: PrecisionLike | None,
    use_bias: bool = False,
) -> nnx.Linear:
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        rngs=rngs,
    )


class MultiHeadLatentAttention(nnx.Module):
    """Attention over a shared ``kv_lora_rank``-wide latent cache.

    The default ``mode="absorbed"`` uses the latent itself as both keys and
    values (``K = V = C``, the MQA form): no projection is learned on the cache,
    and heads are distinguished only by their queries and output.  Because the
    logits then contract against the latent, the query head dimension **must**
    equal ``kv_lora_rank``; passing a different ``qk_head_dim`` raises rather
    than being silently overridden.

    ``mode="materialized"`` instead learns per-head up-projections
    ``K_h = C @ W_K[h]`` and ``V_h = C @ W_V[h]``, which frees ``qk_head_dim``
    but stores the expanded K/V (this is the reference inference "naive" path).

    Args:
        num_heads: number of query heads.
        in_features: query input dimension.
        kv_lora_rank: width of the shared latent (the cache entry).
        mode: ``"absorbed"`` (default; ``K = V = C``, no KV projection) or
            ``"materialized"`` (per-head K/V up-projections).
        out_features: output dimension (defaults to ``in_features``).
        in_kv_features: input dimension for the latent encoder (defaults to
            ``in_features``); set this for cross-attention where the memory has a
            different width than the queries.
        qk_head_dim: query/key head dimension in ``materialized`` mode
            (defaults to ``in_features // num_heads``).  In ``absorbed`` mode it
            is fixed to ``kv_lora_rank`` and any other value raises.
        v_head_dim: value head dimension (defaults to ``qk_head_dim`` in
            ``materialized`` mode and to ``kv_lora_rank`` in ``absorbed`` mode).
        q_lora_rank: optional low-rank query compression (DeepSeek-V3 uses one);
            ``None`` projects the queries in a single step.
        q_projection_cls: optional projection class used for the *factored*
            query path, called as ``q_projection_cls(in_features,
            num_heads*qk_head_dim, rank=q_lora_rank, norm_cls=..., rngs=...,
            dtype=..., param_dtype=..., precision=...)``.  Requires
            ``q_lora_rank``.  Defaults to :class:`LowRankLinear`, which stores
            only the two factors (no full-rank matrix) and places the query norm
            between them.
        latent_norm: RMS-normalize the latent before it is used as cache.  A
            norm keeps the shared entry on a fixed scale regardless of how many
            keys attend to it.
        attention_fn: attention kernel called as ``attention_fn(q, k, v, ...)``
            (default: :func:`flex_attention`).
        use_kernel: route the attention through the fused
            :func:`probjax.nn.pallas_kernels.mla`, which forms the per-head K/V
            inside the kernel and never materializes them.  Requires the default
            ``attention_fn`` (raises otherwise).
        block_sizes: block tiling for the fused kernel.
        kv_encoder: build the latent encoder ``wkv_a`` (+ ``latent_norm``).  Set
            ``False`` for layers that consume a bank encoded once and shared
            across layers: the layer then holds no encoder parameters and the
            positional ``inputs_kv`` is taken to *be* the bank.  Because
            ``latent_norm`` is dropped too, the bank must already be normalized
            (the encoder layer does this via its own ``latent_norm``).
        fold_heads: in ``absorbed`` mode (and when no mask/bias is given), fold
            the heads into the query axis so the shared cache is read as a single
            head -- neither training nor inference broadcasts ``K``/``V`` over
            heads.  The folded and unfolded forms agree exactly without dropout;
            with dropout the attention mask is indexed over ``[Q*H, 1, S]``
            instead of ``[Q, H, S]``, so the outputs are equivalent in
            distribution but not elementwise.  Masks/biases use the unfolded
            form.
        value_projection: in ``absorbed`` mode, apply ``wv`` to the attention
            output.  Set ``False`` to drop ``wv`` entirely so the attended latent
            maps straight through the shared ``out`` projection (stricter cache
            sharing, fewer parameters).
        query_scale_cls: optional query-scaling module class (e.g.
            :class:`~probjax.nn.layers.attention.QASSMaxQueryScale` or
            ``SSMaxQueryScale``), built as in :class:`MultiHeadAttention` and fed
            ``kv_len``.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        kv_lora_rank: int,
        mode: str = "absorbed",
        out_features: int | None = None,
        in_kv_features: int | None = None,
        qk_head_dim: int | None = None,
        v_head_dim: int | None = None,
        q_lora_rank: int | None = None,
        q_projection_cls: ModuleLikeType | None = None,
        latent_norm: bool = True,
        attention_fn: Callable[..., Array] = flex_attention,
        use_kernel: bool = False,
        block_sizes: BlockSizes | None = None,
        kv_encoder: bool = True,
        fold_heads: bool = True,
        value_projection: bool = True,
        query_scale_cls: ModuleLikeType | None = None,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike | None = None,
        rngs: rnglib.Rngs,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if num_heads <= 0 or kv_lora_rank <= 0:
            raise ValueError("num_heads and kv_lora_rank must be positive")
        if not value_projection and mode != "absorbed":
            raise ValueError(
                "value_projection=False is only supported in absorbed mode "
                "(K = V = C); materialized mode needs wv to form values"
            )

        self.num_heads = int(num_heads)
        self.in_features = int(in_features)
        self.in_kv_features = int(in_kv_features or in_features)
        self.kv_lora_rank = int(kv_lora_rank)
        self.mode = mode
        self.out_features = int(out_features or in_features)
        self.q_lora_rank = None if q_lora_rank is None else int(q_lora_rank)
        self.dropout_rate = float(dropout_rate)
        self.deterministic = deterministic
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.attention_fn = attention_fn
        if use_kernel and attention_fn is not flex_attention:
            raise ValueError(
                "use_kernel=True requires the default attention_fn; the fused "
                "mla kernel does not call a custom attention function"
            )
        self.use_kernel = bool(use_kernel)
        self.block_sizes = block_sizes or BlockSizes.get_default()
        self.kv_encoder = bool(kv_encoder)
        self.fold_heads = bool(fold_heads)
        self.value_projection = bool(value_projection)

        if mode == "materialized":
            self.qk_head_dim = int(qk_head_dim or self.in_features // num_heads)
        else:
            # Absorbed: logits contract against the latent, so the query head
            # dimension MUST equal the cache width.  A different requested value
            # is a configuration error, not something to silently override.
            if qk_head_dim is not None and int(qk_head_dim) != self.kv_lora_rank:
                raise ValueError(
                    "absorbed mode (K = V = C) requires qk_head_dim == "
                    f"kv_lora_rank, got qk_head_dim={qk_head_dim} and "
                    f"kv_lora_rank={self.kv_lora_rank}; use "
                    "mode='materialized' to learn a per-head K/V projection, or "
                    "set qk_head_dim=kv_lora_rank"
                )
            self.qk_head_dim = self.kv_lora_rank
        self.v_head_dim = int(
            v_head_dim
            if v_head_dim is not None
            else (self.qk_head_dim if mode == "materialized" else self.kv_lora_rank)
        )
        if not self.value_projection and self.v_head_dim != self.kv_lora_rank:
            # Without ``wv`` the attended latent stays ``kv_lora_rank``-wide, so
            # ``out`` reads ``num_heads * kv_lora_rank``; a different v_head_dim
            # cannot be honoured.  Same contract as absorbed ``qk_head_dim``.
            raise ValueError(
                "value_projection=False leaves the attention output at "
                f"kv_lora_rank={self.kv_lora_rank}, so v_head_dim must equal it; "
                f"got v_head_dim={self.v_head_dim}.  Keep the value projection to "
                "use a different v_head_dim"
            )

        # --- shared latent cache encoder: C = f_KV(inputs_kv) ---------------
        # A bank shared across layers is encoded once; the following layers set
        # ``kv_encoder=False`` and receive the precomputed latent, so they carry
        # no encoder parameters at all.
        self.wkv_a = (
            _linear(
                self.in_kv_features,
                self.kv_lora_rank,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
            )
            if self.kv_encoder
            else None
        )
        self.latent_norm = (
            nnx.RMSNorm(
                self.kv_lora_rank, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            if (self.kv_encoder and latent_norm)
            else None
        )

        # --- query projection ------------------------------------------------
        # Factored (``q_lora_rank``) form is one module -- ``LowRankLinear`` by
        # default -- rather than two linears plus a norm, so the norm lives
        # between the factors and no full-rank matrix is ever stored.
        query_out = num_heads * self.qk_head_dim
        if self.q_lora_rank is None:
            if q_projection_cls is not None:
                raise ValueError(
                    "q_projection_cls requires q_lora_rank (there is no factor to "
                    "project through); pass q_lora_rank or drop q_projection_cls"
                )
            self.q_projection: Any = _linear(
                self.in_features,
                query_out,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
            )
        else:
            projection_cls = q_projection_cls or LowRankLinear
            self.q_projection = projection_cls(
                self.in_features,
                query_out,
                rank=self.q_lora_rank,
                norm_cls=nnx.RMSNorm,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
            )

        # --- per-head K/V up-projections (materialized only) ---------------
        # wk/wv map the latent (``R``, the last axis) to a head slice, so the
        # fan-in is ``R`` -- not ``num_heads * qk_head_dim`` as the default
        # ``variance_scaling`` in/out axes would assume.  Scale 1.0 matches
        # ``nnx.Linear`` (there is no nonlinearity after the projection).
        init = nnx.initializers.variance_scaling(
            1.0,
            "fan_in",
            "truncated_normal",
            in_axis=-1,
            out_axis=-2,
            batch_axis=(0,),
        )
        if mode == "materialized":
            self.wk = nnx.Param(
                init(
                    rngs.params(),
                    (num_heads, self.qk_head_dim, self.kv_lora_rank),
                    param_dtype,
                )
            )
        else:
            self.wk = None
        # In absorbed mode the value projection is applied to the attention
        # output (R -> v_head_dim) rather than to the latent.  With
        # ``value_projection=False`` it is omitted entirely, so heads contribute
        # to the output directly through the shared ``out`` projection.
        if self.value_projection:
            self.wv = nnx.Param(
                init(
                    rngs.params(),
                    (num_heads, self.v_head_dim, self.kv_lora_rank),
                    param_dtype,
                )
            )
        else:
            self.wv = None

        # Optional pre-kernel query scaling (SSMax / QASSMax), fed the KV length
        # so the layer can adapt to context size like ``MultiHeadAttention``.
        self._query_scale = (
            _build_q_scale(
                cls=query_scale_cls,
                num_heads=num_heads,
                head_dim=self.qk_head_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if query_scale_cls is not None
            else None
        )

        self.out = _linear(
            num_heads * self.v_head_dim,
            self.out_features,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    # ------------------------------------------------------------------
    # Overridable role of the latent
    # ------------------------------------------------------------------
    def encode_latent(self, inputs_kv: Array) -> Array:
        """``H -> C``: the single entry written to the cache, ``[..., S, R]``.

        Raises if this layer was built with ``kv_encoder=False``; such a layer
        is meant to be handed a precomputed bank as its ``inputs_kv``.
        """
        if self.wkv_a is None:
            raise RuntimeError(
                "encode_latent is unavailable when kv_encoder=False; this layer "
                "expects a precomputed bank passed as its `inputs_kv` argument."
            )
        latent = self.wkv_a(inputs_kv)
        if self.latent_norm is not None:
            latent = self.latent_norm(latent)
        return latent

    def _kv_from_latent(self, latent: Array) -> tuple[Array, Array]:
        """``C -> (K, V)`` in ``materialized`` mode.

        This is the override point that gives keys and values distinct roles
        while the cache stays shared: ``K_h = C @ W_K[h]`` and
        ``V_h = C @ W_V[h]``, with independent ``W_K`` and ``W_V``.
        """
        wk = self.wk
        wv = self.wv
        assert wk is not None and wv is not None
        key = jnp.einsum("...sr,hdr->...shd", latent, wk[...])
        value = jnp.einsum("...sr,hdr->...shd", latent, wv[...])
        return key, value

    def _project_queries(self, inputs_q: Array) -> Array:
        q = self.q_projection(inputs_q)
        return q.reshape(*inputs_q.shape[:-1], self.num_heads, self.qk_head_dim)

    def _pack_heads(self, query: Array) -> Array:
        """Fold the head axis into the sequence axis: ``[..., Q, H, D] -> [..., Q*H, D]``.

        In ``absorbed`` mode the keys and values are the *shared* latent, so
        neither training nor inference needs to broadcast ``K``/``V`` over heads.
        Stacking the per-head queries along the query axis lets a single
        ``[..., S, R]`` cache serve every head in one call; the head axis is
        restored by :meth:`_unpack_heads`.  The query order is head-major within
        a token (``q_flat[i*H + h]``), matching :meth:`_unpack_heads`.
        """
        shape = query.shape
        return query.reshape(*shape[:-3], shape[-3] * shape[-2], shape[-1])

    def _unpack_heads(self, attn: Array, num_q: int) -> Array:
        """Inverse of :meth:`_pack_heads`: ``[..., Q*H, D] -> [..., Q, H, D]``."""
        return attn.reshape(*attn.shape[:-2], num_q, self.num_heads, attn.shape[-1])

    def _out_from_attention(self, attn: Array) -> Array:
        if self.mode == "absorbed" and self.wv is not None:
            # attn: [..., Q, H, R] -> fold W_V in, then the shared output proj.
            attn = jnp.einsum("...qhr,hdr->...qhd", attn, self.wv[...])
        return self.out(
            attn.reshape(*attn.shape[:-2], self.num_heads * self.v_head_dim)
        )

    def _absorbed_attention(
        self,
        query: Array,
        latent: Array,
        *,
        mask: Any,
        bias: Any,
        dropout_rng: jax.Array | None,
        deterministic: bool | None,
    ) -> Array:
        """Attention with ``K = V = C`` in absorbed mode.

        With ``fold_heads`` (and no mask/bias, which are expressed in token-index
        space) the head axis is folded into the query axis so that a single
        ``[..., S, R]`` cache serves every head in one single-head attention
        call: neither training nor inference broadcasts ``K``/``V`` over heads.
        Masks and biases force the unfolded (broadcast) form, which is
        numerically identical and only differs in how the cache is read.
        """
        num_q = query.shape[-3]
        if self.fold_heads and mask is None and bias is None:
            q_folded = self._pack_heads(query)[..., None, :]  # [..., Q*H, 1, R]
            kv = latent[..., None, :]  # [..., S, 1, R]: one shared head
            attn = self.attention_fn(
                q_folded,
                kv,
                kv,
                mask=None,
                bias=None,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=self.precision,
                module=self,
            )
            attn = attn[..., 0, :]  # [..., Q*H, R]
            return self._unpack_heads(attn, num_q)
        # Unfolded: broadcast the shared latent over heads explicitly.
        kv = jnp.broadcast_to(
            latent[..., None, :],
            latent.shape[:-1] + (self.num_heads, self.kv_lora_rank),
        )
        return self.attention_fn(
            query,
            kv,
            kv,
            mask=mask,
            bias=bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self,
        )

    # ------------------------------------------------------------------
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array | None = None,
        *,
        mask: Any = None,
        bias: Any = None,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        kv_len: int | Array | None = None,
    ) -> Array:
        """Attend ``inputs_q`` over the shared latent of ``inputs_kv``.

        ``inputs_kv`` defaults to ``inputs_q`` (self-attention).  Passing a
        different tensor gives cross-attention where the memory is encoded to a
        latent once and every query head reads from it.

        When the layer was built with ``kv_encoder=False``, ``inputs_kv`` is
        *already* the latent bank: it is used as the cache directly, with no
        encoder and no norm.  One bank can then drive several layers, each with
        its own queries and outputs::

            bank = encoder(H)          # computed once
            y1 = layer1(x, bank)       # kv_encoder=False
            y2 = layer2(x, bank)

        ``kv_len`` is fed to the optional query-scaling module (SSMax/QASSMax)
        as in :class:`MultiHeadAttention`; it defaults to the bank length.
        """
        inputs_q = jnp.asarray(inputs_q)
        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible query dimension, got {inputs_q.shape[-1]} "
                f"but module expects {self.in_features}."
            )

        if inputs_kv is None:
            inputs_kv = inputs_q
        inputs_kv = jnp.asarray(inputs_kv)
        if self.kv_encoder:
            # ``inputs_kv`` is raw memory: encode it to the shared latent.
            if inputs_kv.shape[-1] != self.in_kv_features:
                raise ValueError(
                    f"Incompatible kv dimension, got {inputs_kv.shape[-1]} "
                    f"but module expects {self.in_kv_features}."
                )
            latent = self.encode_latent(inputs_kv)
        else:
            # ``inputs_kv`` *is* the bank (already ``kv_lora_rank``-wide).
            if inputs_kv.shape[-1] != self.kv_lora_rank:
                raise ValueError(
                    f"Incompatible latent dimension, got {inputs_kv.shape[-1]} "
                    f"but this encoder-less layer expects {self.kv_lora_rank}."
                )
            latent = inputs_kv

        query = self._project_queries(inputs_q)
        if self._query_scale is not None:
            effective_kv_len = latent.shape[-2] if kv_len is None else kv_len
            query = self._query_scale(query, kv_len=effective_kv_len)

        # Resolve dropout once, before either branch: the fused kernel path also
        # needs the rng and an explicit `deterministic`.
        dropout_rng = None
        if self.dropout_rate == 0.0:
            deterministic = True
        else:
            if deterministic is None:
                deterministic = self.deterministic
            if not deterministic:
                stream = rngs.dropout if isinstance(rngs, rnglib.Rngs) else rngs
                if rng is not None:
                    dropout_rng = rng
                elif stream is not None:
                    dropout_rng = stream()
                else:
                    raise ValueError("rng or rngs required when dropout is enabled")

        if self.use_kernel:
            # Fused path: the kernel forms K/V from the latent in SRAM, so the
            # per-head K and V are never materialized.  In absorbed mode the
            # value projection is applied to the output, so the kernel sees
            # ``K = V = C`` (``wv=None``).
            #
            # The kernel takes rank-4 queries [B,Q,H,D] and a rank-3 cache
            # [B,S,R]; the flex path accepts any leading batch dims, so fold
            # them into the batch axis here and restore the shape afterwards.
            absorbed = self.mode == "absorbed"
            fold_kernel = absorbed and self.fold_heads and mask is None and bias is None
            if fold_kernel:
                # Folding multiplies the query length by the head count, and the
                # kernel requires that length to tile ``block_q``.  Fold only when
                # it does; otherwise fall back to the unfolded (identical) form,
                # which tiles per-token as usual.
                folded_q = query.shape[-3] * self.num_heads
                block_q = min(self.block_sizes.block_q, folded_q)
                fold_kernel = folded_q % block_q == 0
            if fold_kernel:
                # Fold heads into the query axis so the kernel's cache is read
                # once as a single shared head (no per-head broadcast).
                query = self._pack_heads(query)[..., None, :]
            if absorbed:
                wk, wv = None, None
            else:
                assert self.wk is not None and self.wv is not None
                wk, wv = self.wk[...], self.wv[...]
            lead = query.shape[:-3]
            q_flat = query.reshape(-1, *query.shape[-3:])
            kv_flat = latent.reshape(-1, *latent.shape[-2:])
            # The kernel requires matching dtypes across all operands; the
            # projections already carry the layer's compute dtype.
            compute_dtype = query.dtype
            cast = lambda x: None if x is None else x.astype(compute_dtype)  # noqa: E731
            attn = mla(
                q_flat,
                kv_flat,
                cast(wk),
                cast(wv),
                mask=mask,
                bias=bias,
                rng=dropout_rng,
                dropout_rate=0.0 if deterministic else self.dropout_rate,
                block_sizes=self.block_sizes,
            )
            attn = attn.reshape(*lead, *attn.shape[-3:])
            if fold_kernel:
                attn = self._unpack_heads(attn[..., 0, :], inputs_q.shape[-2])
            return constrain(self._out_from_attention(attn), BATCH)

        if self.mode == "materialized":
            key, value = self._kv_from_latent(latent)
            attn = self.attention_fn(
                query,
                key,
                value,
                mask=mask,
                bias=bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=self.precision,
                module=self,
            )
        else:
            attn = self._absorbed_attention(
                query,
                latent,
                mask=mask,
                bias=bias,
                dropout_rng=dropout_rng,
                deterministic=deterministic,
            )
        return constrain(self._out_from_attention(attn), BATCH)


__all__ = ["MultiHeadLatentAttention", "MODES", "LowRankLinear"]
