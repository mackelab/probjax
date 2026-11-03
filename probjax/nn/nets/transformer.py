from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from probjax.nn.layers.attention import (
    AttentionBias,
    AttentionMask,
    MultiHeadAttention,
    dot_product_attention,
)
from probjax.nn.layers.fuse import AdditiveBinaryFuse, AffineFuse
from probjax.nn.nets.simple import MLP
from probjax.nn.sharding import BATCH, EMBED, SEQ, constrain
from probjax.nn.utils import (
    DEFAULT_MODULE,
    filter_precision_kwargs,
    flatten_to_btd,
    normalize_attn_bias,
    normalize_attn_mask,
    resolve_default_module,
    restore_from_btd,
)
from probjax.utils.typing import DTypeLike, ModuleLikeType, PrecisionLike


class Transformer(nnx.Module):
    """A transformer stack, defaulting to width 256 and eight 32D heads."""

    model_dim: int  # Dimensionality of the embedding vectors.
    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    drop_path_rates: Sequence[float]  # Drop-path rate(s) per layer.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.

    norm_cls = nnx.LayerNorm
    context_fusion_cls = AffineFuse
    attn_fuse_cls = AdditiveBinaryFuse
    mlp_fuse_cls = AdditiveBinaryFuse
    mlp_cls = MLP
    linear_cls = nnx.Linear
    dropout_cls = nnx.Dropout
    mha_cls = MultiHeadAttention

    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        attn_size: int = 32,
        *,
        enable_cross_attention: bool = False,
        kv_in_features: Optional[int] = None,
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
        context_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        dropout_rate_attn: float | None = None,
        drop_path_rate: float | Sequence[float] = 0.0,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable = jax.nn.gelu,
        attention_fn: Optional[Callable] = None,
        cross_attention_fn: Optional[Callable] = None,
        initializer: Optional[nnx.Initializer] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        norm_cls: ModuleLikeType = DEFAULT_MODULE,
        context_fusion_cls: ModuleLikeType = DEFAULT_MODULE,
        attn_fuse_cls: ModuleLikeType = DEFAULT_MODULE,
        mlp_fuse_cls: ModuleLikeType = DEFAULT_MODULE,
        mlp_cls: ModuleLikeType = DEFAULT_MODULE,
        linear_cls: ModuleLikeType = DEFAULT_MODULE,
        dropout_cls: ModuleLikeType = DEFAULT_MODULE,
        mha_cls: ModuleLikeType = DEFAULT_MODULE,
        remat_policy: str | Callable | None = None,
        scan_layers: bool = False,
        rngs: nnx.Rngs,
    ):
        """Initialize a Transformer model.
        Args:
            model_dim (int): The dimension of the model's hidden states.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            attn_size (int): Size of each attention head.
            rngs (nnx.Rngs): Random number generator state.
            context_dim (Optional[int], optional): Dimension of additional context to be
                concatenated with transformer output. If None, no context is used.
                Defaults to None.
            dropout_rate (float, optional): Dropout rate. If 0.0, dropout is disabled.
                Defaults to 0.0.
            dropout_rate_attn (float | None, optional): Dropout rate for attention if
                None, uses `dropout_rate`.
            drop_path_rate (float | Sequence[float], optional): Drop-path rate(s) per
                layer. Provide a single float to apply uniformly, or a sequence of
                length `num_layers`. Defaults to 0.0.
            widening_factor (int, optional): Factor by which to increase the dimension
                in the MLP. Defaults to 4.
            num_hidden_layers (int, optional): Number of hidden layers in the MLP block.
                Defaults to 1.
            act (Callable, optional): Activation function. Defaults to jax.nn.gelu.
            attention_fn (Optional[Callable], optional): Custom attention function.
                If None, uses dot product attention. Defaults to None.
            cross_attention_fn (Optional[Callable], optional): Custom cross attention
                function. Defaults to dot product attention.
            attn_fuse_cls: Binary fusion module for residual connections in the
                attention block. Use None to disable the residual path. Defaults to
                AdditiveBinaryFuse which reproduces a standard residual add.
            mlp_fuse_cls: Binary fusion module for residual connections in the MLP
                block. Use None to disable the residual path. Defaults to
                AdditiveBinaryFuse.
            initializer (Optional[nnx.initializers.Initializer], optional): Weight
                initializer. If None, uses truncated normal with variance scaling.
                Defaults to None.
            remat_policy: Activation checkpointing (rematerialization) per
                layer. ``None`` (default) stores every activation for the
                backward pass. ``"full"`` (alias ``"nothing_saveable"``) keeps
                only each layer's input and recomputes the layer in the
                backward pass -- the largest memory saving for roughly one
                extra forward pass of compute. Any other
                ``jax.checkpoint_policies`` name (e.g. ``"dots_saveable"``,
                ``"dots_with_no_batch_dims_saveable"``) or a policy callable
                is passed to ``jax.checkpoint``. Outputs and gradients are
                unchanged; parameters are unaffected.
            scan_layers: If True, the layers' parameters are stacked on a
                leading ``[num_layers]`` axis (under ``self.layers``) and the
                forward pass is one ``lax.scan`` over them, so tracing and
                compile time no longer grow with depth. This changes the
                parameter tree (and per-seed initial values), not the
                function: the per-layer computation is the same. Needs a single
                ``drop_path_rate`` for all layers. Use :meth:`init_cache` for
                KV-cache decoding in either layout.
        """
        for name, value in {
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attn_size": attn_size,
            "widening_factor": widening_factor,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        norm_cls = resolve_default_module(self, "norm_cls", norm_cls)
        context_fusion_cls = resolve_default_module(
            self, "context_fusion_cls", context_fusion_cls
        )
        attn_fuse_cls = resolve_default_module(self, "attn_fuse_cls", attn_fuse_cls)
        mlp_fuse_cls = resolve_default_module(self, "mlp_fuse_cls", mlp_fuse_cls)
        mlp_cls = resolve_default_module(self, "mlp_cls", mlp_cls)
        linear_cls = resolve_default_module(self, "linear_cls", linear_cls)
        dropout_cls = resolve_default_module(self, "dropout_cls", dropout_cls)
        mha_cls = resolve_default_module(self, "mha_cls", mha_cls)

        precision_kwargs = dict(
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.model_dim = model_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor
        self.dropout_rate = dropout_rate
        self.dropout_rate_attn = (
            dropout_rate_attn if dropout_rate_attn is not None else dropout_rate
        )
        if isinstance(drop_path_rate, Sequence) and not isinstance(
            drop_path_rate, (str, bytes)
        ):
            if len(drop_path_rate) != num_layers:
                raise ValueError(
                    "drop_path_rate sequence length must match num_layers "
                    f"({num_layers}), got {len(drop_path_rate)}."
                )
            drop_path_rates = [float(x) for x in drop_path_rate]
        else:
            drop_path_rates = [float(drop_path_rate)] * num_layers

        self.drop_path_rates = drop_path_rates
        self.initializer = (
            nnx.initializers.variance_scaling(
                2 / self.num_layers, 'fan_in', 'truncated_normal'
            )
            if initializer is None
            else initializer
        )
        self.act = act
        self.enable_cross_attention = enable_cross_attention
        self.remat_policy = _resolve_remat_policy(remat_policy)
        self.scan_layers = bool(scan_layers)
        if self.scan_layers and len(set(drop_path_rates)) > 1:
            raise ValueError(
                "scan_layers=True needs the same drop_path_rate for every layer "
                f"(the scanned layer is one program); got {drop_path_rates}."
            )

        attention_fn = (
            attention_fn if attention_fn is not None else dot_product_attention
        )
        cross_attention_fn = (
            cross_attention_fn
            if cross_attention_fn is not None
            else dot_product_attention
        )
        dims = (
            [model_dim]
            + [widening_factor * model_dim] * num_hidden_layers
            + [model_dim]
        )
        linear = partial(linear_cls, kernel_init=self.initializer)

        # One factory per sub-module of a layer. The unrolled layout below
        # builds them in the historical order (all norms, then all attention
        # blocks, ...) so parameter values for a given seed are unchanged.
        def make_norm(rngs):
            return norm_cls(model_dim, rngs=rngs)

        def make_mha(rngs, cross=False):
            return mha_cls(
                num_heads=num_heads,
                in_features=model_dim,
                qkv_features=attn_size * num_heads,
                out_features=model_dim,
                **({"in_kv_features": kv_in_features} if cross else {}),
                rngs=rngs,
                kernel_init=self.initializer,
                dropout_rate=self.dropout_rate_attn,
                attention_fn=cross_attention_fn if cross else attention_fn,
                normalize_qk=normalize_qk_cross_attn if cross else normalize_qk_attn,
                **filter_precision_kwargs(mha_cls, **precision_kwargs),
            )

        def make_context_fusion(rngs):
            return context_fusion_cls(model_dim, context_dim, rngs=rngs)

        def make_mlp(rngs):
            return mlp_cls(
                dims,
                rngs=rngs,
                linear_cls=linear,
                activation=act,
                **filter_precision_kwargs(mlp_cls, **precision_kwargs),
            )

        def make_dropout(rngs):
            return dropout_cls(rate=dropout_rate, rngs=rngs)

        def make_fuse(fuse_cls, rate, rngs):
            return fuse_cls(
                model_dim,
                context_dim,
                drop_path_rate=rate,
                rngs=rngs,
                **filter_precision_kwargs(fuse_cls, **precision_kwargs),
            )

        if self.scan_layers:
            rate = drop_path_rates[0] if drop_path_rates else 0.0

            def make_layer(rngs):
                parts = dict(
                    norm_attn=make_norm(rngs),
                    norm_dense=make_norm(rngs),
                    attn=make_mha(rngs),
                    dense=make_mlp(rngs),
                    attn_fuse=make_fuse(attn_fuse_cls, rate, rngs),
                    mlp_fuse=make_fuse(mlp_fuse_cls, rate, rngs),
                )
                if enable_cross_attention:
                    parts.update(
                        norm_cross=make_norm(rngs),
                        cross_attn=make_mha(rngs, cross=True),
                        cross_fuse=make_fuse(attn_fuse_cls, rate, rngs),
                    )
                if context_dim is not None:
                    parts.update(
                        context1=make_context_fusion(rngs),
                        context2=make_context_fusion(rngs),
                    )
                if dropout_rate > 0.0:
                    parts.update(dropout=make_dropout(rngs))
                return _TransformerLayer(**parts)

            make_layers = nnx.split_rngs(splits=num_layers)(
                nnx.vmap(make_layer, in_axes=(0,), out_axes=0, axis_size=num_layers)
            )
            # Parameters of all layers stacked on a leading [num_layers] axis.
            self.layers = make_layers(rngs)
            self.dropout_dense = None
            return

        # Norm layers.
        self.layer_norms_attn = nnx.List([make_norm(rngs) for _ in range(num_layers)])
        self.layer_norms_dense = nnx.List([make_norm(rngs) for _ in range(num_layers)])
        if self.enable_cross_attention:
            self.layer_norms_cross_attn = nnx.List([
                make_norm(rngs) for _ in range(num_layers)
            ])

        # Attention block.
        self.attention_blocks = nnx.List([make_mha(rngs) for _ in range(num_layers)])
        if self.enable_cross_attention:
            self.cross_attention_blocks = nnx.List([
                make_mha(rngs, cross=True) for _ in range(num_layers)
            ])

        # Context fusion if context is provided.
        if context_dim is not None:
            self.context_layers1 = nnx.List([
                make_context_fusion(rngs) for _ in range(num_layers)
            ])
            self.context_layers2 = nnx.List([
                make_context_fusion(rngs) for _ in range(num_layers)
            ])

        # Dense block.
        self.dense_blocks = nnx.List([make_mlp(rngs) for _ in range(num_layers)])
        if dropout_rate > 0.0:
            self.dropout_dense = nnx.List([
                make_dropout(rngs) for _ in range(num_layers)
            ])
        else:
            self.dropout_dense = None

        # Skip connection fusers.
        self.attn_skip_fuse = nnx.List([])
        self.mlp_skip_fuse = nnx.List([])
        if enable_cross_attention:
            self.cross_skip_fuse = nnx.List([])
        for num_layer in range(num_layers):
            rate = drop_path_rates[num_layer]
            self.attn_skip_fuse.append(make_fuse(attn_fuse_cls, rate, rngs))
            self.mlp_skip_fuse.append(make_fuse(mlp_fuse_cls, rate, rngs))
            if self.enable_cross_attention:
                self.cross_skip_fuse.append(make_fuse(attn_fuse_cls, rate, rngs))

    def _layer_parts(self, i: int) -> dict:
        """The sub-modules of unrolled layer ``i`` under the scanned names."""
        parts = dict(
            norm_attn=self.layer_norms_attn[i],
            norm_dense=self.layer_norms_dense[i],
            attn=self.attention_blocks[i],
            dense=self.dense_blocks[i],
            attn_fuse=self.attn_skip_fuse[i],
            mlp_fuse=self.mlp_skip_fuse[i],
        )
        if self.enable_cross_attention:
            parts.update(
                norm_cross=self.layer_norms_cross_attn[i],
                cross_attn=self.cross_attention_blocks[i],
                cross_fuse=self.cross_skip_fuse[i],
            )
        if self.context_dim is not None:
            parts.update(
                context1=self.context_layers1[i], context2=self.context_layers2[i]
            )
        if self.dropout_dense is not None:
            parts.update(dropout=self.dropout_dense[i])
        return parts

    def load_unrolled(self, other: "Transformer") -> None:
        """Copy the state of an unrolled model into this ``scan_layers`` model.

        ``other`` must have the same configuration with ``scan_layers=False``;
        its per-layer modules are stacked into ``self.layers``. All other
        state (e.g. a :class:`DiffusionTransformer`'s projections) is copied
        as is. Afterwards both models compute the same function.
        """
        if not self.scan_layers or other.scan_layers:
            raise ValueError("load_unrolled copies an unrolled model into a scanned one")
        layer_states = [
            nnx.state(_TransformerLayer(**other._layer_parts(i)))
            for i in range(other.num_layers)
        ]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *layer_states)
        nnx.update(self.layers, stacked)
        rest = {
            key: value
            for key, value in nnx.state(other).items()
            if key not in _UNROLLED_LISTS
        }
        nnx.update(self, nnx.State(rest))

    def init_cache(self, input_shape, dtype=None):
        """Initialize the self-attention KV caches of every layer for decoding."""
        kwargs = {} if dtype is None else {"dtype": dtype}
        if not self.scan_layers:
            for block in self.attention_blocks:
                block.init_cache(input_shape, **kwargs)
            return

        @nnx.vmap(in_axes=(0,), out_axes=0, axis_size=self.num_layers)
        def init(layer):
            layer.attn.init_cache(input_shape, **kwargs)

        init(self.layers)

    def __call__(
        self,
        q: Array,  # [B, T, D]
        k: Optional[Array] = None,  # [B, T', D]
        v: Optional[Array] = None,  # [B, T', D]
        context: Optional[Array] = None,  # [B, D_context]
        mask: AttentionMask | Array | None = None,
        mask_cross: AttentionMask | Array | None = None,
        bias: AttentionBias | Array | None = None,
        bias_cross: AttentionBias | Array | None = None,
        deterministic: bool | None = None,
        decode: bool = False,
        kv_len: int | Array | None = None,
        rng: jax.Array | None = None,
    ) -> Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        # Normalize masks/bias to broadcastable shapes
        if isinstance(mask, jax.Array):
            mask = normalize_attn_mask(mask)
        if isinstance(mask_cross, jax.Array):
            mask_cross = normalize_attn_mask(mask_cross)
        if isinstance(bias, jax.Array):
            bias = normalize_attn_bias(bias)
        if isinstance(bias_cross, jax.Array):
            bias_cross = normalize_attn_bias(bias_cross)

        # Flatten to (B, T, D)
        q, q_shape = flatten_to_btd(q)
        k, _ = flatten_to_btd(k) if k is not None else (None, None)
        v, _ = flatten_to_btd(v) if v is not None else (None, None)

        q = constrain(q, BATCH)

        # Ensure context has shape [B, 1, Dc] when provided
        if context is not None:
            context = context.reshape(-1, 1, context.shape[-1])

        if k is not None and not self.enable_cross_attention:
            raise ValueError("Cross attention is disabled, but k is provided.")
        if v is not None and not self.enable_cross_attention:
            raise ValueError("Cross attention is disabled, but v is provided.")

        use_context = context is not None and self.context_dim is not None
        enable_cross = self.enable_cross_attention

        def layer_fn(parts, q, keys):
            # Independent keys for each stochastic operation of the layer;
            # stored RNG streams remain responsible for randomness without rng.
            key_iter = iter([None] * 9 if keys is None else list(keys))
            # First the attention block.
            q_res = q
            q = parts["norm_attn"](q)
            if use_context:
                q = parts["context1"](q, context, rng=next(key_iter))
            q = parts["attn"](
                q,
                mask=mask,
                bias=bias,
                deterministic=deterministic,
                decode=decode,
                kv_len=kv_len,
                rng=next(key_iter),
            )
            q = constrain(q, BATCH, SEQ, EMBED)
            q = parts["attn_fuse"](
                q_res, q, context=context, deterministic=deterministic,
                rng=next(key_iter),
            )  # fmt: skip

            # Then cross attention if wanted
            if enable_cross:
                q_res = q
                q = parts["norm_cross"](q)
                q = parts["cross_attn"](
                    q,
                    k,
                    v,
                    mask=mask_cross,
                    bias=bias_cross,
                    deterministic=deterministic,
                    decode=False,
                    rng=next(key_iter),
                )
                q = constrain(q, BATCH, SEQ, EMBED)
                q = parts["cross_fuse"](
                    q_res, q, context=context, deterministic=deterministic,
                    rng=next(key_iter),
                )  # fmt: skip

            # Then the dense block and global context.
            q_res = q
            q = parts["norm_dense"](q)
            if use_context:
                q = parts["context2"](q, context, rng=next(key_iter))
            q = parts["dense"](q, rng=next(key_iter))
            q = constrain(q, BATCH, SEQ, EMBED)
            if "dropout" in parts:
                q = parts["dropout"](
                    q, deterministic=deterministic, rngs=next(key_iter)
                )
            q = parts["mlp_fuse"](
                q_res, q, context=context, deterministic=deterministic,
                rng=next(key_iter),
            )  # fmt: skip
            return q

        def layer_keys(i):
            if rng is None:
                return None
            return jax.random.split(jax.random.fold_in(rng, i), 9)

        if self.remat_policy is not None:
            layer_fn = nnx.remat(layer_fn, policy=self.remat_policy)

        if self.scan_layers:

            @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=nnx.Carry)
            def scan_fn(layer, q, i):
                return layer_fn(layer.parts(), q, layer_keys(i))

            q = scan_fn(self.layers, q, jnp.arange(self.num_layers))
        else:
            for i in range(self.num_layers):
                q = layer_fn(self._layer_parts(i), q, layer_keys(i))

        return restore_from_btd(q, q_shape)


_UNROLLED_LISTS = frozenset({
    "layer_norms_attn",
    "layer_norms_dense",
    "layer_norms_cross_attn",
    "attention_blocks",
    "cross_attention_blocks",
    "context_layers1",
    "context_layers2",
    "dense_blocks",
    "dropout_dense",
    "attn_skip_fuse",
    "mlp_skip_fuse",
    "cross_skip_fuse",
})

_LAYER_PARTS = (
    "norm_attn",
    "norm_dense",
    "attn",
    "dense",
    "attn_fuse",
    "mlp_fuse",
    "norm_cross",
    "cross_attn",
    "cross_fuse",
    "context1",
    "context2",
    "dropout",
)


class _TransformerLayer(nnx.Module):
    """One transformer layer's sub-modules (the unit stacked by ``scan_layers``)."""

    def __init__(self, **parts):
        self._names = tuple(name for name in _LAYER_PARTS if name in parts)
        for name in self._names:
            setattr(self, name, parts[name])

    def parts(self) -> dict:
        return {name: getattr(self, name) for name in self._names}


def _resolve_remat_policy(policy):
    """``None`` (no remat), a ``jax.checkpoint_policies`` name, or a policy."""
    if policy is None or callable(policy):
        return policy
    if isinstance(policy, str):
        if policy in ("full", "nothing_saveable"):
            return jax.checkpoint_policies.nothing_saveable
        resolved = getattr(jax.checkpoint_policies, policy, None)
        if resolved is not None and not policy.startswith("_"):
            return resolved
    raise ValueError(
        "remat_policy must be None, a callable checkpoint policy, 'full', or the "
        f"name of a jax.checkpoint_policies policy; got {policy!r}."
    )
