from functools import partial
from typing import Callable, Optional, Tuple

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
from probjax.nn.layers.fuse import AffineFuse
from probjax.nn.nets.simple import MLP
from probjax.nn.utils import filter_precision_kwargs, get_active_precision_kwargs
from probjax.utils.typing import DTypeLike, ModuleLikeType, PrecisionLike


# ---------------- Helper utilities ----------------
def _flatten_to_btd(x: Array | None) -> Tuple[Array | None, tuple | None]:
    """Flatten leading batch dims to (B, T, D) for attention blocks.

    Returns the flattened array and the original shape to restore later.
    If x is None, returns (None, None).
    """
    if x is None:
        return None, None
    x = jnp.asarray(x)
    orig = x.shape
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    return x, orig


def _restore_from_btd(x: Array, orig_shape: tuple | None) -> Array:
    """Restore tensor from (B, T, D) back to original leading batch dims."""
    if orig_shape is None:
        return x
    return x.reshape(orig_shape)


def _normalize_attn_mask(mask: Array | None) -> Array | None:
    """Normalize attention mask shapes to [B, 1, T, T] or broadcastable.

    Accepts 2D [T, T], 3D [B, T, T], or 4D [B, 1, T, T]. Returns a shape
    that broadcasts with [B, H, T, T]. If mask is None, returns None.
    """
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.ndim == 2:
        return mask[None, None, :, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Mask must have ndim 2, 3, or 4; got {mask.ndim}.")


def _normalize_attn_bias(bias: Array | None) -> Array | None:
    """Normalize attention bias shapes similar to masks.

    Accepts 2D [T, T], 3D [B, T, T], or 4D [B, 1, T, T]. Returns a shape
    broadcastable with attention logits [B, H, T, T]. If bias is None, returns None.
    Other shapes (e.g., custom bias types) are passed through by the caller.
    """
    if bias is None:
        return None
    bias = jnp.asarray(bias)
    if bias.ndim == 2:
        return bias[None, None, :, :]
    if bias.ndim == 3:
        return bias[:, None, :, :]
    if bias.ndim == 4:
        return bias
    raise ValueError(f"Bias must have ndim 2, 3, or 4; got {bias.ndim}.")


class Transformer(nnx.Module):
    """A transformer stack."""

    model_dim: int  # Dimensionality of the embedding vectors.
    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float | None  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        *,
        enable_cross_attention: bool = False,
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
        context_dim: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable = jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        initializer: Optional[nnx.Initializer] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        norm_cls: ModuleLikeType = nnx.LayerNorm,
        context_fusion_cls: ModuleLikeType = AffineFuse,
        attention_fn: Optional[Callable] = None,
        cross_attention_fn: Optional[Callable] = None,
        mlp_cls: ModuleLikeType = MLP,
        mha_cls: ModuleLikeType = MultiHeadAttention,
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
            dropout_rate (Optional[float], optional): Dropout rate. If None, no dropout
                is applied. Defaults to None.
            widening_factor (int, optional): Factor by which to increase the dimension
                in the MLP. Defaults to 4.
            num_hidden_layers (int, optional): Number of hidden layers in the MLP block.
                Defaults to 1.
            act (Callable, optional): Activation function. Defaults to jax.nn.gelu.
            skip_connection_attn (bool, optional): Whether to use skip connections in
                attention blocks. Defaults to True.
            skip_connection_mlp (bool, optional): Whether to use skip connections in
                MLP blocks. Defaults to True.
            initializer (Optional[nnx.initializers.Initializer], optional): Weight
                initializer. If None, uses truncated normal with variance scaling.
                Defaults to None.
            attention_fn (Optional[Callable], optional): Custom attention function.
                If None, uses dot product attention. Defaults to None.
        """
        super().__init__()
        self.model_dim = model_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.initializer = (
            nnx.initializers.variance_scaling(
                2 / self.num_layers, 'fan_in', 'truncated_normal'
            )
            if initializer is None
            else initializer
        )
        self.act = act
        self.enable_cross_attention = enable_cross_attention
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp

        # Precision and dtype settings.
        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # Layer norms for the attention and dense blocks.
        self.layer_norms_attn = nnx.List([
            norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
        ])
        self.layer_norms_dense = nnx.List([
            norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
        ])

        if self.enable_cross_attention:
            self.layer_norms_cross_attn = nnx.List([
                norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
            ])

        self.out_layer_norm = norm_cls(model_dim, rngs=rngs)

        # Attention block.
        attention_fn = (
            attention_fn if attention_fn is not None else dot_product_attention
        )
        self.attention_blocks = nnx.List([
            mha_cls(
                num_heads,
                model_dim,
                attn_size * num_heads,
                model_dim,
                rngs=rngs,
                kernel_init=self.initializer,
                dropout_rate=dropout_rate if dropout_rate is not None else 0.0,
                attention_fn=attention_fn,
                normalize_qk=normalize_qk_attn,
                **filter_precision_kwargs(mha_cls, **precision_kwargs),
            )
            for _ in range(num_layers)
        ])

        if self.enable_cross_attention:
            cross_attention_fn = (
                cross_attention_fn
                if cross_attention_fn is not None
                else nnx.dot_product_attention
            )
            self.cross_attention_blocks = nnx.List([
                mha_cls(
                    num_heads,
                    model_dim,
                    attn_size * num_heads,
                    model_dim,
                    rngs=rngs,
                    kernel_init=self.initializer,
                    dropout_rate=dropout_rate if dropout_rate is not None else 0.0,
                    attention_fn=cross_attention_fn,
                    normalize_qk=normalize_qk_cross_attn,
                    **filter_precision_kwargs(mha_cls, **precision_kwargs),
                )
                for _ in range(num_layers)
            ])

        # Context fusion if context is provided.
        if context_dim is not None:
            self.context_layers = nnx.List([
                context_fusion_cls(model_dim, context_dim, rngs=rngs)
                for _ in range(num_layers)
            ])

        # Dense block.
        dims = (
            [model_dim]
            + [widening_factor * model_dim] * num_hidden_layers
            + [model_dim]
        )
        linear = partial(nnx.Linear, kernel_init=self.initializer)
        self.dense_blocks = nnx.List([
            mlp_cls(
                dims,
                rngs=rngs,
                linear_cls=linear,
                activation=act,
                activate_final=True,
                **filter_precision_kwargs(mlp_cls, **precision_kwargs),
            )
            for _ in range(num_layers)
        ])

        if dropout_rate is not None:
            self.dropout_dense = nnx.List([
                nnx.Dropout(rate=dropout_rate, rngs=rngs) for _ in range(num_layers)
            ])
        else:
            self.dropout_dense = None

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
    ) -> Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""
        q = jnp.asarray(q)
        k = None if k is None else jnp.asarray(k)
        v = None if v is None else jnp.asarray(v)
        context = None if context is None else jnp.asarray(context)

        # Normalize masks/bias to broadcastable shapes
        if isinstance(mask, jax.Array):
            mask = _normalize_attn_mask(mask)
        if isinstance(mask_cross, jax.Array):
            mask_cross = _normalize_attn_mask(mask_cross)
        if isinstance(bias, jax.Array):
            bias = _normalize_attn_bias(bias)
        if isinstance(bias_cross, jax.Array):
            bias_cross = _normalize_attn_bias(bias_cross)

        # Flatten to (B, T, D)
        q, q_shape = _flatten_to_btd(q)
        k, _ = _flatten_to_btd(k)
        v, _ = _flatten_to_btd(v)

        # Ensure context has shape [B, 1, Dc] when provided
        if context is not None:
            context = context.reshape(-1, 1, context.shape[-1])

        if k is not None and not self.enable_cross_attention:
            raise ValueError("Cross attention is disabled, but k is provided.")
        if v is not None and not self.enable_cross_attention:
            raise ValueError("Cross attention is disabled, but v is provided.")

        for i in range(self.num_layers):
            # First the attention block.
            q = self.layer_norms_attn[i](q)
            h_attn = self.attention_blocks[i](
                q, mask=mask, bias=bias, deterministic=deterministic, decode=decode
            )
            q = q + h_attn if self.skip_connection_attn else h_attn

            # Then cross attention if wanted
            if self.enable_cross_attention:
                q = self.layer_norms_cross_attn[i](q)
                h_cross_attn = self.cross_attention_blocks[i](
                    q,
                    k,
                    v,
                    mask=mask_cross,
                    bias=bias_cross,
                    deterministic=deterministic,
                    decode=False,
                )
                q = q + h_cross_attn

            # Then the dense block and global context.
            q = self.layer_norms_dense[i](q)
            if context is not None and self.context_dim is not None:
                h_context = self.context_layers[i](q, context)
            else:
                h_context = q

            h_dense = self.dense_blocks[i](h_context)
            if self.dropout_dense is not None:
                h_dense = self.dropout_dense[i](h_dense, deterministic=deterministic)
            q = q + h_dense if self.skip_connection_mlp else h_dense

        q = self.out_layer_norm(q)

        return _restore_from_btd(q, q_shape)
