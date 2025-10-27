import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax.nnx import MultiHeadAttention as FlaxMultiHeadAttention
from flax.nnx import combine_masks, rnglib
from flax.nnx import dot_product_attention as flax_dot_product_attention
from flax.nnx.module import first_from
from jax import lax

from probjax.nn.pallas_kernels.attention import BlockSizes, mha
from probjax.nn.pallas_kernels.attention_mask_bias import (
    AttentionBias,
    AttentionMask,
    QKVLengthMask,
)
from probjax.nn.utils import pad_to_power_of_2
from probjax.utils.typing import Array, ArrayLike


class MultiHeadAttention(FlaxMultiHeadAttention):
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: AttentionMask | ArrayLike | None = None,
        bias: AttentionBias | ArrayLike | None = None,
        deterministic: bool | None = True,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        sow_weights: bool = False,
        decode: bool | None = False,
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

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
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
            ) = self.cached_key.value.shape
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
    backward_pass_impl: str = "triton_fused",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
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
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
    )

    output = output[:, :l_q, :h, :n]

    return output
