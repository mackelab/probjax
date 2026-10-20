from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, Initializer, PrecisionLike

from probjax.nn.sharding import EMBED, HIDDEN, param_metadata
from probjax.utils.typing import Array


class FeedForward(nnx.Module):
    """Two-layer MLP ``proj(activation(hidden(x)))`` on the last axis.

    The hidden layer is column-parallel and the output layer row-parallel
    under a mesh (Megatron-style), matching :class:`probjax.nn.MLP`.
    ``out_kernel_init``/``out_bias_init`` set the output layer's init, e.g.
    zeros for a block that starts as a no-op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        out_kernel_init: Initializer = nnx.initializers.lecun_normal(),
        out_bias_init: Initializer = nnx.initializers.zeros,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        preferred_element_type: Dtype | None = None,
        rngs: nnx.Rngs,
    ):
        hidden_features = hidden_features or max(in_features, out_features)
        linear_kwargs: dict[str, Any] = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.activation = activation
        self.hidden = nnx.Linear(
            in_features,
            hidden_features,
            kernel_metadata=param_metadata(EMBED, HIDDEN),
            bias_metadata=param_metadata(HIDDEN),
            **linear_kwargs,
        )
        self.proj = nnx.Linear(
            hidden_features,
            out_features,
            kernel_init=out_kernel_init,
            bias_init=out_bias_init,
            kernel_metadata=param_metadata(HIDDEN, EMBED),
            bias_metadata=param_metadata(EMBED),
            **linear_kwargs,
        )

    def __call__(self, x: Array, *, rng: jax.Array | None = None) -> Array:
        del rng
        return self.proj(self.activation(self.hidden(x)))
