"""Low-rank linear projection: ``in -> rank -> out`` (a usable factored Linear).

This is the same factorization as a LoRA layer (``x @ A @ B``) with two changes
that make it a projection rather than an adapter:

* it stores **only** the factors -- there is no full-rank base matrix -- so the
  parameter count is ``in*rank + rank*out`` instead of ``in*out``;
* both factors start at the usual ``nnx.Linear`` scale, so the layer is not the
  zero function at initialisation the way ``nnx.LoRA`` (zero ``lora_b``) is.

The optional norm between the factors is the DeepSeek query/encoder form
(``wq_a -> q_norm -> wq_b``).  The constructor mirrors ``nnx.Linear`` so the
module can stand in wherever a projection is expected.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.typing import Initializer

from probjax.utils.typing import Array, DTypeLike, PrecisionLike


def _default_kernel_init() -> Initializer:
    # Same scale as nnx.Linear; there is no activation between the factors.
    return nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")


class LowRankLinear(nnx.Module):
    """``x -> down -> [norm] -> up``, a rank-``rank`` map from ``in`` to ``out``.

    Args:
        in_features: input width.
        out_features: output width.
        rank: the bottleneck width (``rank <= min(in_features, out_features)``,
            otherwise the factorization is not low-rank and this raises).
        use_bias: bias on the output factor only (the down factor is unbiased,
            matching DeepSeek's query path).
        norm_cls: optional norm applied to the bottleneck, called as
            ``norm_cls(rank, dtype=..., param_dtype=..., rngs=...)``; ``None``
            for no norm.
        kernel_init: initializer for both factors (default: ``nnx.Linear`` scale).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        use_bias: bool = False,
        norm_cls: Any = None,
        kernel_init: Initializer | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        precision: PrecisionLike | None = None,
        rngs: rnglib.Rngs,
    ):
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank={rank} is not a bottleneck: it exceeds "
                f"min(in_features, out_features)={min(in_features, out_features)}"
            )
        if kernel_init is None:
            kernel_init = _default_kernel_init()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.down = nnx.Linear(
            in_features,
            rank,
            use_bias=False,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.norm = (
            None
            if norm_cls is None
            else norm_cls(rank, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        )
        self.up = nnx.Linear(
            rank,
            out_features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array, out_sharding: Any = None) -> Array:
        h = self.down(x, out_sharding)
        if self.norm is not None:
            h = self.norm(h)
        return self.up(h, out_sharding)


__all__ = ["LowRankLinear"]
