from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from

from probjax.utils.typing import Array, ArrayLike

__all__ = ["DropPath"]


class DropPath(Module):
    """Stochastic depth (DropPath) for residual branches.

    Matching nnx.Dropout conventions:
    - Accepts optional rngs / rng_collection and handles forking.
    - Resolves `deterministic` and `rngs` at call time with `first_from`.
    - Supports overriding broadcast dims and scaling in the call.
    """

    def __init__(
        self,
        drop_rate: float,
        *,
        broadcast_dims: Sequence[int] | None = None,
        scale_by_keep: bool = True,
        deterministic: bool = False,
        rng_collection: str = "droppath",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ):
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}.")

        self.drop_rate = float(drop_rate)
        self.broadcast_dims = None if broadcast_dims is None else tuple(broadcast_dims)
        self.scale_by_keep = scale_by_keep
        self.deterministic = deterministic
        self.rng_collection = rng_collection

        if isinstance(rngs, rnglib.Rngs):
            self.rngs = rngs[self.rng_collection].fork()
        elif isinstance(rngs, rnglib.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = nnx.data(None)
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}."
            )

    def __call__(
        self,
        inputs: ArrayLike,
        *,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
        scale_by_keep: bool | None = None,
    ) -> Array:
        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropPath
                as either a __call__ argument or class attribute""",
        )

        x = jnp.asarray(inputs)
        if not jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(jnp.float32)

        if (self.drop_rate == 0.0) or deterministic:
            return x

        # Drop entirely when drop_rate == 1.0 to avoid NaNs.
        if self.drop_rate == 1.0:
            return jnp.zeros_like(x)

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg="""`deterministic` is False, but no `rngs` argument was provided to DropPath
                as either a __call__ argument or class attribute.""",
        )

        if isinstance(rngs, rnglib.Rngs):
            key = rngs[self.rng_collection]()
        elif isinstance(rngs, rnglib.RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or jax.Array, but got {type(rngs)}."
            )

        keep_prob = 1.0 - self.drop_rate
        if self.broadcast_dims is None:
            if x.ndim == 0:
                mask_shape = ()
            else:
                mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            mask_shape = tuple(
                1 if dim in self.broadcast_dims else x.shape[dim]
                for dim in range(x.ndim)
            )

        keep_mask = jax.random.bernoulli(key, p=keep_prob, shape=mask_shape)
        keep_mask = keep_mask.astype(x.dtype)

        # Match Dropout behavior by allowing runtime override.
        scale = first_from(
            scale_by_keep,
            self.scale_by_keep,
            error_msg="Internal error resolving scale_by_keep flag for DropPath.",
        )
        if scale and keep_prob > 0.0:
            x = x / keep_prob

        return x * keep_mask
