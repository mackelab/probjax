import jax
import jax.numpy as jnp
from flax import nnx

from probjax.utils.typing import Array, ArrayLike

__all__ = ["DropPath"]


class DropPath(nnx.Module):
    """Stochastic depth (DropPath) for residual branches.

    If `broadcast_dims=None`: per-example mask with shape (B, 1, 1, ...).
    If `broadcast_dims=(1,)`: share mask across sequence length, etc.
    """

    def __init__(
        self,
        drop_rate: float = 0.0,
        *,
        broadcast_dims: tuple[int, ...] | None = None,
        rngs: nnx.Rngs,  # injected by NNX
    ):
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}.")
        self.drop_rate = float(drop_rate)
        self.broadcast_dims = broadcast_dims
        self.rngs = rngs  # make sure this is stored

    def __call__(
        self,
        x: ArrayLike,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
        scale_by_keep: bool = True,
    ) -> Array:
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(jnp.float32)

        if self.drop_rate == 0.0 or deterministic:
            return x

        keep_prob = 1.0 - self.drop_rate

        if self.broadcast_dims is None:
            mask_shape = () if x.ndim == 0 else (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            mask_shape = tuple(1 if i in self.broadcast_dims else x.shape[i]
                               for i in range(x.ndim))

        rngs = rngs or self.rngs
        key = rngs.dropout()
        keep_mask = jax.random.bernoulli(key, keep_prob, mask_shape).astype(x.dtype)

        if scale_by_keep and keep_prob > 0:
            x = x / keep_prob
        return x * keep_mask
