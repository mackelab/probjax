import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from probjax.nn.sharding import ShardingCfg


class MaskedLinear(nnx.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: ArrayLike,
        *,
        sharding_cfg: ShardingCfg | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)
        super().__init__(in_features, out_features, rngs=rngs, **kwargs)
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        if mask.shape != (in_features, out_features):
            raise ValueError("Mask shape must be (in_features, out_features)")
        self.mask = nnx.Variable(mask)

    def __call__(self, inputs, rng: jax.Array | None = None):
        del rng
        kernel = jnp.where(self.mask[...], self.kernel[...], 0.0)
        bias = self.bias[...] if self.bias else None

        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        # We use dot_general_kwargs for BC compatibility with
        # user custom self.dot_general method which may not have
        # preferred_element_type argument to avoid breaking
        # existing code
        dot_general_kwargs = {}
        if self.preferred_element_type is not None:
            dot_general_kwargs["preferred_element_type"] = self.preferred_element_type
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
            **dot_general_kwargs,
        )
        assert self.use_bias == (bias is not None)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
