import haiku as hk
import jax
import jax.numpy as jnp

from haiku.nets import MLP
from jaxtyping import Array, PyTree

from typing import Callable, Any


def autoregressive_mask_getter(
    d: int, fully_connected_dimensions=0, last_layer: bool = False
):
    """Custom getter for autoregressive masks.

    Args:
        d (int): Dimension of the input
        last_layer (bool, optional): Whether this is the last layer. Defaults to False.

    Returns:
        Callable: Getter function
    """

    def getter(next_getter: Callable, value: Array, context: PyTree):
        shape = context.original_shape
        name = context.full_name
        module = context.module
        if isinstance(module, hk.Linear):
            if "/w" in name:
                input_dim = shape[0]
                if fully_connected_dimensions == 0:
                    x1 = jnp.arange(input_dim).reshape(-1, 1) % d + 1
                    x2 = jnp.arange(shape[-1]).reshape(1, -1) % d + 1
                else:
                    x1 = (
                        jnp.arange(input_dim - fully_connected_dimensions).reshape(
                            -1, 1
                        )
                        % d
                        + 1
                    )
                    x1 = jnp.concatenate(
                        [x1, jnp.zeros((fully_connected_dimensions, 1))], axis=0
                    )
                    x2 = jnp.arange(shape[-1]).reshape(1, -1) % d + 1

                if not last_layer:
                    mask = x2 >= x1
                else:
                    mask = x2 > x1

                return next_getter(value * mask)
            else:
                return next_getter(value)
        else:
            raise NotImplementedError("Only Linear layers are supported, currently")

    return getter


class AutoregressiveMLP(MLP):
    def __init__(self, *args, fully_connected_input=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fully_connected_input = fully_connected_input

    def __call__(self, inputs: Array, rng=None) -> Array:
        num_layers = len(self.layers)
        in_dim = inputs.shape[-1]
        out = inputs
        for i, layer in enumerate(self.layers):
            # Here the masks are always recomputed, but we could also cache them (memory vs. compute tradeoff)
            if i == 0:
                fully_connected_input = self.fully_connected_input
            else:
                fully_connected_input = 0

            # Masking the weights to be autoregressive (at selected dimensions)!
            with hk.custom_getter(
                autoregressive_mask_getter(
                    in_dim,
                    fully_connected_dimensions=fully_connected_input,
                    last_layer=i == num_layers - 1,
                )
            ):
                out = layer(out)
                if i < (num_layers - 1) or self.activate_final:
                    out = self.activation(out)
        return out


class InvertibleTransformer(hk.Module):
    def __init__(
        self, conditionor, transformer, transformer_inv=None, name: str | None = None
    ):
        super().__init__(name)
        self.conditionor = conditionor
        self.transformer = transformer
        self.transformer_inv = transformer_inv

    def __call__(self, x):
        params = self.conditionor(x)
        return self.transformer(params, x)

    def inverse(self, y):
        x = jnp.ones_like(y)
        for i in range(x.shape[-1]):
            params = self.conditionor(x)
            # Maybe just always relay on vector information
            x = x.at[..., i].set(self.transformer_inv(params, y)[..., i])
        return x


def construct_affine_autoregressive(d, hidden_units=[50, 50, 50]):
    ann = AutoregressiveMLP(hidden_units + [d * 2])

    def f(params, x):
        scale, bias = params.split(2, axis=-1)
        scale = jax.nn.softplus(scale)
        return scale * x + bias

    def f_inv(params, x):
        scale, bias = params.split(2, axis=-1)
        scale = jax.nn.softplus(scale)
        return (x - bias) / scale

    return InvertibleTransformer(ann, f, f_inv)


@hk.without_apply_rng
@hk.transform
def forward(x):
    nn = construct_affine_autoregressive(5)
    return nn(x)


@hk.without_apply_rng
@hk.transform
def inverse(x):
    nn = construct_affine_autoregressive(5)
    return nn.inverse(x)
