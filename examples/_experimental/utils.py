from typing import Sequence

import distrax
import haiku as hk
import numpy as np
from jax import Array
from jax import numpy as jnp


class Context(hk.Module):
    def __init__(self, context, name: str | None = None):
        super().__init__(name)
        self.context = context

    def __call__(self, x):
        x_new = jnp.concatenate([x, self.context], axis=-1)
        return x_new


def make_conditioner(
    context,
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""

    return hk.Sequential(
        [
            Context(context),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(params, range_min=-2, range_max=2.0)


def make_flow(
    context,
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins,
) -> distrax.Transformed:
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                context, event_shape, hidden_sizes, num_bijector_params
            ),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        distrax.Uniform(
            low=-2.0 * jnp.ones(event_shape), high=2.0 * jnp.ones(event_shape)
        ),
        reinterpreted_batch_ndims=len(event_shape),
    )

    return distrax.Transformed(base_distribution, flow)


def rbf_kernel(X, Y, gamma):
    """
    Computes the RBF kernel between two matrices X and Y.
    """
    XX = jnp.sum(X**2, axis=-1, keepdims=True)
    YY = jnp.sum(Y**2, axis=-1, keepdims=True)
    XY = jnp.dot(X, Y.T)
    K = 10 * jnp.exp(-gamma * (XX - 2 * XY + YY.T))
    return K


def mmd(X, Y, gamma):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y.
    """
    K_XX = rbf_kernel(X, X, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    mmd = jnp.mean(K_XX) - 2 * jnp.mean(K_XY) + jnp.mean(K_YY)
    return mmd
