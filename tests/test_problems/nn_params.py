import pytest
import jax.numpy as jnp

from probjax.nn import (
    MLP,
    DeepSet,
    MultiHeadAttention,
    CouplingMLP,
    AutoregressiveMLP,
    GaussianFourierEmbedding,
)
from flax import nnx
import jax


@pytest.fixture(
    params=[
        1,
        2,
        10,
    ]
)
def seq_len(request):
    return request.param


@pytest.fixture(
    params=[
        (1,),
        (2,),
        (1, 1),
        (1, 2),
        (2, 1, 1),
    ]
)
def batch_shape(request):
    return request.param


@pytest.fixture(
    params=[
        (1, 1, [1, 2], jnp.tanh, True),
        (1, 1, [10, 3], nnx.relu, False),
        (1, 1, [1, 2, 1, 2], lambda x: x, False),
        (1, 1, [5, 1], jax.nn.gelu, True),
    ]
)
def mlp(request):
    in_dim, out_dim, hidden_units, activation, activation_final = request.param
    dims = [in_dim] + hidden_units + [out_dim]
    model = MLP(
        dims, rngs=nnx.Rngs(0), activation=activation, activate_final=activation_final
    )
    return in_dim, out_dim, model


@pytest.fixture(
    params=[
        (1, 1, 1, [1, 2]),
        (1, 10, 1, [10, 3]),
        (1, 1, 2, [1, 2, 1, 2]),
        (2, 1, 1, [5, 1]),
    ]
)
def deepset(request):
    in_dim, latent_dim, out_dim, hidden_units = request.param
    phi = MLP(
        [in_dim] + hidden_units + [latent_dim],
        rngs=nnx.Rngs(0),
        activation=jax.nn.relu,
        activate_final=True,
    )
    rho = MLP(
        [latent_dim] + hidden_units + [out_dim],
        rngs=nnx.Rngs(0),
        activation=jax.nn.relu,
        activate_final=True,
    )
    model = DeepSet(phi, rho)
    return in_dim, out_dim, model


@pytest.fixture(
    params=[
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (1, 3),
        (3, 1),
    ]
)
def multi_head_attention(request):
    num_heads, in_features = request.param
    out_dim = in_features * num_heads
    model = MultiHeadAttention(
        num_heads, in_features, out_dim, out_dim, rngs=nnx.Rngs(0)
    )
    return in_features, out_dim, model


def affine_bijector(params, x):
    return params + x


def scale_bijector(params, x):
    return jnp.exp(params) * x


@pytest.fixture(
    params=[
        (2, affine_bijector, [1, 2]),
        (4, affine_bijector, [10, 3]),
        (6, scale_bijector, [1, 2, 1, 2]),
        (8, scale_bijector, [5, 1]),
    ]
)
def coupling_mlp(request):
    in_dim, bijector, hidden_units = request.param
    model = CouplingMLP(
        in_dim // 2,
        in_dim - in_dim // 2,
        bijector,
        hidden_dims=hidden_units,
        rngs=nnx.Rngs(0),
    )
    return in_dim, in_dim, model


@pytest.fixture(
    params=[
        (1, affine_bijector, [1, 2]),
        (2, affine_bijector, [10, 3]),
        (3, scale_bijector, [1, 2, 1, 2]),
        (4, scale_bijector, [5, 1]),
    ]
)
def autoregressive_mlp(request):
    in_dim, bijector, hidden_units = request.param
    model = AutoregressiveMLP(
        in_dim,
        1,
        bijector,
        hidden_dims=hidden_units,
        rngs=nnx.Rngs(0),
    )
    return in_dim, in_dim, model


@pytest.fixture(
    params=[
        (1, 1, True),
        (2, 1, False),
        (1, 2, True),
        (2, 2, False),
        (1, 3, True),
        (3, 1, False),
    ]
)
def gaussian_fourier_embedding(request):
    in_dim, out_dim, learnable = request.param
    model = GaussianFourierEmbedding(
        in_dim, out_dim, rngs=nnx.Rngs(0), learnable=learnable
    )
    return in_dim, out_dim, model
