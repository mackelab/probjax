import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import (
    LRU,
    MLP,
    AutoregressiveMLP,
    CouplingMLP,
    DeepSet,
    GaussianFourierEmbedding,
    MultiHeadAttention,
    Transformer,
)
from probjax.nn.utils import AffineFuse, AdditiveFuse, ConcatFuse
from probjax.nn.nets.flows import (
    AdditiveAutoregressiveFlow,
    AdditiveCouplingFlow,
    AffineAutoregressiveFlow,
    AffineCouplingFlow,
    SplineAutoregressiveFlow,
    SplineCouplingFlow,
)
from probjax.nn.nets.simple import ResNet


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
        (),
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
        (2, 1, [10, 3], nnx.relu, False),
        (1, 2, [1, 2, 1, 2], lambda x: x, False),
        (2, 2, [5, 1], jax.nn.gelu, True),
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
        (1, 1, 10),
        (1, 2, 2),
        (2, 1, 5),
        (3, 3, 4),
    ]
)
def resnet(request):
    in_dim, out_dim, hidden_units = request.param
    model = ResNet(
        in_dim,
        out_dim,
        rngs=nnx.Rngs(0),
        hidden_dim=hidden_units,
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


@pytest.fixture(
    params=[
        (1, 1, 1, 1),
        (2, 1, 1, 2),
        (1, 2, 1, 5),
        (2, 2, 2, 10),
        (1, 3, 1, 3),
        (3, 2, 3, 2),
    ]
)
def transformer(request):
    model_dim, num_heads, num_layers, attn_size = request.param
    model = Transformer(
        model_dim,
        num_heads,
        num_layers,
        attn_size,
        rngs=nnx.Rngs(0),
    )
    return model_dim, model


@pytest.fixture(
    params=[
        (1, AffineFuse),
        (2, AdditiveFuse),
        (1, ConcatFuse),
        (2, AffineFuse),
        (1, AdditiveFuse),
        (2, ConcatFuse),
    ],
    ids=[
        "AffineFuse",
        "AdditiveFuse",
        "ConcatFuse",
        "AffineFuse",
        "AdditiveFuse",
        "ConcatFuse",
    ],
)
def transformer_with_context(request):
    context_dim, fussion_method = request.param
    model_dim = 2
    num_heads = 1
    num_layers = 1
    attn_size = 2
    model = Transformer(
        model_dim,
        num_heads,
        num_layers,
        attn_size,
        context_dim=context_dim,
        context_fusion=fussion_method,
        rngs=nnx.Rngs(0),
    )
    return model_dim, context_dim, model


@pytest.fixture(
    params=[
        (1, 1, 1),
        (2, 1, 2),
        (1, 2, 1),
        (2, 2, 2),
        (1, 3, 1),
        (3, 1, 3),
    ]
)
def lru(request):
    in_dim, out_dim, hidden_dim = request.param
    model = LRU(in_dim, out_dim, hidden_dim, rngs=nnx.Rngs(0))
    return in_dim, out_dim, model


@pytest.fixture(
    params=[
        ("coupling", "additive", 2),
        ("coupling", "affine", 2),
        ("coupling", "spline", 2),
        ("autoregressive", "additive", 2),
        ("autoregressive", "affine", 2),
        ("autoregressive", "spline", 2),
        ("coupling", "additive", 3),
        ("coupling", "affine", 3),
        ("coupling", "spline", 3),
        ("autoregressive", "additive", 3),
        ("autoregressive", "affine", 3),
        ("autoregressive", "spline", 3),
    ]
)
def flow(request):
    kind, bij, input_dim = request.param
    if bij == "affine":
        if kind == "coupling":
            model = AffineCouplingFlow(input_dim, 1, rngs=nnx.Rngs(0))
        elif kind == "autoregressive":
            model = AffineAutoregressiveFlow(input_dim, 1, rngs=nnx.Rngs(0))
    elif bij == "spline":
        if kind == "coupling":
            model = SplineCouplingFlow(input_dim, 1, rngs=nnx.Rngs(0))
        elif kind == "autoregressive":
            model = SplineAutoregressiveFlow(input_dim, 1, rngs=nnx.Rngs(0))
    elif bij == "additive":
        if kind == "coupling":
            model = AdditiveCouplingFlow(input_dim, 1, rngs=nnx.Rngs(0))
        elif kind == "autoregressive":
            model = AdditiveAutoregressiveFlow(input_dim, 1, rngs=nnx.Rngs(0))
    return input_dim, model
