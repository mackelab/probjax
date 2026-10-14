"""Invertibility, density accounting, and training of composable flow mixers."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.core import inverse_and_logabsdet
from probjax.nn import (
    AutoregressiveNFlowConfig,
    CirculantDiagonal,
    CirculantDiagonalMixingConfig,
    CompositeMixingConfig,
    CouplingNFlowConfig,
    ElementwiseNFlowConfig,
    NFlow,
    NoMixingConfig,
    PermuteMixingConfig,
)


@pytest.mark.parametrize("features", [1, 2, 5, 6])
@pytest.mark.parametrize("num_diagonals", [1, 2, 3])
def test_circulant_diagonal_roundtrip_and_dense_logdet(features, num_diagonals):
    layer = CirculantDiagonal(
        features, num_diagonals=num_diagonals, init_std=0.2, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.key(1), (features,))
    y = layer(x)
    recovered, inverse_logdet = jax.jit(inverse_and_logabsdet(layer))(y)
    dense_jacobian = jax.jacfwd(layer)(x)
    _, forward_logdet = jnp.linalg.slogdet(dense_jacobian)

    np.testing.assert_allclose(recovered, x, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(inverse_logdet, -forward_logdet, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("features", [5, 6])
def test_batch_axes_match_vmap_and_logdet_counts_events(features):
    layer = CirculantDiagonal(features, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 3, features))
    expected = jax.vmap(jax.vmap(layer))(x)
    np.testing.assert_allclose(jax.jit(layer)(x), expected, atol=1e-6)

    recovered, logdet = inverse_and_logabsdet(layer)(layer(x))
    _, event_logdet = inverse_and_logabsdet(layer)(layer(x[0, 0]))
    np.testing.assert_allclose(recovered, x, atol=2e-6)
    np.testing.assert_allclose(logdet, 6 * event_logdet, atol=2e-6)


def test_identity_initialization():
    layer = CirculantDiagonal(5, num_diagonals=3, init_std=0, rngs=nnx.Rngs(0))
    x = jnp.arange(5, dtype=jnp.float32)
    recovered, logdet = inverse_and_logabsdet(layer)(x)
    np.testing.assert_allclose(layer(x), x, atol=1e-6)
    np.testing.assert_allclose(recovered, x, atol=1e-6)
    np.testing.assert_allclose(logdet, 0, atol=1e-6)


@pytest.mark.parametrize("features", [5, 6])
def test_float64_is_preserved(features):
    with jax.enable_x64(True):
        layer = CirculantDiagonal(features, param_dtype=jnp.float64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(1), (features,), dtype=jnp.float64)
        y = layer(x)
        recovered, logdet = inverse_and_logabsdet(layer)(y)
        assert y.dtype == recovered.dtype == logdet.dtype == jnp.float64
        np.testing.assert_allclose(recovered, x, atol=1e-12)


@pytest.mark.parametrize("permutation_first", [True, False])
def test_composition_order_and_inverse(permutation_first):
    permute = PermuteMixingConfig(permutation=(2, 0, 3, 1))
    learned = CirculantDiagonalMixingConfig(init_std=0.2)
    configs = (permute, learned) if permutation_first else (learned, permute)
    layer = CompositeMixingConfig(configs).build(4, rngs=nnx.Rngs(0))
    x = jnp.array([0.3, -1.2, 2.0, 0.7])
    first, second = layer.layers
    expected = second(first(x))
    np.testing.assert_allclose(layer(x), expected, atol=1e-6)
    assert not jnp.allclose(layer(x), first(second(x)))

    recovered, logdet = inverse_and_logabsdet(layer)(layer(x))
    _, dense_logdet = jnp.linalg.slogdet(jax.jacfwd(layer)(x))
    np.testing.assert_allclose(recovered, x, atol=2e-6)
    np.testing.assert_allclose(logdet, -dense_logdet, atol=2e-6)
    permutation_layer = first if permutation_first else second
    assert not jax.tree.leaves(nnx.state(permutation_layer, nnx.Param))


def test_composition_skips_noop_configs():
    assert CompositeMixingConfig(()).build(4, rngs=nnx.Rngs(0)) is None
    assert CompositeMixingConfig((NoMixingConfig(),)).build(4, rngs=nnx.Rngs(0)) is None
    config = CompositeMixingConfig((
        NoMixingConfig(),
        PermuteMixingConfig(permutation=(1, 0)),
    ))
    layer = config.build(2, rngs=nnx.Rngs(0))
    np.testing.assert_array_equal(layer(jnp.array([1.0, 2.0])), jnp.array([2.0, 1.0]))


@pytest.mark.parametrize("permutation", [(0, 0, 2), (0, 1), (0, 1, 3), (0.0, 1.0, 2.0)])
def test_explicit_permutation_must_be_bijective(permutation):
    with pytest.raises(ValueError, match="permutation"):
        PermuteMixingConfig(permutation=permutation).build(3, rngs=nnx.Rngs(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_out_features": 0},
        {"num_diagonals": 0},
        {"init_std": -1},
        {"init_std": float("nan")},
        {"max_log_scale": 0},
        {"max_log_scale": float("inf")},
    ],
)
def test_invalid_mixer_configuration(kwargs):
    options = {"in_out_features": 4, **kwargs}
    with pytest.raises(ValueError):
        CirculantDiagonal(**options, rngs=nnx.Rngs(0))


def test_input_shape_and_dtype_validation():
    layer = CirculantDiagonal(4, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="feature axis"):
        layer(jnp.ones(3))
    with pytest.raises(TypeError, match="real floating"):
        layer(jnp.ones(4, dtype=jnp.complex64))
    with pytest.raises(TypeError, match="real floating"):
        layer(jnp.arange(4))
    with pytest.raises(TypeError, match="MixingConfigProtocol"):
        CompositeMixingConfig((object(),)).build(4, rngs=nnx.Rngs(0))


@pytest.mark.parametrize(
    "config_cls",
    [CouplingNFlowConfig, AutoregressiveNFlowConfig, ElementwiseNFlowConfig],
)
def test_composed_mixing_in_flows_and_trainable_density(config_cls):
    mixing = CompositeMixingConfig((
        PermuteMixingConfig(),
        CirculantDiagonalMixingConfig(),
    ))
    flow = NFlow(
        config_cls(input_dim=4, num_transforms=3, mixing=mixing),
        nnx.Rngs(0),
        standardize=False,
    )
    x = jax.random.normal(jax.random.key(1), (8, 4))
    loss, grads = nnx.value_and_grad(lambda model: model.loss(None, x))(flow)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
    first_mixer = flow.transformation.layers[1].layers[1]
    second_mixer = flow.transformation.layers[3].layers[1]
    assert first_mixer.log_scales is not second_mixer.log_scales
    mixing_grads = grads["transformation"]["layers"][1]["layers"][1]
    assert jnp.linalg.norm(mixing_grads["log_scales"][...]) > 0

    optimizer = nnx.Optimizer(flow, optax.adam(1e-3), wrt=nnx.Param)
    optimizer.update(flow, grads)
    distribution = flow.as_dist()
    samples = distribution.sample(jax.random.key(2), (8,))
    assert samples.shape == (8, 4)
    assert jnp.all(jnp.isfinite(distribution.logpdf(samples)))


def test_spectral_phases_receive_gradients():
    layer = CirculantDiagonal(5, rngs=nnx.Rngs(0))
    x = jnp.array([0.5, -1.0, 3.0, 2.0, -0.7])
    weights = jnp.array([1.0, 0.2, -0.5, 0.7, 2.0])
    grads = nnx.grad(lambda model: jnp.dot(model(x), weights))(layer)
    assert jnp.linalg.norm(grads["phases"][...]) > 0
    assert jnp.all(jnp.isfinite(grads["phases"][...]))
