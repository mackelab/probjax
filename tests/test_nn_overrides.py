"""Subclass defaults must reach actual constructors and permit explicit overrides."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn import MLP, ResNet, Transformer, UNet
from probjax.nn.layers.encoding import GaussianFourierEmbedding
from probjax.nn.nets.ssm import SSMModel
from probjax.nn.nets.time import TimeMLP


class CustomLinear(nnx.Linear):
    pass


def test_mlp_and_resnet_subclass_defaults_and_explicit_none():
    class CustomMLP(MLP):
        linear_cls = CustomLinear
        norm_cls = nnx.LayerNorm

    class CustomResNet(ResNet):
        linear_cls = CustomLinear

    model = CustomMLP([2, 3, 1], rngs=nnx.Rngs(0))
    assert all(isinstance(layer, CustomLinear) for layer in model.layers)
    explicit = CustomMLP(
        [2, 3, 1], linear_cls=nnx.Linear, norm_cls=None, rngs=nnx.Rngs(0)
    )
    assert all(type(layer) is nnx.Linear for layer in explicit.layers)
    baseline = MLP([2, 3, 1], rngs=nnx.Rngs(0))
    np.testing.assert_array_equal(explicit(jnp.ones(2)), baseline(jnp.ones(2)))
    resnet = CustomResNet(2, 1, rngs=nnx.Rngs(0))
    assert isinstance(resnet.in_layer, CustomLinear)
    assert isinstance(resnet.out_layer, CustomLinear)


def test_ssm_projection_and_transformer_dense_overrides():
    class CustomSSM(SSMModel):
        linear_cls = CustomLinear

    class CustomTransformer(Transformer):
        linear_cls = CustomLinear

    ssm = CustomSSM(2, 4, 2, 1, rngs=nnx.Rngs(0))
    assert isinstance(ssm.in_layer, CustomLinear)
    assert isinstance(ssm.out_layer, CustomLinear)
    assert isinstance(ssm.block_out1[0], CustomLinear)
    assert ssm(jnp.ones((1, 3, 2))).shape == (1, 3, 2)
    transformer = CustomTransformer(4, 2, 1, 2, rngs=nnx.Rngs(0))
    assert isinstance(transformer.dense_blocks[0].layers[0], CustomLinear)
    assert transformer(jnp.ones((1, 3, 4))).shape == (1, 3, 4)


def test_time_mlp_submodules_are_replaceable():
    class Fourier(GaussianFourierEmbedding):
        pass

    class Body(ResNet):
        pass

    class TimeNet(TimeMLP):
        fourier_cls = Fourier
        body_cls = Body

    model = TimeNet(
        2, hidden_dim=4, depth=1, fourier_dim=4, time_embed_dim=4, rngs=nnx.Rngs(0)
    )
    assert isinstance(model.time_fourier, Fourier)
    assert isinstance(model.body, Body)
    assert model(0.5, jnp.ones((3, 2))).shape == (3, 2)


def test_unet_projection_subclass_override():
    class Conv(nnx.Conv):
        pass

    class Net(UNet):
        conv_cls = Conv

    model = Net(2, [32, 32], kernel_size=2, rngs=nnx.Rngs(0))
    custom = [module for _, module in nnx.iter_graph(model) if isinstance(module, Conv)]
    assert len(custom) >= 2
    assert model(jnp.ones((1, 8, 2))).shape == (1, 8, 2)


def test_deepset_forwards_rho_positional_arguments():
    from probjax.nn import DeepSet

    model = DeepSet(lambda x: x, lambda x, scale: x * scale, rngs=nnx.Rngs(0))
    np.testing.assert_array_equal(model(jnp.ones((3, 2)), rho_args=(2.0,)), [6.0, 6.0])
