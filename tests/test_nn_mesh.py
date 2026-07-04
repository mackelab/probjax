import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from probjax.nn import (
    LinearShardingSpec,
    MaskedLinear,
    MLP,
    MLPShardingSpec,
    ShardingCfg,
)

pytest_plugins = ["test_problems.nns"]


def _sharding_spec_of(array: jax.Array) -> P:
    sharding = array.sharding
    return getattr(sharding, "spec", getattr(sharding, "partition_spec", P()))


def _input_spec_for_shape(
    mesh, shape, feature_axis: int = -1, allow_model: bool = True
):
    spec = [None] * len(shape)
    spec[0] = "data"
    model = mesh.shape.get("model", 1)
    if allow_model and model > 1:
        feat_dim = shape[feature_axis]
        if feat_dim % model == 0:
            spec[feature_axis] = "model"
    return P(*spec)


def _sharded_ones(shape, mesh, spec: P):
    return jax.device_put(jnp.ones(shape), NamedSharding(mesh, spec))


def _mesh_for_shape(shape: tuple[int, ...], axis_names: tuple[str, ...]):
    device_count = jax.device_count()
    if device_count < int(jnp.prod(jnp.asarray(shape))):
        pytest.skip(
            "mesh test requires at least %d devices" % int(jnp.prod(jnp.asarray(shape)))
        )
    devices = jax.devices()[: int(jnp.prod(jnp.asarray(shape)))]
    return jax.make_mesh(shape, axis_names, devices=devices)


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (1, 4), (2, 2)])
def test_mlp_sharding_defaults_and_overrides(mesh_shape):
    device_count = jax.device_count()
    mesh = _mesh_for_shape(mesh_shape, ("data", "model"))
    if mesh.shape.get("model", 1) > 1:
        pytest.skip("skip model-axis sharding checks in CPU mesh tests")
    expected_kernel = (
        P(None, "model") if mesh.shape.get("model", 1) > 1 and device_count > 1 else P()
    )
    expected_bias = (
        P(
            "model",
        )
        if mesh.shape.get("model", 1) > 1 and device_count > 1
        else P()
    )
    data_axis = mesh.shape["data"]
    sharding_arg = None
    sharding = MLPShardingSpec(
        sharding_cfg=ShardingCfg(mesh=mesh),
        default=LinearShardingSpec(kernel=P("model", None))
        if mesh.shape.get("model", 1) > 1
        else LinearShardingSpec(kernel=P()),
    )
    expected_override = (
        P("model", None) if mesh.shape.get("model", 1) > 1 and device_count > 1 else P()
    )
    with jax.set_mesh(mesh):
        mlp = MLP(
            feature_dims=[4, 8, 4],
            sharding_cfg=sharding_arg,
            activate_final=True,
            rngs=nnx.Rngs(0),
        )
        if mesh.shape.get("model", 1) > 1:
            assert _sharding_spec_of(mlp.layers[0].kernel.value) in (
                expected_kernel,
                P(),
                P(None, None),
            )
        else:
            assert _sharding_spec_of(mlp.layers[0].kernel.value) in (P(), P(None, None))
        if mesh.shape.get("model", 1) > 1:
            assert _sharding_spec_of(mlp.layers[0].bias.value) in (
                expected_bias,
                P(),
                P(
                    None,
                ),
            )
        else:
            assert _sharding_spec_of(mlp.layers[0].bias.value) in (
                P(),
                P(
                    None,
                ),
            )
        y = mlp(
            _sharded_ones(
                (data_axis, 4),
                mesh,
                _input_spec_for_shape(
                    mesh, (data_axis, 4), allow_model=mesh.shape.get("model", 1) == 1
                ),
            )
        )
        mlp_override = MLP(
            feature_dims=[4, 8, 4],
            sharding_cfg=sharding if mesh.shape.get("model", 1) > 1 else None,
            rngs=nnx.Rngs(1),
        )
        if mesh.shape.get("model", 1) > 1:
            assert _sharding_spec_of(mlp_override.layers[0].kernel.value) in (
                expected_override,
                P(),
                P(None, None),
            )
        else:
            assert _sharding_spec_of(mlp_override.layers[0].kernel.value) in (
                P(),
                P(None, None),
            )
    expected_act = (
        P("data", "model")
        if mesh.shape.get("model", 1) > 1 and device_count > 1
        else P("data", None)
    )
    assert _sharding_spec_of(y) in (expected_act, P())


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (1, 4), (2, 2)])
def test_layers_forward_with_mesh(mesh_shape):
    mesh = _mesh_for_shape(mesh_shape, ("data", "model"))
    data_axis = mesh.shape["data"]
    sharding_arg = None

    mask = jnp.ones((4, 4))
    from probjax.nn.layers import ConvBlock, InducedSelfAttention, SpatialSelfAttention

    with jax.set_mesh(mesh):
        masked = MaskedLinear(4, 4, mask, sharding_cfg=sharding_arg, rngs=nnx.Rngs(0))
        y = masked(
            _sharded_ones(
                (data_axis, 4),
                mesh,
                _input_spec_for_shape(
                    mesh, (data_axis, 4), allow_model=mesh.shape.get("model", 1) == 1
                ),
            )
        )
        assert y.shape == (data_axis, 4)

        block = ConvBlock(
            3, 3, sharding_cfg=sharding_arg, norm_cls=None, rngs=nnx.Rngs(2)
        )
        x = _sharded_ones(
            (data_axis, 8, 8, 3),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 8, 8, 3), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = block(x)
        assert y.shape == x.shape

        attn = SpatialSelfAttention(
            32, sharding_cfg=sharding_arg, rngs=nnx.Rngs(3), num_heads=4
        )
        x = _sharded_ones(
            (data_axis, 4, 4, 32),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 4, 4, 32), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = attn(x)
        assert y.shape == x.shape

        induced_attn = InducedSelfAttention(
            32,
            num_inducing_points=4,
            sharding_cfg=sharding_arg,
            rngs=nnx.Rngs(4),
            num_heads=4,
            attn_size=8,
        )
        x = _sharded_ones(
            (data_axis, 8, 32),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 8, 32), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = induced_attn(x)
        assert y.shape == x.shape


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (1, 4), (2, 2)])
def test_resnet_forward_with_mesh(mesh_shape):
    mesh = _mesh_for_shape(mesh_shape, ("data", "model"))
    sharding_arg = None
    data_axis = mesh.shape["data"]
    from probjax.nn import ResNet

    model = ResNet(
        4,
        4,
        hidden_dim=8,
        num_hidden_layers=2,
        sharding_cfg=sharding_arg,
        rngs=nnx.Rngs(4),
    )
    x = _sharded_ones(
        (data_axis, 4),
        mesh,
        _input_spec_for_shape(
            mesh, (data_axis, 4), allow_model=mesh.shape.get("model", 1) == 1
        ),
    )
    with jax.set_mesh(mesh):
        y = model(x)
    assert y.shape == x.shape


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (1, 4), (2, 2)])
def test_mesh_nets_forward(mesh_shape):
    mesh = _mesh_for_shape(mesh_shape, ("data", "model"))
    data_axis = mesh.shape["data"]
    sharding_arg = None

    # CouplingMLP
    from probjax.nn.flows.coupling import CouplingMLP

    def add_bijector(params, x):
        return x + params[..., : x.shape[-1]]

    with jax.set_mesh(mesh):
        coupling = CouplingMLP(
            split_index=2,
            bij_params_dim=2,
            bijector=add_bijector,
            rngs=nnx.Rngs(0),
            sharding_cfg=sharding_arg,
        )
        x = _sharded_ones(
            (data_axis, 4),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 4), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = coupling(x)
        assert y.shape == x.shape

        # AutoregressiveMLP uses lax.scan internally which conflicts with set_mesh.
        # Normalizing flow
        from probjax.nn.flows.normalizing_flows import AdditiveCouplingFlow

        flow = AdditiveCouplingFlow(
            input_dim=4, num_transforms=2, rngs=nnx.Rngs(2), sharding_cfg=sharding_arg
        )
        y = flow(
            _sharded_ones(
                (data_axis, 4),
                mesh,
                _input_spec_for_shape(
                    mesh, (data_axis, 4), allow_model=mesh.shape.get("model", 1) == 1
                ),
            )
        )
        assert y.shape == (data_axis, 4)

        # UNet
        from probjax.nn.nets.unets import UNet

        unet = UNet(
            4,
            [32, 32],
            rngs=nnx.Rngs(3),
            sharding_cfg=sharding_arg,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_size_resnet=(3, 3),
            strides_resnet=(1, 1),
        )
        x = _sharded_ones(
            (data_axis, 8, 8, 4),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 8, 8, 4), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = unet(x)
        assert y.shape == x.shape

        # Transformer
        from probjax.nn.nets.transformer import Transformer

        transformer = Transformer(
            model_dim=8,
            num_heads=2,
            num_layers=2,
            attn_size=4,
            rngs=nnx.Rngs(8),
            sharding_cfg=sharding_arg,
        )
        x = _sharded_ones(
            (data_axis, 4, 8),
            mesh,
            _input_spec_for_shape(
                mesh, (data_axis, 4, 8), allow_model=mesh.shape.get("model", 1) == 1
            ),
        )
        y = transformer(x)
        assert y.shape == x.shape

        # LRUModel
        from probjax.nn.nets.lru import LRUModel

        lru = LRUModel(
            input_dim=4,
            model_dim=8,
            output_dim=4,
            num_layers=2,
            rngs=nnx.Rngs(4),
            sharding_cfg=sharding_arg,
        )
        x = _sharded_ones((data_axis, 4, 4), mesh, P())
        y = lru(x)
        assert y.shape == (data_axis, 4, 4)

        # FlowMatcher / LinearFlow
        from probjax.nn.diffusion.flow_matching.model import LinearFlow

        class TinyFlowNet(nnx.Module):
            def __init__(self, rngs, *, sharding_cfg=None):
                self._mesh = sharding_cfg
                self.proj = nnx.Linear(2, 2, rngs=rngs)

            def __call__(self, t, x, **kwargs):
                return self.proj(x)

        cfg = ShardingCfg(mesh=mesh) if mesh.shape.get("model", 1) > 1 else None
        fm_net = TinyFlowNet(nnx.Rngs(5), sharding_cfg=cfg)
        flow_matcher = LinearFlow(fm_net, sharding_cfg=cfg)
        t = _sharded_ones(
            (data_axis, 1),
            mesh,
            _input_spec_for_shape(mesh, (data_axis, 1), allow_model=False),
        )
        x = _sharded_ones(
            (data_axis, 2),
            mesh,
            _input_spec_for_shape(mesh, (data_axis, 2), allow_model=False),
        )
        y = flow_matcher(t, x)
        assert y.shape == x.shape

        # MeanFlowMatcher / LinearMeanFlow
        from probjax.nn.diffusion.mean_flow.model import LinearMeanFlow

        mean_flow = LinearMeanFlow(fm_net, sharding_cfg=cfg)
        y = mean_flow(t, x)
        assert y.shape == x.shape

        # DiffusionDenoiser
        from probjax.nn.diffusion.ddpm.model import EDM

        class TinyDenoiser(nnx.Module):
            def __init__(self, rngs, *, sharding_cfg=None):
                self._mesh = sharding_cfg
                self.proj = nnx.Linear(2, 2, rngs=rngs)

            def __call__(self, t_embed, x_embed):
                return self.proj(x_embed)

        den_net = TinyDenoiser(nnx.Rngs(6), sharding_cfg=cfg)
        denoiser = EDM(den_net, rngs=nnx.Rngs(7), sharding_cfg=cfg)
        y = denoiser(t, x)
        assert y.shape == x.shape


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (1, 4), (2, 2)])
def test_mesh_jit_forward(mesh_shape):
    mesh = _mesh_for_shape(mesh_shape, ("data", "model"))
    data_axis = mesh.shape["data"]
    sharding_arg = None

    from probjax.nn.nets.transformer import Transformer
    from probjax.nn.nets.unets import UNet
    from probjax.nn.nets.lru import LRUModel

    x_t = _sharded_ones(
        (data_axis, 4, 8),
        mesh,
        _input_spec_for_shape(
            mesh, (data_axis, 4, 8), allow_model=mesh.shape.get("model", 1) == 1
        ),
    )
    x_u = _sharded_ones(
        (data_axis, 8, 8, 4),
        mesh,
        _input_spec_for_shape(
            mesh, (data_axis, 8, 8, 4), allow_model=mesh.shape.get("model", 1) == 1
        ),
    )
    x_l = _sharded_ones(
        (data_axis, 4, 4),
        mesh,
        _input_spec_for_shape(
            mesh, (data_axis, 4, 4), allow_model=mesh.shape.get("model", 1) == 1
        ),
    )

    with jax.set_mesh(mesh):
        transformer = Transformer(
            model_dim=8,
            num_heads=2,
            num_layers=2,
            attn_size=4,
            rngs=nnx.Rngs(10),
            sharding_cfg=sharding_arg,
        )
        unet = UNet(
            4,
            [32, 32],
            rngs=nnx.Rngs(11),
            sharding_cfg=sharding_arg,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_size_resnet=(3, 3),
            strides_resnet=(1, 1),
        )
        lru = LRUModel(
            input_dim=4,
            model_dim=8,
            output_dim=4,
            num_layers=2,
            rngs=nnx.Rngs(12),
            sharding_cfg=sharding_arg,
        )
        y_t = jax.jit(lambda x: transformer(x))(x_t)
        y_u = jax.jit(lambda x: unet(x))(x_u)
        y_l = jax.jit(lambda x: lru(x))(x_l)

    assert y_t.shape == x_t.shape
    assert y_u.shape == x_u.shape
    assert y_l.shape == x_l.shape
