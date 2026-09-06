"""Multi-device sharding tests for probjax.nn (flax eager sharding).

Run with multiple host devices, e.g.::

    XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest -m mesh

Models annotate their parameters with logical axis names; constructing them
under ``jax.set_mesh(mesh)`` shards the parameters eagerly. These tests
assert the exact kernel shardings (no permissive fallbacks) and the
numerical equivalence of sharded and unsharded models.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from probjax.nn import MLP, MoELayer, SSMModel, Transformer, UNet, maf
from probjax.nn import sharding as shd


def _auto_mesh(shape):
    total = 1
    for dim in shape:
        total *= dim
    if jax.device_count() < total:
        pytest.skip(f"mesh test requires at least {total} devices")
    return jax.make_mesh(
        shape,
        ("data", "model"),
        devices=jax.devices()[:total],
        axis_types=(AxisType.Auto,) * len(shape),
    )


def _batch(mesh, shape):
    return jax.device_put(
        jnp.ones(shape), NamedSharding(mesh, P("data", *([None] * (len(shape) - 1))))
    )


@pytest.mark.mesh
def test_mlp_kernels_shard_megatron_style():
    mesh = _auto_mesh((1, 4))
    with jax.set_mesh(mesh):
        model = MLP([8, 32, 32, 8], rngs=nnx.Rngs(0))
        assert model.layers[0].kernel[...].sharding.spec == P(None, "model")
        assert model.layers[0].bias[...].sharding.spec == P("model")
        assert model.layers[1].kernel[...].sharding.spec == P("model", None)
        assert model.layers[2].kernel[...].sharding.spec == P(None, "model")


@pytest.mark.mesh
def test_attention_kernels_shard_over_heads():
    mesh = _auto_mesh((1, 4))
    with jax.set_mesh(mesh):
        model = Transformer(
            model_dim=16, num_heads=4, num_layers=2, attn_size=4, rngs=nnx.Rngs(0)
        )
        attn = model.attention_blocks[0]
        assert attn.query.kernel[...].sharding.spec == P(None, "model", None)
        assert attn.key.kernel[...].sharding.spec == P(None, "model", None)
        assert attn.value.kernel[...].sharding.spec == P(None, "model", None)
        assert attn.out.kernel[...].sharding.spec == P("model", None, None)


@pytest.mark.mesh
def test_moe_experts_shard_over_model_router_replicated():
    mesh = _auto_mesh((1, 4))
    with jax.set_mesh(mesh):
        model = MoELayer(8, 16, 4, rngs=nnx.Rngs(0))
        for name in ("w_gate", "w_up", "w_down"):
            assert getattr(model.experts, name)[...].sharding.spec == P(
                "model", None, None
            )
        assert model.router.w_router[...].sharding.spec == P()


@pytest.mark.mesh
@pytest.mark.parametrize("sparse", [True, False])
def test_moe_sharded_matches_unsharded(sparse):
    mesh = _auto_mesh((2, 2))
    x = jax.random.normal(jax.random.key(0), (4, 6, 8))
    kwargs = dict(d_model=8, d_hidden=16, num_experts=4, top_k=2)

    reference = MoELayer(**kwargs, sparse=sparse, rngs=nnx.Rngs(42))
    y_ref, aux_ref = reference(x, return_aux=True)

    xs = jax.device_put(x, NamedSharding(mesh, P("data", None, None)))
    with jax.set_mesh(mesh):
        sharded = MoELayer(**kwargs, sparse=sparse, rngs=nnx.Rngs(42))
        y = nnx.jit(lambda m, x: m(x))(sharded, xs)
        # return_aux works under jit now that MoEAux is a pytree.
        _, aux = nnx.jit(lambda m, x: m(x, return_aux=True))(sharded, xs)
    assert jnp.allclose(y_ref, jax.device_get(y), atol=1e-5)
    assert jnp.allclose(
        aux_ref.load_balance_loss, jax.device_get(aux.load_balance_loss), atol=1e-5
    )
    # Output keeps its batch sharding (sparse path is re-constrained).
    assert y.sharding.spec[0] == "data"

    def loss_fn(model, x):
        out, aux = model(x, return_aux=True)
        return jnp.sum(out**2) + aux.load_balance_loss

    with jax.set_mesh(mesh):
        loss, grads = nnx.value_and_grad(loss_fn)(sharded, xs)
    assert jnp.isfinite(jax.device_get(loss))
    for leaf in jax.tree_util.tree_leaves(grads):
        if isinstance(leaf, jax.Array):
            assert jnp.all(jnp.isfinite(jax.device_get(leaf)))


@pytest.mark.mesh
def test_moe_expert_axis_on_3d_mesh():
    if jax.device_count() < 4:
        pytest.skip("mesh test requires at least 4 devices")
    mesh = jax.make_mesh(
        (2, 1, 2),
        ("data", "model", "expert"),
        devices=jax.devices()[:4],
        axis_types=(AxisType.Auto,) * 3,
    )
    with (
        jax.set_mesh(mesh),
        nnx.logical_axis_rules(shd.default_rules(expert_axis="expert")),
    ):
        model = MoELayer(8, 16, 4, rngs=nnx.Rngs(0))
        assert model.experts.w_gate[...].sharding.spec == P("expert", None, None)


@pytest.mark.mesh
def test_sharded_model_matches_unsharded_numerically():
    mesh = _auto_mesh((2, 4))
    x = jnp.ones((8, 8))

    reference = MLP([8, 32, 32, 8], rngs=nnx.Rngs(42))
    y_ref = reference(x)

    with jax.set_mesh(mesh):
        sharded = MLP([8, 32, 32, 8], rngs=nnx.Rngs(42))
        y_sharded = sharded(_batch(mesh, (8, 8)))
    assert jnp.allclose(y_ref, jax.device_get(y_sharded), atol=1e-5)

    t_ref = Transformer(
        model_dim=16, num_heads=4, num_layers=2, attn_size=4, rngs=nnx.Rngs(7)
    )
    ty_ref = t_ref(jnp.ones((8, 5, 16)))
    with jax.set_mesh(mesh):
        t_sharded = Transformer(
            model_dim=16, num_heads=4, num_layers=2, attn_size=4, rngs=nnx.Rngs(7)
        )
        ty_sharded = t_sharded(_batch(mesh, (8, 5, 16)))
    assert jnp.allclose(ty_ref, jax.device_get(ty_sharded), atol=1e-4)


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(4, 1), (2, 2)])
def test_data_parallel_forward(mesh_shape):
    mesh = _auto_mesh(mesh_shape)
    with jax.set_mesh(mesh):
        mlp = MLP([8, 32, 8], rngs=nnx.Rngs(0))
        y = mlp(_batch(mesh, (8, 8)))
        assert y.sharding.spec[0] == "data"

        jitted = jax.jit(mlp)
        y_jit = jitted(_batch(mesh, (8, 8)))
        assert jnp.allclose(y, y_jit, atol=1e-6)

        transformer = Transformer(
            model_dim=16, num_heads=2, num_layers=1, attn_size=8, rngs=nnx.Rngs(0)
        )
        ty = transformer(_batch(mesh, (8, 5, 16)))
        assert ty.sharding.spec[0] == "data"

        unet = UNet(
            4,
            [32, 32],
            rngs=nnx.Rngs(0),
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_size_resnet=(3, 3),
            strides_resnet=(1, 1),
        )
        uy = unet(_batch(mesh, (8, 8, 8, 4)))
        assert uy.shape == (8, 8, 8, 4)

        ssm = SSMModel(
            input_dim=4, model_dim=8, output_dim=4, num_layers=1, rngs=nnx.Rngs(0)
        )
        ly = ssm(_batch(mesh, (8, 6, 4)))
        assert ly.shape == (8, 6, 4)


@pytest.mark.mesh
def test_explicit_axes_mesh_constrain_branch():
    # jax.make_mesh defaults to Explicit axis types; constrain must take the
    # reshard branch there.
    total = 4
    if jax.device_count() < total:
        pytest.skip(f"mesh test requires at least {total} devices")
    mesh = jax.make_mesh((4, 1), ("data", "model"), devices=jax.devices()[:total])
    with jax.set_mesh(mesh):
        mlp = MLP([8, 16, 8], rngs=nnx.Rngs(0))
        x = jax.device_put(jnp.ones((8, 8)), NamedSharding(mesh, P("data", None)))
        y = mlp(x)
        assert y.shape == (8, 8)


@pytest.mark.mesh
def test_fit_under_mesh():
    mesh = _auto_mesh((4, 1))
    data = jax.random.normal(jax.random.key(0), (64, 2))
    with jax.set_mesh(mesh):
        flow = maf(2, 2, rngs=nnx.Rngs(0))
        sharded_data = jax.device_put(data, NamedSharding(mesh, P("data", None)))
        losses = flow.fit(jax.random.key(1), sharded_data, num_steps=5)
        assert jnp.all(jnp.isfinite(losses))


@pytest.mark.mesh
def test_autoregressive_flow_logpdf_under_mesh():
    # AutoregressiveMLP uses lax.scan internally; historically this conflicted
    # with set_mesh under Explicit axis types. Verify it works with Auto axes.
    mesh = _auto_mesh((4, 1))
    with jax.set_mesh(mesh):
        flow = maf(2, 2, rngs=nnx.Rngs(0))
        distribution = flow.as_dist()
        x = jax.device_put(jnp.ones((8, 2)), NamedSharding(mesh, P("data", None)))
        logprob = distribution.logpdf(x)
        assert logprob.shape == (8,)
        assert jnp.all(jnp.isfinite(logprob))


@pytest.mark.mesh
def test_flow_logpdf_per_shard_no_allgather():
    # batch_shard wraps the custom_inverse evaluation: the flow inverse runs
    # per-shard under a batch-sharded mesh with no all-gathers.
    mesh = _auto_mesh((4, 1))
    flow = maf(2, 2, rngs=nnx.Rngs(0))
    distribution = flow.as_dist()
    x = jnp.ones((8, 2))
    lp_ref = distribution.logpdf(x)
    xs = jax.device_put(x, NamedSharding(mesh, P("data", None)))
    with jax.set_mesh(mesh):
        fn = jax.jit(distribution.logpdf)
        lp = fn(xs)
        assert jnp.allclose(lp_ref, jax.device_get(lp), atol=1e-5)
        assert lp.sharding.spec[0] == "data"
        hlo = fn.lower(xs).compile().as_text()
        assert "all-gather" not in hlo
