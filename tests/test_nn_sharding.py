"""No-mesh behavior of probjax.nn.sharding (runs in the default suite).

The mesh-dependent behavior is covered by tests/test_nn_mesh.py (run with
``-m mesh`` and multiple host devices).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn import MLP, MoELayer, Transformer
from probjax.nn import sharding as shd


def test_no_mesh_helpers_are_noops():
    assert shd.resolve_axes(shd.EMBED, shd.HIDDEN) is None
    assert shd.param_metadata(shd.EMBED, shd.HIDDEN) == {}

    x = jnp.ones((4, 3))
    assert shd.constrain(x, shd.BATCH) is x
    assert shd.replicate(x) is x


def test_no_mesh_construction_carries_no_sharding_annotation():
    # Guards against flax's always-shard error: with no mesh, variables must
    # not carry a sharding annotation at all.
    model = MLP([4, 16, 16, 2], rngs=nnx.Rngs(0))
    for layer in model.layers:
        assert "sharding" not in layer.kernel.get_metadata()

    transformer = Transformer(
        model_dim=8, num_heads=2, num_layers=1, attn_size=4, rngs=nnx.Rngs(0)
    )
    attn = transformer.attention_blocks[0]
    assert "sharding" not in attn.query.kernel.get_metadata()


def test_default_rules_cover_all_names():
    rules = dict(shd.default_rules())
    for name in (
        shd.BATCH,
        shd.SEQ,
        shd.EMBED,
        shd.HIDDEN,
        shd.HEADS,
        shd.HEAD_DIM,
        shd.EXPERT,
    ):
        assert name in rules
    assert rules[shd.BATCH] == "data"
    assert rules[shd.HIDDEN] == "model"
    assert rules[shd.HEADS] == "model"
    # Experts shard over the model axis by default ...
    assert rules[shd.EXPERT] == "model"
    # ... or over a dedicated axis on a 3-D mesh.
    assert dict(shd.default_rules(expert_axis="expert"))[shd.EXPERT] == "expert"


def test_no_mesh_moe_carries_no_sharding_annotation():
    moe = MoELayer(8, 16, 4, rngs=nnx.Rngs(0))
    for name in ("w_gate", "w_up", "w_down"):
        assert "sharding" not in getattr(moe.experts, name).get_metadata()
    assert "sharding" not in moe.router.w_router.get_metadata()


def test_moe_aux_is_jit_compatible_pytree():
    moe = MoELayer(8, 16, 4, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 8))

    def loss_fn(model, x):
        _, aux = model(x, return_aux=True)
        return jnp.sum(aux.expert_counts) + aux.load_balance_loss

    leaves, treedef = jax.tree_util.tree_flatten(moe(x, return_aux=True)[1])
    assert len(leaves) == 6  # capacity is static
    assert jnp.isfinite(nnx.jit(loss_fn)(moe, x))

    # return_aux itself works under jit.
    y, aux = nnx.jit(lambda m, x: m(x, return_aux=True))(moe, x)
    assert y.shape == (4, 8)
    assert aux.capacity >= 1


def test_size_one_axis_resolves_to_replicated():
    mesh = jax.make_mesh(
        (1, 1),
        ("data", "model"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    with jax.set_mesh(mesh):
        # Both axes have size 1 -> nothing resolves.
        assert shd.resolve_axes(shd.EMBED, shd.HIDDEN) is None
        assert shd.param_metadata(shd.BATCH) == {}
