"""Per-shard (GShard-style) MoE routing under a data-parallel mesh.

Run with ``pytest -m mesh`` (8 host devices on CPU).  ``MoELayer`` routes
each data shard's tokens on its own device by default; these tests pin the
semantics (dropless/dense exact, capacity per shard, global-batch aux
statistics), the gradients, the fallbacks and the collectives.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from probjax.nn import MoELayer

# Exact comparisons against single-device references: no TF32 on GPU.
pytestmark = pytest.mark.float32_matmul

KW = dict(d_model=8, d_hidden=16, num_experts=4, top_k=2)
AUX_FIELDS = (
    "load_balance_loss",
    "router_z_loss",
    "expert_counts",
    "expert_utilization",
    "router_probabilities",
    "overflow_fraction",
)


def _mesh(shape, axis_type=AxisType.Auto):
    total = int(np.prod(shape))
    if jax.device_count() < total:
        pytest.skip(f"mesh test requires at least {total} devices")
    return jax.make_mesh(
        shape,
        ("data", "model"),
        devices=jax.devices()[:total],
        axis_types=(axis_type,) * len(shape),
    )


def _shard(mesh, x):
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def _loss(model, x, mask=None):
    y, aux = model(x, return_aux=True, token_mask=mask)
    return jnp.sum(y**2) + aux.load_balance_loss + 0.1 * aux.router_z_loss


def _grads(model, x, mask=None):
    return jax.tree.leaves(nnx.grad(_loss)(model, x, mask))


def _close(a, b, tol=1e-5):
    np.testing.assert_allclose(
        np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b)), atol=tol, rtol=tol
    )


# (routing, dropless_impl): "pallas" is the GPU default (interpret mode on
# CPU); it runs fully manual, with the experts gathered over "model".
ROUTINGS = [("dense", "auto"), ("dropless", "auto"), ("dropless", "pallas")]


@pytest.mark.mesh
@pytest.mark.parametrize(("routing", "impl"), ROUTINGS)
@pytest.mark.parametrize("mesh_shape", [(2, 1), (4, 1), (2, 2)])
def test_local_routing_is_exact_for_dense_and_dropless(routing, impl, mesh_shape):
    """Each token's output depends only on its own routing, so per-shard
    routing changes nothing, also with experts sharded over "model"."""
    x = jax.random.normal(jax.random.key(0), (8, 6, KW["d_model"]))
    mask = jax.random.bernoulli(jax.random.key(1), 0.7, (8, 6))
    kw = dict(KW, routing=routing, dropless_impl=impl)
    reference = MoELayer(**kw, rngs=nnx.Rngs(3))
    y_ref, aux_ref = reference(x, return_aux=True, token_mask=mask)
    g_ref = _grads(reference, x, mask)

    mesh = _mesh(mesh_shape)
    with jax.set_mesh(mesh):
        model = MoELayer(**kw, rngs=nnx.Rngs(3))
        xs, ms = _shard(mesh, x), _shard(mesh, mask)
        y, aux = nnx.jit(lambda m, x, k: m(x, return_aux=True, token_mask=k))(
            model, xs, ms
        )
        grads = jax.tree.leaves(nnx.jit(_grads)(model, xs, ms))
    _close(y_ref, y)
    assert y.sharding.spec[0] == "data"
    for field in AUX_FIELDS:
        _close(getattr(aux_ref, field), getattr(aux, field))
    for a, b in zip(g_ref, grads, strict=True):
        _close(a, b, tol=1e-4)


@pytest.mark.mesh
@pytest.mark.parametrize("mesh_shape", [(2, 1), (2, 2)])
def test_capacity_is_per_shard_and_aux_stays_global(mesh_shape):
    """With capacity routing every data shard is one GShard group: outputs
    equal routing each shard's tokens on its own; the balance statistics
    are those of the whole batch."""
    # Few tokens per expert so the capacity actually drops assignments.
    x = jax.random.normal(jax.random.key(0), (4, 6, KW["d_model"]))
    reference = MoELayer(**KW, capacity_factor=0.5, rngs=nnx.Rngs(7))
    per_group = jnp.concatenate([reference(x[:2]), reference(x[2:])])
    _, aux_global = reference(x, return_aux=True)
    g_ref = jax.tree.leaves(
        nnx.grad(lambda m: jnp.sum(m(x[:2]) ** 2) + jnp.sum(m(x[2:]) ** 2))(reference)
    )

    mesh = _mesh(mesh_shape)
    with jax.set_mesh(mesh):
        model = MoELayer(**KW, capacity_factor=0.5, rngs=nnx.Rngs(7))
        xs = _shard(mesh, x)
        y, aux = nnx.jit(lambda m, x: m(x, return_aux=True))(model, xs)
        grads = jax.tree.leaves(
            nnx.jit(nnx.grad(lambda m, x: jnp.sum(m(x) ** 2)))(model, xs)
        )
    _close(per_group, y)
    assert aux.capacity == 3  # ceil(0.5 * 12 * 2 / 4) for 12 tokens per shard
    assert float(aux.overflow_fraction) > 0
    for field in ("load_balance_loss", "router_z_loss", "expert_counts"):
        _close(getattr(aux_global, field), getattr(aux, field))
    for a, b in zip(g_ref, grads, strict=True):
        _close(a, b, tol=1e-4)


@pytest.mark.mesh
def test_local_routing_false_keeps_global_capacity():
    x = jax.random.normal(jax.random.key(0), (4, 6, KW["d_model"]))
    reference = MoELayer(**KW, capacity_factor=0.5, rngs=nnx.Rngs(7))
    mesh = _mesh((2, 1))
    with jax.set_mesh(mesh):
        model = MoELayer(
            **KW, capacity_factor=0.5, local_routing=False, rngs=nnx.Rngs(7)
        )
        y = nnx.jit(lambda m, x: m(x))(model, _shard(mesh, x))
    _close(reference(x), y)


@pytest.mark.mesh
def test_indivisible_batch_falls_back_to_global_routing():
    x = jax.random.normal(jax.random.key(0), (3, 5, KW["d_model"]))
    reference = MoELayer(**KW, capacity_factor=0.5, rngs=nnx.Rngs(7))
    mesh = _mesh((2, 1))
    with jax.set_mesh(mesh):
        model = MoELayer(**KW, capacity_factor=0.5, rngs=nnx.Rngs(7))
        y = nnx.jit(lambda m, x: m(x))(model, x)
    _close(reference(x), y)


@pytest.mark.mesh
def test_local_routing_eager_and_explicit_mesh():
    """Eager calls and autodiff (no outer jit) and Explicit-axes meshes."""
    x = jax.random.normal(jax.random.key(0), (4, 6, KW["d_model"]))
    reference = MoELayer(**KW, routing="dropless", rngs=nnx.Rngs(1))
    g_ref = _grads(reference, x)
    for axis_type in (AxisType.Auto, AxisType.Explicit):
        mesh = _mesh((2, 1), axis_type)
        with jax.set_mesh(mesh):
            model = MoELayer(**KW, routing="dropless", rngs=nnx.Rngs(1))
            xs = _shard(mesh, x)
            y = model(xs)
            grads = _grads(model, xs)
        _close(reference(x), y)
        for a, b in zip(g_ref, grads, strict=True):
            _close(a, b, tol=1e-4)


@pytest.mark.mesh
def test_local_routing_under_vmap_routes_globally():
    x = jax.random.normal(jax.random.key(0), (2, 4, 6, KW["d_model"]))
    # Capacity routing: the Pallas grouped GEMM has no vmap rule for
    # unbatched group layouts, with or without a mesh.
    reference = MoELayer(**KW, rngs=nnx.Rngs(1))
    mesh = _mesh((2, 1))
    with jax.set_mesh(mesh):
        model = MoELayer(**KW, rngs=nnx.Rngs(1))
        y = nnx.vmap(lambda m, x: m(x), in_axes=(None, 0))(model, x)
    _close(jax.vmap(reference)(x), y)


_COLLECTIVE = re.compile(
    r"=\s+(\([^=]*?\)|\S+)\s+"
    r"(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)"
    r"(?:-start)?\("
)
_ARRAY = re.compile(r"[a-z]+[0-9]*\[([0-9,]*)\]")


def _collectives(hlo):
    """``(op, bytes)`` per collective in a compiled HLO module (f32/s32 only)."""
    out = []
    for shape, op in _COLLECTIVE.findall(hlo):
        n = sum(
            int(np.prod([int(d) for d in dims.split(",") if d]))
            for dims in _ARRAY.findall(shape)
        )
        out.append((op, 4 * n))
    return out


@pytest.mark.mesh
@pytest.mark.parametrize("routing", ["capacity", "dense", "dropless"])
def test_data_parallel_step_only_all_reduces_parameter_gradients(routing):
    """Global routing all-gathers routing ids and all-reduces token buffers
    every step; per-shard routing leaves only the gradient all-reduce."""
    mesh = _mesh((2, 1))
    with jax.set_mesh(mesh):
        model = MoELayer(32, 64, 8, top_k=2, routing=routing, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(model)
        param_bytes = sum(4 * leaf.size for leaf in jax.tree.leaves(state))
        x = _shard(mesh, jnp.ones((8, 256, 32)))
        step = jax.jit(jax.grad(lambda s, x: jnp.sum(nnx.merge(graphdef, s)(x) ** 2)))
        found = _collectives(step.lower(state, x).compile().as_text())
    assert found, "expected the gradient all-reduce"
    assert {op for op, _ in found} == {"all-reduce"}, found
    assert sum(b for _, b in found) <= param_bytes, (found, param_bytes)
