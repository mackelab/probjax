"""Grouped-GEMM kernel behind dropless MoE routing.

Reference is ``jax.lax.ragged_dot`` at HIGHEST precision.  Runs the Triton
kernel on GPU and the same kernel in Pallas interpret mode on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from probjax.nn.layers.moe import MoELayer
from probjax.nn.layers.grouped_matmul import (
    gather_from_layout,
    grouped_matmul,
    make_group_layout,
    scatter_to_layout,
)

TOL = 2e-3  # tf32/bf16-class accumulation on GPU


def _ours(gs, x, w, block_m=16):
    layout = make_group_layout(gs, x.shape[0], block_m)
    return gather_from_layout(
        grouped_matmul(scatter_to_layout(x, layout), w, layout, block_m=block_m), layout
    )


def _ref(gs, x, w):
    return jax.lax.ragged_dot(x, w, gs, precision=jax.lax.Precision.HIGHEST)


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-12))


@pytest.mark.parametrize("sizes", [
    [10, 0, 37, 1, 16],      # empty and single-row groups, unaligned sizes
    [64, 0, 0, 0, 0],        # everything in one group
    [0, 0, 0, 0, 64],        # ...the last one
])
def test_layout_puts_each_block_in_one_group(sizes):
    gs = jnp.array(sizes, jnp.int32)
    layout = make_group_layout(gs, int(gs.sum()), 16)
    rows = np.asarray(layout.padded_rows)
    assert len(set(rows.tolist())) == len(rows)            # no collisions
    owner = np.repeat(np.arange(len(sizes)), sizes)
    assert (np.asarray(layout.block_expert)[rows // 16] == owner).all()
    starts, ends = np.asarray(layout.block_start), np.asarray(layout.block_end)
    for g, n in enumerate(sizes):
        assert ends[g] - starts[g] == -(-n // 16)


@pytest.mark.parametrize("sizes,k,n", [
    ([10, 0, 37, 1, 16], 32, 64),
    ([64, 0, 0, 0, 0], 32, 64),
    ([7, 9, 0, 3, 5], 8, 24),   # dims below / not a multiple of 16: zero-padded
])
def test_forward_and_gradients_match_ragged_dot(sizes, k, n):
    gs = jnp.array(sizes, jnp.int32)
    m = int(gs.sum())
    x = jax.random.normal(jax.random.key(0), (m, k))
    w = jax.random.normal(jax.random.key(1), (5, k, n)) / 6
    assert _rel(jax.jit(_ours)(gs, x, w), _ref(gs, x, w)) < TOL

    def loss(fn):
        return lambda x, w: jnp.sum(jnp.sin(fn(gs, x, w)))

    ga = jax.jit(jax.grad(loss(_ours), argnums=(0, 1)))(x, w)
    gb = jax.grad(loss(_ref), argnums=(0, 1))(x, w)
    for a, b in zip(ga, gb):
        assert _rel(a, b) < TOL


def test_moe_layer_pallas_matches_dense():
    fast = MoELayer(16, 32, 4, routing="dropless", dropless_impl="pallas", rngs=nnx.Rngs(0))
    dense = MoELayer(16, 32, 4, routing="dense", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (3, 21, 16))
    assert _rel(fast(x), dense(x)) < TOL


def test_moe_layer_pallas_gradients_match_dense():
    fast = MoELayer(16, 32, 4, routing="dropless", dropless_impl="pallas", rngs=nnx.Rngs(0))
    dense = MoELayer(16, 32, 4, routing="dense", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(3), (40, 16))
    loss = lambda m: jnp.sum(m(x) ** 2)
    for a, b in zip(jax.tree.leaves(nnx.grad(loss)(fast)), jax.tree.leaves(nnx.grad(loss)(dense))):
        assert _rel(a, b) < TOL


def test_sharded_weights_match_replicated():
    """Tensor-parallel on the output columns: the custom_partitioning rule
    runs the kernel per shard instead of gathering w."""
    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("needs 2 devices")
    mesh = Mesh(np.array(devices), ("model",))
    gs = jnp.array([20, 0, 44], jnp.int32)
    x = jax.random.normal(jax.random.key(4), (64, 32))
    w = jax.random.normal(jax.random.key(5), (3, 32, 64)) / 6
    w_sharded = jax.device_put(w, NamedSharding(mesh, P(None, None, "model")))
    out = jax.jit(_ours)(gs, x, w_sharded)
    assert _rel(out, _ref(gs, x, w)) < TOL


@pytest.mark.skipif(
    not __import__("probjax.nn.pallas_kernels.kernels.grouped_matmul",
                   fromlist=["is_hopper"]).is_hopper(),
    reason="Mosaic-GPU ragged dot needs an sm_90 GPU",
)
def test_hopper_mosaic_path_matches_ragged_dot():
    from probjax.nn.layers.grouped_matmul import ragged_matmul_hopper

    gs = jnp.array([100, 0, 300, 112], jnp.int32)
    x = jax.random.normal(jax.random.key(0), (512, 256), jnp.bfloat16)
    w = jax.random.normal(jax.random.key(1), (4, 256, 384), jnp.bfloat16) / 16
    f = lambda fn: (lambda x, w: jnp.sum(jnp.sin(fn(x, w).astype(jnp.float32))))
    ours = lambda x, w: ragged_matmul_hopper(x, w, gs)
    ref = lambda x, w: jax.lax.ragged_dot(x.astype(jnp.float32), w.astype(jnp.float32), gs)
    assert _rel(jax.jit(ours)(x, w).astype(jnp.float32), ref(x, w)) < 1e-2
    ga = jax.jit(jax.grad(f(ours), argnums=(0, 1)))(x, w)
    gb = jax.grad(f(ref), argnums=(0, 1))(x, w)
    for a, b in zip(ga, gb):
        assert _rel(a.astype(jnp.float32), b) < 1e-2
