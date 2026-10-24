import jax
import jax.numpy as jnp
import pytest
import numpy as np
from flax import nnx

from probjax.nn.layers.moe import ExpertSwiGLU, MoELayer, load_balance_loss


def _make_layer(
    d_model=8, d_hidden=16, num_experts=4, top_k=2, capacity_factor=4.0, sparse=True
):
    return MoELayer(
        d_model,
        d_hidden,
        num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        sparse=sparse,
        rngs=nnx.Rngs(0),
    )


def test_output_shape():
    moe = _make_layer()
    x = jnp.ones((2, 5, 8))
    y = moe(x)
    assert y.shape == (2, 5, 8)


def test_top_k_1():
    moe = _make_layer(top_k=1)
    x = jax.random.normal(jax.random.key(1), (6, 8))
    y = moe(x)
    assert y.shape == (6, 8)
    assert jnp.all(jnp.isfinite(y))


def test_finite_gradients():
    moe = _make_layer()
    x = jax.random.normal(jax.random.key(3), (6, 8))

    def loss_fn(model, x):
        y, aux = model(x, return_aux=True)
        return jnp.sum(y**2) + aux.load_balance_loss

    loss, grads = nnx.value_and_grad(loss_fn)(moe, x)
    assert jnp.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves if isinstance(g, jax.Array))

    # Input gradients are finite too.
    def loss_x(x):
        return jnp.sum(moe(x))

    gx = jax.grad(loss_x)(x)
    assert jnp.all(jnp.isfinite(gx))


def test_jit():
    moe = _make_layer()
    x = jax.random.normal(jax.random.key(4), (2, 6, 8))
    jitted = nnx.jit(lambda m, x: m(x))
    y1 = moe(x)
    y2 = jitted(moe, x)
    assert jnp.allclose(y1, y2, atol=1e-5)


def test_deterministic():
    moe = _make_layer()
    x = jax.random.normal(jax.random.key(5), (4, 8))
    assert jnp.array_equal(moe(x), moe(x))


def test_dense_against_manual():
    d_model, d_hidden, num_experts, top_k = 4, 8, 3, 2
    moe = MoELayer(
        d_model,
        d_hidden,
        num_experts,
        top_k=top_k,
        sparse=False,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(6), (5, d_model))
    y = moe(x)

    # Manual SwiGLU over all experts + top-k gather.
    w_gate = moe.experts.gate.kernel[...]
    w_up = moe.experts.up.kernel[...]
    w_down = moe.experts.down.kernel[...]
    logits = x @ moe.router.kernel[...]
    top_logits, expert_ids = jax.lax.top_k(logits, top_k)
    weights = jax.nn.softmax(top_logits, axis=-1)
    gate = jnp.einsum("nd,edh->neh", x, w_gate)
    up = jnp.einsum("nd,edh->neh", x, w_up)
    expert_y = jnp.einsum("neh,ehd->ned", jax.nn.silu(gate) * up, w_down)
    expected = jnp.sum(
        jnp.take_along_axis(expert_y, expert_ids[..., None], axis=1)
        * weights[..., None],
        axis=1,
    )
    assert jnp.allclose(y, expected, atol=1e-5)


def test_sparse_matches_dense_without_dropping():
    kwargs = dict(d_model=8, d_hidden=16, num_experts=4, top_k=2)
    sparse = MoELayer(**kwargs, capacity_factor=4.0, sparse=True, rngs=nnx.Rngs(0))
    dense = MoELayer(**kwargs, sparse=False, rngs=nnx.Rngs(0))
    # Same rngs -> identical parameters.

    x = jax.random.normal(jax.random.key(7), (8, 8))
    y_sparse, aux = sparse(x, return_aux=True)
    y_dense = dense(x)
    assert aux.overflow_fraction == pytest.approx(0.0)
    assert jnp.allclose(y_sparse, y_dense, atol=1e-5)


def test_aux_statistics():
    moe = _make_layer()
    x = jax.random.normal(jax.random.key(8), (8, 8))
    _, aux = moe(x, return_aux=True)
    n, k, e = 8, moe.top_k, moe.num_experts
    assert aux.expert_counts.shape == (e,)
    assert jnp.allclose(jnp.sum(aux.expert_counts), n * k)
    assert jnp.allclose(jnp.sum(aux.expert_utilization), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(aux.router_probabilities), 1.0, atol=1e-6)
    assert 0.0 <= float(aux.overflow_fraction) <= 1.0
    assert float(aux.load_balance_loss) >= 1.0 - 1e-5  # E*sum(p f) >= 1
    assert aux.capacity >= 1
    assert float(aux.router_z_loss) >= 0.0


def test_router_runs_in_float32_under_bf16():
    moe = MoELayer(8, 16, 4, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(11), (6, 8), dtype=jnp.bfloat16)
    assert moe.router(x).dtype == jnp.float32
    assert moe(x).dtype == jnp.bfloat16


def test_expert_contract():
    experts = ExpertSwiGLU(3, d_model=8, d_hidden=16, rngs=nnx.Rngs(0))
    packed = jax.random.normal(jax.random.key(9), (3, 5, 8))
    assert experts(packed).shape == (3, 5, 8)
    with pytest.raises(ValueError):
        experts(jnp.ones((2, 5, 8)))  # wrong expert axis


def test_capacity_overflow_drops_assignments():
    moe = _make_layer(top_k=1, capacity_factor=0.25)
    x = jax.random.normal(jax.random.key(10), (16, 8))
    y, aux = moe(x, return_aux=True)
    assert aux.capacity == 1
    assert float(aux.overflow_fraction) > 0.0
    assert jnp.all(aux.capacity_utilization <= 1.0)
    # Dropped tokens produce zero output.
    n_zero_rows = int(jnp.sum(jnp.all(y == 0, axis=-1)))
    assert n_zero_rows == round(float(aux.overflow_fraction) * 16)


def test_load_balance_loss_uniform():
    probs = jnp.full((10, 4), 0.25)
    ids = jnp.tile(jnp.arange(4)[None, :], (10, 1))[:, :2]
    loss = load_balance_loss(probs, ids, 4)
    assert loss == pytest.approx(1.0)


def _routing_pair(routing, **kw):
    """Two layers with identical parameters: `routing` and the dense reference."""
    a = MoELayer(8, 16, 4, routing=routing, rngs=nnx.Rngs(0), **kw)
    b = MoELayer(8, 16, 4, routing="dense", rngs=nnx.Rngs(0), **kw)
    return a, b


def test_dropless_matches_dense():
    moe, ref = _routing_pair("dropless")
    x = jax.random.normal(jax.random.key(3), (3, 7, 8))
    assert jnp.allclose(moe(x), ref(x), atol=1e-5)


def test_dropless_gradients_match_dense():
    moe, ref = _routing_pair("dropless")
    x = jax.random.normal(jax.random.key(4), (12, 8))

    def loss(m):
        return jnp.sum(m(x) ** 2)

    ga = nnx.grad(loss)(moe)
    gb = nnx.grad(loss)(ref)
    for la, lb in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
        assert jnp.allclose(la, lb, atol=1e-4)


@pytest.mark.parametrize("impl", ["triton", "ragged_dot", "pallas"])
def test_dropless_impls_match_dense(impl):
    """Every dropless backend computes the dense reference; "triton" forces the
    Triton grouped GEMM (interpret mode off-GPU), on Hopper too."""
    if impl == "pallas" and jax.default_backend() == "gpu":
        pytest.skip("pallas picks Mosaic on Hopper; covered by the GPU suite")
    moe, ref = _routing_pair("dropless", dropless_impl=impl)
    x = jax.random.normal(jax.random.key(5), (3, 7, 8))
    assert jnp.allclose(moe(x), ref(x), atol=1e-4)


def test_dropless_triton_gradients_match_dense():
    moe, ref = _routing_pair("dropless", dropless_impl="triton")
    x = jax.random.normal(jax.random.key(6), (12, 8))

    def loss(m):
        return jnp.sum(m(x) ** 2)

    for la, lb in zip(jax.tree.leaves(nnx.grad(loss)(moe)), jax.tree.leaves(nnx.grad(loss)(ref))):
        assert jnp.allclose(la, lb, atol=1e-3)


def test_dropless_impl_and_backend_are_validated():
    with pytest.raises(ValueError, match="dropless_impl"):
        MoELayer(8, 16, 4, routing="dropless", dropless_impl="cuda", rngs=nnx.Rngs(0))
    moe = MoELayer(8, 16, 4, routing="dropless", rngs=nnx.Rngs(0))
    x = jnp.ones((4, 8))
    with pytest.raises(ValueError, match="backend"):
        moe.experts.grouped_forward(x, jnp.array([1, 1, 1, 1], jnp.int32), backend="cuda")


def test_dropless_never_drops_even_when_every_token_picks_one_expert():
    moe = MoELayer(8, 16, 4, top_k=1, routing="dropless", rngs=nnx.Rngs(0))
    # Bias the router so all tokens pick expert 0.
    moe.router.kernel[...] = jnp.zeros_like(moe.router.kernel[...]).at[:, 0].set(10.0)
    x = jnp.abs(jax.random.normal(jax.random.key(5), (32, 8)))
    y, aux = moe(x, return_aux=True)
    assert float(aux.overflow_fraction) == 0.0
    ref = MoELayer(8, 16, 4, top_k=1, routing="dense", rngs=nnx.Rngs(0))
    ref.router.kernel[...] = moe.router.kernel[...]
    assert jnp.allclose(y, ref(x), atol=1e-5)


def test_capacity_routing_couples_tokens_but_dropless_does_not():
    """With capacity, a token's output depends on the other tokens in the call
    (they compete for slots); dropless keeps every token independent."""
    x = jax.random.normal(jax.random.key(6), (64, 8))
    diffs = {}
    for routing in ("capacity", "dropless"):
        moe = MoELayer(8, 16, 4, top_k=2, capacity_factor=0.5, routing=routing,
                       rngs=nnx.Rngs(0))
        diffs[routing] = float(jnp.max(jnp.abs(moe(x[:4]) - moe(x)[:4])))
    # Dropless differs only by matmul rounding across call sizes (GPU tiling).
    assert diffs["dropless"] < 1e-4, diffs
    assert diffs["capacity"] > 1e-2, diffs


def test_dropless_under_jit_and_bf16():
    moe = MoELayer(8, 16, 4, routing="dropless", dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(7), (10, 8)).astype(jnp.bfloat16)
    y = nnx.jit(lambda m, v: m(v))(moe, x)
    assert y.dtype == jnp.bfloat16 and bool(jnp.isfinite(y).all())


@pytest.mark.parametrize("routing", ["dropless", "capacity", "dense"])
def test_token_mask_keeps_padding_out_of_the_balance_stats(routing):
    """Identical pad tokens pile onto a few experts; masked, they must not
    move the balance loss or the load, and outputs must not change."""
    layer = MoELayer(16, 32, 4, top_k=2, routing=routing, capacity_factor=8.0,
                     rngs=nnx.Rngs(0))
    real = jax.random.normal(jax.random.key(0), (40, 16))
    pads = jnp.broadcast_to(jax.random.normal(jax.random.key(1), (1, 16)), (40, 16))
    x = jnp.concatenate([real, pads])
    mask = jnp.arange(80) < 40

    _, aux_real = layer(real, return_aux=True)
    y_all, aux_all = layer(x, return_aux=True)
    y_masked, aux_masked = layer(x, return_aux=True, token_mask=mask)

    np.testing.assert_allclose(y_masked, y_all, atol=1e-6)
    np.testing.assert_allclose(aux_masked.load_balance_loss, aux_real.load_balance_loss, rtol=1e-5)
    np.testing.assert_allclose(aux_masked.expert_utilization, aux_real.expert_utilization, atol=1e-6)
    np.testing.assert_allclose(aux_masked.router_probabilities, aux_real.router_probabilities, atol=1e-6)
    # Guard: unmasked, the pads do distort it.
    assert abs(float(aux_all.load_balance_loss) - float(aux_real.load_balance_loss)) > 1e-3
    # And the loss still trains the router through the masked mean.
    g = nnx.grad(lambda m: m(x, return_aux=True, token_mask=mask)[1].load_balance_loss)(layer)
    assert float(jnp.abs(g.router.kernel[...]).max()) > 0


def test_token_mask_broadcasts_over_leading_axes():
    layer = MoELayer(8, 16, 4, top_k=1, routing="dropless", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (2, 5, 8))
    mask = jnp.array([True, True, True, False, False])
    _, a = layer(x, return_aux=True, token_mask=mask)
    _, b = layer(x[:, :3], return_aux=True)
    # 1e-4: the router matmul runs at default (TF32) precision on GPU, and a
    # different token count changes its tiling (measured 2e-5).
    np.testing.assert_allclose(a.load_balance_loss, b.load_balance_loss, rtol=1e-4)


def test_shared_expert_adds_a_dense_swiglu_to_the_routed_output():
    moe = MoELayer(
        8, 4, 16, top_k=4, shared_hidden=32, routing="dense", rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.key(6), (2, 5, 8))
    routed = moe(x)
    shared = moe.shared
    moe.shared = None
    assert jnp.allclose(routed - moe(x), shared(x), atol=1e-5)


def test_fine_grained_dropless_with_shared_expert_matches_dense():
    """DeepSeekMoE shape: many narrow experts, top_k > 2, one wide shared MLP."""
    kw = dict(top_k=6, shared_hidden=32)
    moe = MoELayer(8, 4, 32, routing="dropless", rngs=nnx.Rngs(0), **kw)
    ref = MoELayer(8, 4, 32, routing="dense", rngs=nnx.Rngs(0), **kw)
    x = jax.random.normal(jax.random.key(7), (40, 8))

    assert jnp.allclose(moe(x), ref(x), atol=1e-5)

    def loss(m):
        return jnp.sum(m(x) ** 2)

    ga, gb = nnx.grad(loss)(moe), nnx.grad(loss)(ref)
    assert jnp.any(ga["shared"]["down"]["kernel"][...] != 0)
    for la, lb in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=True):
        assert jnp.allclose(la, lb, atol=1e-4)
