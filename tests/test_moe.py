import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.moe import (
    ExpertSwiGLU,
    LocalDispatcher,
    MoELayer,
    TopKRouter,
    load_balance_loss,
)


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


def test_routing_weights_sum_to_one():
    router = TopKRouter(8, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (10, 8))
    out = router(x, top_k=2)
    assert out.expert_ids.shape == (10, 2)
    assert out.route_weights.shape == (10, 2)
    assert jnp.allclose(jnp.sum(out.route_weights, axis=-1), jnp.ones((10,)), atol=1e-6)
    assert jnp.allclose(jnp.sum(out.router_probs, axis=-1), jnp.ones((10,)), atol=1e-6)


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
    w_gate = moe.experts.w_gate[...]
    w_up = moe.experts.w_up[...]
    w_down = moe.experts.w_down[...]
    logits = x @ moe.router.w_router[...]
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
    # Share parameters between the two instances.
    dense.experts.w_gate[...] = sparse.experts.w_gate[...]
    dense.experts.w_up[...] = sparse.experts.w_up[...]
    dense.experts.w_down[...] = sparse.experts.w_down[...]
    dense.router.w_router[...] = sparse.router.w_router[...]

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


def test_expert_contract():
    experts = ExpertSwiGLU(3, d_model=8, d_hidden=16, rngs=nnx.Rngs(0))
    packed = jax.random.normal(jax.random.key(9), (3, 5, 8))
    assert experts(packed).shape == (3, 5, 8)
    with pytest.raises(ValueError):
        experts(jnp.ones((2, 5, 8)))  # wrong expert axis


def test_dispatcher_roundtrip_single_expert():
    disp = LocalDispatcher(num_experts=1, capacity_factor=1.0)
    x = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    expert_ids = jnp.zeros((3, 2), dtype=jnp.int32)
    weights = jnp.full((3, 2), 0.5)
    routed = disp.dispatch(x, expert_ids, weights)
    assert routed.tokens.shape == (1, 6, 2)
    # Identity experts: combine should recover x (0.5 + 0.5 per token).
    y = disp.combine(routed.tokens, routed)
    assert jnp.allclose(y, x, atol=1e-5)


def test_load_balance_loss_uniform():
    probs = jnp.full((10, 4), 0.25)
    ids = jnp.tile(jnp.arange(4)[None, :], (10, 1))[:, :2]
    loss = load_balance_loss(probs, ids, 4)
    assert loss == pytest.approx(1.0)
