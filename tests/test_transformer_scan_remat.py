"""Opt-in layer scanning and activation checkpointing of Transformer stacks."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import DiffusionTransformer, Transformer
from probjax.nn.pallas_kernels import CausalMask

DIMS = dict(model_dim=32, num_heads=2, num_layers=3, attn_size=16)


def _rel(a, b):
    a, b = jnp.ravel(a), jnp.ravel(b)
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-12))


def _call_kwargs(config, x):
    kwargs = {}
    if "context_dim" in config:
        kwargs["context"] = jnp.linspace(-1, 1, 2 * 8).reshape(2, 8)
    if config.get("enable_cross_attention"):
        kwargs["k"] = x[:, :5]
    if config.get("dropout_rate"):
        kwargs.update(deterministic=False, rng=jax.random.key(3))
    return kwargs


def _grads(model, x, kwargs):
    def loss(model, x):
        return jnp.sum(jnp.sin(model(x, **kwargs)))

    return nnx.value_and_grad(loss, argnums=(0, 1))(model, x)


CONFIGS = {
    "plain": {},
    "context": dict(context_dim=8),
    "cross": dict(enable_cross_attention=True),
    "dropout": dict(dropout_rate=0.1, drop_path_rate=0.1),
}


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
@pytest.mark.parametrize("policy", ["full", "dots_saveable"])
def test_remat_matches_unrolled(config, policy):
    x = jax.random.normal(jax.random.key(1), (2, 7, DIMS["model_dim"]))
    base = Transformer(**DIMS, rngs=nnx.Rngs(0), **config)
    remat = Transformer(**DIMS, rngs=nnx.Rngs(0), remat_policy=policy, **config)
    # Same parameter tree and values: remat changes only the backward pass.
    assert jax.tree.structure(nnx.state(base)) == jax.tree.structure(nnx.state(remat))
    kwargs = _call_kwargs(config, x)
    (loss_b, (gm_b, gx_b)) = _grads(base, x, kwargs)
    (loss_r, (gm_r, gx_r)) = _grads(remat, x, kwargs)
    assert _rel(loss_r, loss_b) < 1e-6
    assert _rel(gx_r, gx_b) < 1e-5
    flat_b = dict(nnx.to_flat_state(nnx.state(gm_b, nnx.Param)))
    flat_r = dict(nnx.to_flat_state(nnx.state(gm_r, nnx.Param)))
    for path, value in flat_b.items():
        # Key biases have mathematically zero gradient (softmax shift).
        if path[-2:] == ("key", "bias"):
            continue
        assert _rel(flat_r[path][...], value[...]) < 1e-5, path


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
@pytest.mark.parametrize("policy", [None, "full"])
def test_scan_layers_matches_unrolled(config, policy):
    x = jax.random.normal(jax.random.key(1), (2, 7, DIMS["model_dim"]))
    base = Transformer(**DIMS, rngs=nnx.Rngs(0), **config)
    scanned = Transformer(
        **DIMS, rngs=nnx.Rngs(5), scan_layers=True, remat_policy=policy, **config
    )
    leaves = jax.tree.leaves(nnx.state(scanned.layers, nnx.Param))
    assert all(leaf.shape[0] == DIMS["num_layers"] for leaf in leaves)
    scanned.load_unrolled(base)

    kwargs = _call_kwargs(config, x)
    (loss_b, (gm_b, gx_b)) = _grads(base, x, kwargs)
    (loss_s, (gm_s, gx_s)) = _grads(scanned, x, kwargs)
    assert _rel(loss_s, loss_b) < 1e-5
    assert _rel(gx_s, gx_b) < 1e-4
    # Bring the unrolled gradients into the stacked layout and compare.
    grad_base = nnx.merge(nnx.graphdef(base), gm_b, nnx.state(base, nnx.Not(nnx.Param)))
    grad_scanned = Transformer(
        **DIMS, rngs=nnx.Rngs(5), scan_layers=True, remat_policy=policy, **config
    )
    grad_scanned.load_unrolled(grad_base)
    flat_ref = dict(nnx.to_flat_state(nnx.state(grad_scanned, nnx.Param)))
    flat_s = dict(nnx.to_flat_state(nnx.state(gm_s, nnx.Param)))
    assert flat_ref.keys() == flat_s.keys()
    for path, value in flat_ref.items():
        if path[-2:] == ("key", "bias"):
            continue
        assert _rel(flat_s[path][...], value[...]) < 1e-4, path


@pytest.mark.parametrize(
    "variant",
    [{}, dict(scan_layers=True), dict(scan_layers=True, remat_policy="full")],
)
def test_decode_cache_matches_causal_forward(variant):
    x = jax.random.normal(jax.random.key(1), (2, 6, DIMS["model_dim"]))
    model = Transformer(**DIMS, rngs=nnx.Rngs(0), **variant)
    full = model(x, mask=CausalMask())
    model.init_cache(x.shape, dtype=x.dtype)
    steps = [model(x[:, t : t + 1], decode=True) for t in range(x.shape[1])]
    assert jnp.allclose(jnp.concatenate(steps, axis=1), full, atol=1e-5)


@pytest.mark.parametrize("variant", [dict(scan_layers=True), dict(remat_policy="full")])
def test_stored_rng_dropout_under_jit(variant):
    model = Transformer(**DIMS, dropout_rate=0.2, rngs=nnx.Rngs(0), **variant)
    x = jax.random.normal(jax.random.key(4), (2, 5, DIMS["model_dim"]))
    call = nnx.jit(lambda m, x: m(x, deterministic=False))
    first, second = call(model, x), call(model, x)
    assert jnp.isfinite(first).all()
    assert not jnp.allclose(first, second)  # the stored streams advance
    det = model(x, deterministic=True)
    assert jnp.allclose(det, model(x, deterministic=True))


def test_diffusion_transformer_scan_and_remat():
    kwargs = dict(
        model_dim=32, num_heads=2, num_layers=2, attn_size=16, time_embed_dim=8,
        fourier_dim=8,
    )  # fmt: skip
    base = DiffusionTransformer(3, rngs=nnx.Rngs(0), **kwargs)
    scanned = DiffusionTransformer(
        3, rngs=nnx.Rngs(1), scan_layers=True, remat_policy="full", **kwargs
    )
    scanned.load_unrolled(base)
    t, x = jnp.full((2,), 0.3), jax.random.normal(jax.random.key(2), (2, 5, 3))

    def loss(model, x):
        return jnp.sum(model(t, x) ** 2)

    (lb, gb) = nnx.value_and_grad(loss, argnums=1)(base, x)
    (ls, gs) = nnx.value_and_grad(loss, argnums=1)(scanned, x)
    assert _rel(ls, lb) < 1e-5 and _rel(gs, gb) < 1e-4


def test_invalid_options():
    with pytest.raises(ValueError, match="remat_policy"):
        Transformer(**DIMS, remat_policy="bogus", rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="drop_path_rate"):
        Transformer(
            **DIMS, drop_path_rate=[0.0, 0.1, 0.2], scan_layers=True,
            rngs=nnx.Rngs(0),
        )  # fmt: skip
    unrolled = Transformer(**DIMS, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="load_unrolled"):
        unrolled.load_unrolled(unrolled)


def test_default_layout_unchanged():
    model = Transformer(**DIMS, rngs=nnx.Rngs(0))
    assert len(model.attention_blocks) == DIMS["num_layers"]
    assert not hasattr(model, "layers")
    assert model.remat_policy is None and not model.scan_layers
