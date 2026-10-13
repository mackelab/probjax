"""Contract for ``local_context``: shares every axis of ``x`` but the last."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.nets.diffusion_transformer import DiffusionTransformer
from probjax.nn.nets.transformer import Transformer

LOCAL_DIM = 3
KEY = jax.random.key(0)


def _call(net, x, local_context=None):
    """Nets differ only in whether time comes first."""
    kwargs = {} if local_context is None else {"local_context": local_context}
    if isinstance(net, DiffusionTransformer):
        return net(0.5, x, **kwargs)
    return net(x, **kwargs)


# name -> (builder taking local_context_dim, x shape)
NETS = {
    "transformer": (
        lambda dim: Transformer(16, 2, 2, 8, local_context_dim=dim, rngs=nnx.Rngs(0)),
        (2, 6, 16),
    ),
    "diffusion_transformer": (
        lambda dim: DiffusionTransformer(
            4,
            model_dim=16,
            num_heads=2,
            num_layers=2,
            attn_size=8,
            local_context_dim=dim,
            rngs=nnx.Rngs(0),
        ),
        (2, 6, 4),
    ),
}


def _randomize(net, key):
    """Break the identity initialisation of the fusion layers."""
    state = nnx.state(net, nnx.Param)
    leaves, tree = jax.tree.flatten(state)
    keys = jax.random.split(key, len(leaves))
    nnx.update(
        net,
        jax.tree.unflatten(
            tree,
            [
                leaf + 0.5 * jax.random.normal(k, leaf.shape)
                for leaf, k in zip(leaves, keys, strict=True)
            ],
        ),
    )


@pytest.fixture(params=sorted(NETS))
def case(request):
    build, x_shape = NETS[request.param]
    x = jax.random.normal(KEY, x_shape)
    local = jax.random.normal(jax.random.key(1), x_shape[:-1] + (LOCAL_DIM,))
    return build, x, local


def test_local_context_changes_the_output(case):
    build, x, local = case
    net = build(LOCAL_DIM)
    _randomize(net, jax.random.key(2))
    out = _call(net, x, local)
    assert out.shape == _call(build(None), x).shape
    assert jnp.all(jnp.isfinite(out))
    assert not jnp.allclose(out, _call(net, x, local + 1.0), atol=1e-5)


def test_default_is_unchanged_without_local_context(case):
    build, x, local = case
    out = _call(build(None), x)
    assert jnp.all(jnp.isfinite(out))
    with pytest.raises(ValueError, match="local_context_dim is None"):
        _call(build(None), x, local)


def test_configured_local_context_is_required(case):
    build, x, _ = case
    with pytest.raises(ValueError, match="local_context is required"):
        _call(build(LOCAL_DIM), x)


def test_local_context_must_match_x_except_last_axis(case):
    build, x, local = case
    net = build(LOCAL_DIM)
    with pytest.raises(ValueError, match="must have 3 features"):
        _call(net, x, local[..., :-1])
    with pytest.raises(ValueError, match="match x on all but the last axis"):
        _call(net, x, local[..., :-1, :] if local.ndim > 2 else local[:-1])
