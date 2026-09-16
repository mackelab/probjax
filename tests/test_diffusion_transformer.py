import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import EDM, DiffusionTransformer


def make(**kwargs):
    return DiffusionTransformer(
        3,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        attn_size=4,
        time_embed_dim=8,
        fourier_dim=8,
        rngs=nnx.Rngs(0),
        **kwargs,
    )


@pytest.mark.parametrize('shape', [(5, 3), (2, 5, 3), (2, 3, 5, 3)])
def test_jit_shapes_and_conditioning(shape):
    model = make(context_dim=4)
    call = nnx.jit(lambda m, t, x, c: m(t, x, context=c))
    batch = shape[:-2]
    result = call(model, jnp.ones(batch), jnp.ones(shape), jnp.ones(batch + (4,)))
    assert result.shape == shape
    assert jnp.isfinite(result).all()
    assert model(0.1, jnp.ones(shape), r=0.2, context=jnp.ones(4)).shape == shape


def test_diffusion_training_and_sampling_overrides():
    model = EDM(make(), event_spec=(5, 3), num_steps=3)
    loss, grad = nnx.value_and_grad(
        lambda m: m.loss(jax.random.key(0), jnp.ones((2, 5, 3)))
    )(model)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(grad))
    dist = model.as_dist(event_spec=(7, 3))
    assert dist.sample(jax.random.key(1), (2,)).shape == (2, 7, 3)


def test_factories_and_invalid_context():
    class Projection(nnx.Linear):
        pass

    class Custom(DiffusionTransformer):
        projection_cls = Projection
        position_cls = None

    model = Custom(
        3, model_dim=8, num_heads=2, num_layers=1, attn_size=4, rngs=nnx.Rngs(0)
    )
    assert isinstance(model.input_projection, Projection)
    assert model.position is None
    with pytest.raises(ValueError, match='context'):
        model(0.1, jnp.ones((2, 3)), context=jnp.ones(4))
    with pytest.raises(ValueError, match='context'):
        make(context_dim=4)(0.1, jnp.ones((2, 3)))


def test_conditioning_learns_and_mask_is_forwarded():
    model = make(context_dim=2, position_cls=None)
    x = jax.random.normal(jax.random.key(4), (2, 4, 3))
    c = jnp.ones((2, 2))
    grad = nnx.grad(lambda m: jnp.mean(m(0.2, x, context=c) ** 2))(model)
    params = nnx.state(model, nnx.Param)
    nnx.update(model, jax.tree.map(lambda p, g: p - 0.01 * g, params, grad))
    baseline = model(0.2, x, context=c)
    assert not jnp.allclose(baseline, model(0.8, x, context=c))
    assert not jnp.allclose(baseline, model(0.2, x, context=-c))
    mask = jnp.eye(4, dtype=bool)
    assert not jnp.allclose(baseline, model(0.2, x, context=c, mask=mask))
    perm = jnp.array([2, 0, 3, 1])
    assert jnp.allclose(model(0.2, x[:, perm], context=c), baseline[:, perm], atol=1e-6)


def test_explicit_factory_override_and_odd_width_positions():
    class Custom(DiffusionTransformer):
        projection_cls = lambda *args, **kwargs: None

    model = Custom(
        3,
        model_dim=7,
        num_heads=1,
        num_layers=1,
        attn_size=4,
        projection_cls=nnx.Linear,
        rngs=nnx.Rngs(0),
    )
    assert isinstance(model.input_projection, nnx.Linear)
    assert model(0.2, jnp.ones((4, 3))).shape == (4, 3)
