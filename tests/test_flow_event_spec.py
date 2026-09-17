"""Flow defaults remain overridable without restricting network input shapes."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import LinearFlow, LinearMeanFlow


class Net(nnx.Module):
    def __call__(self, t, x, r=None, **kwargs):
        return jax.tree.map(lambda leaf: leaf * 0.1, x)


@pytest.mark.parametrize("cls", [LinearFlow, LinearMeanFlow])
def test_required_default_and_polymorphic_sampling(cls):
    with pytest.raises(TypeError, match="event_spec"):
        cls(Net())
    model = cls(Net(), event_spec=3)
    dist = model.as_dist(num_steps=3)
    assert dist.sample(jax.random.key(0), (2,)).shape == (2, 3)
    exported = dist._get_sampler().exported
    assert dist.sample(jax.random.key(1), (5,)).shape == (5, 3)
    assert dist._get_sampler().exported is exported
    override = model.as_dist((2, 4), num_steps=3)
    assert override.sample(jax.random.key(2), (2,)).shape == (2, 2, 4)
    assert model.event_shape == (3,)
    call = nnx.jit(lambda m, x: m(0.5, x))
    assert call(model, jnp.ones((2, 7))).shape == (2, 7)


@pytest.mark.parametrize("cls", [LinearFlow, LinearMeanFlow])
@pytest.mark.parametrize(
    "spec", [None, 0, (3, 0), {}, jax.ShapeDtypeStruct((2,), jnp.int32)]
)
def test_invalid_defaults(cls, spec):
    with pytest.raises((TypeError, ValueError)):
        cls(Net(), event_spec=spec)


@pytest.mark.parametrize('cls', [LinearFlow, LinearMeanFlow])
def test_default_change_invalidates_export_but_preserves_bound_views(cls):
    model = cls(Net(), event_spec=2)
    bound = model.as_dist(num_steps=3)
    bound.sample(jax.random.key(0), (1,))
    sampler = bound._get_sampler()
    model.event_spec = (4,)
    assert sampler.invalidated
    assert model.event_shape == (4,)
    assert bound.event_shape == (2,)
    assert model.as_dist(num_steps=3).sample(jax.random.key(1), (1,)).shape == (1, 4)
    assert bound.sample(jax.random.key(2), (1,)).shape == (1, 2)


@pytest.mark.parametrize('cls', [LinearFlow, LinearMeanFlow])
def test_pytree_default_and_override_do_not_mutate_caller_metadata(cls):
    spec = {'a': 2, 'b': (3,)}
    model = cls(Net(), event_spec=spec)
    spec['a'] = 99
    assert model.event_shape == {'a': (2,), 'b': (3,)}
    sample = model.as_dist(num_steps=3).sample(jax.random.key(0), (2,))
    assert sample['a'].shape == (2, 2)
    override = model.as_dist({'a': (4,), 'b': (1,)})
    assert override.sample(jax.random.key(1), (2,))['a'].shape == (2, 4)
    exposed = model.event_spec
    exposed['a'] = None
    assert model.event_shape['a'] == (2,)


@pytest.mark.parametrize('cls', [LinearFlow, LinearMeanFlow])
def test_default_dtype_can_be_overridden_for_a_distribution(cls):
    with jax.enable_x64():
        model = cls(Net(), event_spec=jax.ShapeDtypeStruct((2,), jnp.float32))
        dist = model.as_dist(dtype=jnp.float64)
        assert dist.event_spec.dtype == jnp.float64
        assert model.event_spec.dtype == jnp.float32
        assert dist.sample(jax.random.key(0), (2,)).dtype == jnp.float64


@pytest.mark.parametrize('cls', [LinearFlow, LinearMeanFlow])
def test_composable_base_requires_default(cls):
    preset = cls(Net(), event_spec=2)
    base = cls.__bases__[0]
    args = (Net(), preset.schedule, preset.preconditioning, preset.train_cfg)
    with pytest.raises(TypeError, match='event_spec'):
        base(*args)
    assert base(*args, event_spec=(2, 4)).event_shape == (2, 4)


@pytest.mark.parametrize('cls', [LinearFlow, LinearMeanFlow])
def test_invalid_override_does_not_change_default(cls):
    model = cls(Net(), event_spec=2)
    with pytest.raises(TypeError, match='floating'):
        model.as_dist(dtype=jnp.int32)
    with pytest.raises(ValueError):
        model.set_event_spec((0,))
    assert model.event_shape == (2,)
