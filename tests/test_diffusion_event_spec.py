"""Default diffusion event metadata must not freeze supported sampling shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM, DiffusionDenoiser
from probjax.nn.generative.diffusion.config import (
    EDMNoiseSchedule,
    EDMPreconditioning,
    EDMTrainingConfig,
)
from probjax.nn.generative.discrete import (
    MultinomialCosineDM,
    MultinomialDiffusion,
    MultinomialDiffusionSchedule,
    MultinomialLogSNRDM,
)


class Net(nnx.Module):
    def __call__(self, t, x):
        return jax.tree.map(lambda leaf: leaf * 0.1, x)


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
def test_required_default_and_sampling_override(cls):
    with pytest.raises(TypeError, match='event_spec'):
        cls(Net())
    model = cls(Net(), event_spec=3, num_steps=3)
    assert model.event_shape == (3,)
    assert model.as_dist().event_shape == (3,)
    override = model.as_dist((2, 4), num_steps=3)
    assert override.sample(jax.random.key(0), (2,)).shape == (2, 2, 4)
    assert model.event_shape == (3,)


def test_composable_diffusion_requires_spec():
    args = (Net(), EDMNoiseSchedule(), EDMPreconditioning(), EDMTrainingConfig())
    with pytest.raises(TypeError, match='event_spec'):
        DiffusionDenoiser(*args)
    assert DiffusionDenoiser(*args, event_spec=(2,)).event_shape == (2,)


@pytest.mark.parametrize('mode', ['ode', 'sde'])
def test_default_batch_polymorphism_reuses_export(mode):
    model = EDM(Net(), event_spec=(3,), num_steps=3)
    dist = model.as_dist(mode=mode)
    assert dist.sample(jax.random.key(0), (2,)).shape == (2, 3)
    exported = dist._get_sampler().exported
    assert dist.sample(jax.random.key(1), (2, 4)).shape == (2, 4, 3)
    assert dist._get_sampler().exported is exported


def test_default_change_invalidates_export_but_preserves_bound_views():
    model = EDM(Net(), event_spec=2, num_steps=3)
    bound = model.as_dist()
    bound.sample(jax.random.key(0), (1,))
    sampler = bound._get_sampler()
    model.event_spec = (4,)
    assert sampler.invalidated
    assert model.event_shape == (4,)
    assert bound.event_shape == (2,)
    assert model.as_dist().sample(jax.random.key(1), (1,)).shape == (1, 4)
    assert bound.sample(jax.random.key(2), (1,)).shape == (1, 2)


def test_pytree_default_and_override_do_not_mutate_caller_metadata():
    spec = {'a': 2, 'b': (3,)}
    model = EDM(Net(), event_spec=spec, num_steps=3)
    spec['a'] = 99
    assert model.event_shape == {'a': (2,), 'b': (3,)}
    sample = model.as_dist().sample(jax.random.key(0), (2,))
    assert sample['a'].shape == (2, 2)
    override = model.as_dist({'a': (4,), 'b': (1,)})
    assert override.sample(jax.random.key(1), (2,))['a'].shape == (2, 4)
    exposed = model.event_spec
    exposed['a'] = None
    assert model.event_shape['a'] == (2,)


def test_default_dtype_can_be_overridden_for_a_distribution():
    with jax.enable_x64():
        model = EDM(
            Net(), event_spec=jax.ShapeDtypeStruct((2,), jnp.float32), num_steps=3
        )
        dist = model.as_dist(dtype=jnp.float64)
        assert dist.event_spec.dtype == jnp.float64
        assert model.event_spec.dtype == jnp.float32
        assert dist.sample(jax.random.key(0), (2,)).dtype == jnp.float64


@pytest.mark.parametrize('spec', [None, 0, -2, (3, 0), (None, 3), {}, True])
def test_invalid_specs_fail_at_construction(spec):
    with pytest.raises((ValueError, TypeError)):
        EDM(Net(), event_spec=spec)


@pytest.mark.parametrize('cls', [MultinomialCosineDM, MultinomialLogSNRDM])
def test_discrete_presets_require_default_integer_event(cls):
    with pytest.raises(TypeError, match='event_spec'):
        cls(Net(), num_classes=4)
    model = cls(Net(), num_classes=4, event_spec=3, num_steps=3)
    assert model.event_shape == (3,)
    assert model.as_dist().event_spec.dtype == jnp.int32
    sample = model.as_dist((5,)).sample(jax.random.key(0), (2,))
    assert sample.shape == (2, 5)
    assert jnp.issubdtype(sample.dtype, jnp.integer)
    assert np.all(np.asarray(sample) < 4)


def test_discrete_base_rejects_structured_or_float_defaults():
    schedule = MultinomialDiffusionSchedule.from_cosine(num_steps=3, num_classes=4)
    with pytest.raises(TypeError, match='event_spec'):
        MultinomialDiffusion(Net(), schedule)
    with pytest.raises(TypeError, match='single-array'):
        MultinomialDiffusion(Net(), schedule, event_spec={'a': (3,)})
    with pytest.raises(TypeError, match='integer'):
        MultinomialDiffusion(
            Net(), schedule, event_spec=jax.ShapeDtypeStruct((3,), jnp.float32)
        )


def test_default_metadata_is_static_and_does_not_restrict_jitted_forward():
    model = EDM(Net(), event_spec=3, num_steps=3)
    call = nnx.jit(lambda model, x: model(0.5, x))
    assert call(model, jnp.ones((2, 3))).shape == (2, 3)
    assert call(model, jnp.ones((2, 7))).shape == (2, 7)
    model.set_event_spec((7,))
    assert call(model, jnp.ones((2, 7))).shape == (2, 7)
