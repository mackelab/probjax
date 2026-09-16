import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM
from probjax.nn.generative.diffusion.config import (
    BaseNoiseSchedule,
    CosineNoiseSchedule,
    EDMNoiseSchedule,
    VENoiseSchedule,
    VPNoiseSchedule,
)
from probjax.nn.generative.discrete.config import MultinomialDiffusionSchedule


@pytest.mark.parametrize(
    'cls', [EDMNoiseSchedule, VENoiseSchedule, VPNoiseSchedule, CosineNoiseSchedule]
)
def test_schedule_inverse_and_forward_sde_identity(cls):
    schedule = cls(t_min=0.2, t_max=2.7)
    t = jnp.linspace(0.201, 2.699, 101)
    sigma = schedule.sigma_eff(t)
    assert jnp.all(jnp.diff(sigma) > 0)
    np.testing.assert_allclose(schedule.inv_sigma_eff(sigma), t, atol=2e-6)
    scale = schedule.scale(t)
    variance = schedule.std(t) ** 2
    drift = schedule.drift(t, jnp.ones_like(t))
    diffusion = schedule.diffusion(t, jnp.ones_like(t))
    scale_dt = jax.jvp(schedule.scale, (t,), (jnp.ones_like(t),))[1]
    variance_dt = jax.jvp(lambda t: schedule.std(t) ** 2, (t,), (jnp.ones_like(t),))[1]
    np.testing.assert_allclose(drift, scale_dt / scale, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(
        diffusion**2, variance_dt - 2 * drift * variance, rtol=3e-5, atol=2e-6
    )


def test_generic_scaled_schedule_diffusion():
    class Schedule(BaseNoiseSchedule):
        def scale(self, t):
            return jnp.exp(-t)

        def std(self, t):
            return jnp.sqrt(-jnp.expm1(-2 * t))

    schedule = Schedule()
    np.testing.assert_allclose(
        schedule.diffusion(jnp.array(0.3), jnp.ones(3)), jnp.sqrt(2), rtol=1e-6
    )


@pytest.mark.parametrize(
    'schedule',
    [VPNoiseSchedule(min_tau=0), CosineNoiseSchedule(), CosineNoiseSchedule(s=0)],
)
def test_float32_low_noise_round_trip(schedule):
    t = jnp.geomspace(1e-7, 0.01, 50)
    np.testing.assert_allclose(
        schedule.inv_sigma_eff(schedule.sigma_eff(t)), t, rtol=3e-5, atol=1e-9
    )
    assert jnp.isfinite(schedule.diffusion(jnp.array(1.0), jnp.ones(2))).all()


class Net(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array(0.1))

    def __call__(self, t, x, **kwargs):
        return self.weight[...] * x


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
@pytest.mark.parametrize('target', ['x0', 'eps', 'v'])
def test_presets_train_and_sample(cls, target):
    model = cls(Net(), event_spec=2, num_steps=16)
    model.loss_type = target
    loss, grads = nnx.value_and_grad(
        lambda m: m.loss(jax.random.key(0), jnp.ones((8, 2)))
    )(model)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(grads))
    times = model.solver_cfg.solve_schedule(
        model.train_cfg.t_min, model.train_cfg.t_max
    )
    assert jnp.all(jnp.diff(times) < 0)
    for mode in ['ode', 'sde']:
        values = model.as_dist(mode=mode).sample(jax.random.key(1), (16,))
        assert jnp.isfinite(values).all()


@pytest.mark.parametrize('kind', ['linear', 'cosine', 'sigmoid', 'logsnr'])
def test_categorical_schedule_probabilities(kind):
    schedule = getattr(MultinomialDiffusionSchedule, 'from_' + kind)(32, 4)
    t = jnp.linspace(0.0, 1.0, 64)[:, None]
    alpha = schedule.alpha_bar(t)
    assert jnp.all(jnp.diff(alpha[:, 0]) <= 0)
    probs = schedule.q_xt_given_x0_probs(jnp.zeros((64, 3), dtype=jnp.int32), t)
    assert jnp.isfinite(probs).all() and jnp.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=2e-6)


class ZeroNet(nnx.Module):
    def __call__(self, t, x, **kwargs):
        return jnp.zeros_like(x)


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
def test_gaussian_oracle_preconditioning(cls):
    model = cls(ZeroNet(), event_spec=2)
    t = jnp.linspace(model.train_cfg.t_min, model.train_cfg.t_max, 64)[:, None]
    a, sigma = model.scale_fn(t), model.std_fn(t)
    x = jnp.ones((64, 2))
    var0 = model.std0.get_value() ** 2
    expected = a * var0 / (a * a * var0 + sigma * sigma) * x
    np.testing.assert_allclose(model.denoise(t, x), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
@pytest.mark.parametrize('mode', ['ode', 'sde'])
def test_gaussian_oracle_sampling(cls, mode):
    model = cls(ZeroNet(), event_spec=1, num_steps=256)
    values = model.as_dist(mode=mode).sample(jax.random.key(42), (4096,))
    expected_std = model.marginal_std(model.train_cfg.t_min)
    assert abs(float(values.mean())) < 0.06 * float(jnp.squeeze(expected_std))
    np.testing.assert_allclose(values.std(), jnp.squeeze(expected_std), rtol=0.08)


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
def test_preset_sde_has_nonzero_noise(cls):
    model = cls(ZeroNet(), event_spec=2)
    _, diffusion = model.solver_cfg.build_sde_drift_and_diffusion(model)
    t = (model.train_cfg.t_min + model.train_cfg.t_max) / 2
    assert jnp.all(diffusion(t, jnp.ones(2)) > 0)
