"""Tests for reverse-time twisted SMC with a frozen diffusion model."""

import jax
import jax.numpy as jnp
import pytest

from probjax.inference.smc.pretrained_twist import run_pretrained_diffusion_smc
from probjax.nn.generative.diffusion.twists import (
    DenoisedPotentialTwist,
    NullTwist,
)


class _FakeSolverCfg:
    num_steps = 6

    def solve_schedule(self, t_min, t_max, num_steps=None):
        n = num_steps or self.num_steps
        return jnp.linspace(t_max, t_min, n)


class _FakeTrainCfg:
    t_min = 0.0
    t_max = 1.0


class _FakeIdentityModel:
    """Degenerate diffusion: reverse kernel is the identity, prior is N(0, I).

    SMC then reduces to exact importance sampling with lookahead
    resampling, so Gaussian conjugacy holds analytically.
    """

    solver_cfg = _FakeSolverCfg()
    train_cfg = _FakeTrainCfg()
    event_spec = None

    def _build_sde_drift_and_diffusion(self, inputs=None):
        def drift(t, x):
            return jax.tree.map(jnp.zeros_like, x)

        def diffusion(t, x):
            return jax.tree.map(jnp.zeros_like, x)

        return drift, diffusion

    def denoise(self, t, x, **kwargs):
        return x

    def _sample_base(self, rng, sample_shape, spec, t_max=None):
        leaves, treedef = jax.tree.flatten(spec)
        keys = jax.random.split(rng, len(leaves))
        return treedef.unflatten([
            jax.random.normal(k, sample_shape + leaf.shape, leaf.dtype)
            for k, leaf in zip(keys, leaves, strict=True)
        ])


class _FakeOUModel(_FakeIdentityModel):
    """OU reverse SDE with exact Gaussian score; stationary law is N(0, I)."""

    def _build_sde_drift_and_diffusion(self, inputs=None):
        def drift(t, x):
            return jax.tree.map(lambda xi: 0.5 * xi, x)

        def diffusion(t, x):
            return jax.tree.map(lambda xi: jnp.ones_like(xi), x)

        return drift, diffusion


def _gaussian_potential(y):
    y = jnp.asarray(y)

    def log_potential(x):
        return -0.5 * jnp.sum((x - y) ** 2)

    return log_potential


def _weighted_stats(particles, log_weights):
    w = jnp.exp(log_weights)
    while w.ndim < particles.ndim:
        w = w[..., None]
    mean = jnp.sum(w * particles)
    var = jnp.sum(w * (particles - mean) ** 2)
    return mean, var


def test_temperature_zero_recovers_prior():
    model = _FakeIdentityModel()
    result = run_pretrained_diffusion_smc(
        jax.random.key(0),
        model=model,
        event_spec=(1,),
        log_potential_fn=_gaussian_potential(5.0),
        num_particles=2000,
        temperature=0.0,
    )
    mean, var = _weighted_stats(result.state.particles, result.state.log_weights)
    assert mean == pytest.approx(0.0, abs=0.06)
    assert var == pytest.approx(1.0, abs=0.1)
    assert not bool(jnp.any(result.info.resampled))
    assert result.log_evidence == pytest.approx(0.0, abs=1e-6)


def test_gaussian_conjugacy_matches_analytic_posterior():
    y = 1.0
    model = _FakeIdentityModel()
    result = run_pretrained_diffusion_smc(
        jax.random.key(1),
        model=model,
        event_spec=(1,),
        log_potential_fn=_gaussian_potential(y),
        num_particles=4000,
        num_steps=5,
    )
    particles = jnp.ravel(result.state.particles)
    mean, var = _weighted_stats(particles, result.state.log_weights)
    assert mean == pytest.approx(y / 2, abs=0.05)
    assert var == pytest.approx(0.5, abs=0.08)
    expected_log_z = -(y**2) / 4 - 0.5 * jnp.log(2.0)
    assert float(result.log_evidence) == pytest.approx(float(expected_log_z), abs=0.08)


def test_ou_reverse_sde_preserves_prior():
    model = _FakeOUModel()
    result = run_pretrained_diffusion_smc(
        jax.random.key(2),
        model=model,
        event_spec=(1,),
        log_potential_fn=_gaussian_potential(0.0),
        num_particles=4000,
        num_steps=16,
        temperature=0.0,
    )
    mean, var = _weighted_stats(
        jnp.ravel(result.state.particles), result.state.log_weights
    )
    assert mean == pytest.approx(0.0, abs=0.06)
    assert var == pytest.approx(1.0, abs=0.12)


def test_info_shapes_and_normalized_weights():
    model = _FakeIdentityModel()
    n_particles, n_steps = 128, 4
    result = run_pretrained_diffusion_smc(
        jax.random.key(3),
        model=model,
        event_spec=(2,),
        log_potential_fn=_gaussian_potential(jnp.zeros(2)),
        num_particles=n_particles,
        num_steps=n_steps,
    )
    n_transitions = n_steps - 1
    assert result.state.particles.shape == (n_particles, 2)
    assert result.state.log_weights.shape == (n_particles,)
    assert float(
        jax.scipy.special.logsumexp(result.state.log_weights)
    ) == pytest.approx(0.0, abs=1e-5)
    assert result.info.ess.shape == (n_transitions,)
    assert result.info.resampled.shape == (n_transitions,)
    assert result.info.log_z_increments.shape == (n_transitions,)
    assert result.info.ancestors.shape == (n_transitions, n_particles)
    assert result.num_steps == n_transitions
    assert result.completed is True


def test_deterministic_under_fixed_key():
    kwargs = dict(
        model=_FakeIdentityModel(),
        event_spec=(1,),
        log_potential_fn=_gaussian_potential(1.0),
        num_particles=256,
    )
    first = run_pretrained_diffusion_smc(jax.random.key(4), **kwargs)
    second = run_pretrained_diffusion_smc(jax.random.key(4), **kwargs)
    assert jnp.array_equal(first.state.particles, second.state.particles)
    assert jnp.array_equal(first.log_evidence, second.log_evidence)


def test_null_twist_is_valid_but_less_efficient():
    kwargs = dict(
        model=_FakeIdentityModel(),
        event_spec=(1,),
        log_potential_fn=_gaussian_potential(1.0),
        num_particles=4000,
        num_steps=5,
        twist=NullTwist(),
    )
    result = run_pretrained_diffusion_smc(jax.random.key(5), **kwargs)
    mean, _ = _weighted_stats(
        jnp.ravel(result.state.particles), result.state.log_weights
    )
    # Still consistent (final weight is exact) but no lookahead resampling.
    assert mean == pytest.approx(0.5, abs=0.08)
    assert not bool(jnp.any(result.info.resampled))


def test_twist_values_match_denoised_potential():
    model = _FakeIdentityModel()
    twist = DenoisedPotentialTwist(
        denoise_fn=model.denoise,
        log_potential_fn=_gaussian_potential(2.0),
        temperature=2.0,
    )
    particles = jnp.array([[1.0], [3.0]])
    values = twist.log_twist(jnp.asarray(0.5), particles)
    assert values.shape == (2,)
    assert jnp.allclose(values, 2.0 * jnp.array([-0.5, -0.5]))


def test_invalid_args_raise():
    model = _FakeIdentityModel()
    with pytest.raises(ValueError, match="num_particles"):
        run_pretrained_diffusion_smc(
            jax.random.key(0),
            model=model,
            event_spec=(1,),
            log_potential_fn=_gaussian_potential(0.0),
            num_particles=0,
        )
    with pytest.raises(ValueError, match="ess_threshold"):
        run_pretrained_diffusion_smc(
            jax.random.key(0),
            model=model,
            event_spec=(1,),
            log_potential_fn=_gaussian_potential(0.0),
            ess_threshold=1.5,
        )
