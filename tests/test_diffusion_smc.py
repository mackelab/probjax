"""Twisted SMC along a reverse diffusion: generic guides, pretrained models, RDSMC."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.inference import Guide, diffusion_smc, run, score_guide
from probjax.inference.neural import diffusion_guide
from probjax.inference.smc.diffusion import _score_and_marginal
from probjax.nn.generative.diffusion import VP
from probjax.nn.generative.diffusion.config import BaseSolverConfig, VPNoiseSchedule


def _gaussian_potential(y):
    return lambda x: -0.5 * jnp.sum((x - y) ** 2)


def _guide(potential, drift_rate=0.0, diffusion=0.0, twist=True):
    """A linear reverse SDE whose base N(0, I) it preserves, with twist V(x).

    ``drift_rate=0, diffusion=0`` freezes the particles, which reduces the SMC
    to importance sampling from the prior with lookahead resampling.
    """
    potential = jax.vmap(potential)

    def sample_base(key, t, num_particles, params):
        return jax.random.normal(key, (num_particles, 1)), jnp.zeros(num_particles)

    def evaluate(key, t, particles, params):
        log_twist = potential(particles) if twist else jnp.zeros(particles.shape[0])
        return (
            log_twist,
            drift_rate * particles,
            diffusion * jnp.ones_like(particles),
            jnp.zeros_like(particles),
        )

    return Guide(sample_base, evaluate, lambda particles, params: potential(particles))


def _sample(key, guide, times, num_particles, **options):
    kernel = diffusion_smc(guide, times, **options)
    init_key, run_key = jax.random.split(key)
    state = kernel.init(init_key, num_particles)
    return run(
        run_key,
        kernel,
        state,
        None,
        len(times) - 1,
        collect=("ess", "resampled", "ancestors"),
    )


def _weighted_stats(state):
    weights = jnp.exp(state.log_weights)
    x = jnp.ravel(state.particles)
    mean = jnp.sum(weights * x)
    return mean, jnp.sum(weights * (x - mean) ** 2)


def test_gaussian_conjugacy_matches_analytic_posterior():
    y = 1.0
    result = _sample(
        jax.random.key(1), _guide(_gaussian_potential(y)), jnp.linspace(1, 0, 5), 4000
    )
    mean, var = _weighted_stats(result.state)
    assert mean == pytest.approx(y / 2, abs=0.05)
    assert var == pytest.approx(0.5, abs=0.08)
    expected_log_z = -(y**2) / 4 - 0.5 * jnp.log(2.0)
    assert float(result.state.log_evidence) == pytest.approx(
        float(expected_log_z), abs=0.08
    )


def test_ou_reverse_sde_preserves_the_base():
    guide = _guide(lambda x: 0.0 * jnp.sum(x), drift_rate=0.5, diffusion=1.0)
    result = _sample(jax.random.key(2), guide, jnp.linspace(1, 0, 16), 4000)
    mean, var = _weighted_stats(result.state)
    assert mean == pytest.approx(0.0, abs=0.06)
    assert var == pytest.approx(1.0, abs=0.12)
    assert not bool(jnp.any(result.trace["resampled"]))
    assert float(result.state.log_evidence) == pytest.approx(0.0, abs=1e-6)


def test_no_twist_is_consistent_but_never_resamples_early():
    guide = _guide(_gaussian_potential(1.0), twist=False)
    result = _sample(jax.random.key(5), guide, jnp.linspace(1, 0, 5), 4000)
    mean, _ = _weighted_stats(result.state)
    # The exact final weight keeps it consistent; without lookahead the
    # weights stay flat until then.
    assert mean == pytest.approx(0.5, abs=0.08)
    assert not bool(jnp.any(result.trace["resampled"][:-1]))


def test_shapes_and_steps_past_the_end_change_nothing():
    guide = _guide(_gaussian_potential(0.0))
    times = jnp.linspace(1, 0, 4)
    kernel = diffusion_smc(guide, times)
    state = kernel.init(jax.random.key(3), 128)
    result = run(jax.random.key(4), kernel, state, None, 3, collect=("ancestors",))
    assert result.state.particles.shape == (128, 1)
    assert result.trace["ancestors"].shape == (3, 128)
    assert float(jax.scipy.special.logsumexp(result.state.log_weights)) == (
        pytest.approx(0.0, abs=1e-5)
    )
    more = run(jax.random.key(5), kernel, result.state, None, 2)
    assert jnp.array_equal(more.state.particles, result.state.particles)
    assert more.state.log_evidence == result.state.log_evidence


def test_invalid_args_raise():
    guide = _guide(_gaussian_potential(0.0))
    with pytest.raises(ValueError, match="ess_threshold"):
        diffusion_smc(guide, jnp.linspace(1, 0, 4), ess_threshold=1.5)
    with pytest.raises(ValueError, match="two points"):
        diffusion_smc(guide, jnp.ones(1))


class _ShrinkNet(nnx.Module):
    def __call__(self, t, x):
        return jax.tree.map(lambda leaf: 0.1 * leaf, x)


def test_pretrained_model_is_steered_towards_the_potential():
    model = VP(_ShrinkNet(), event_spec=2, num_steps=16)
    target = 1.0
    guide, params = diffusion_guide(model, _gaussian_potential(target))
    times = model.solver_cfg.solve_schedule(
        model.train_cfg.t_min, model.train_cfg.t_max, 16
    )
    kernel = diffusion_smc(guide, times)
    state = kernel.init(jax.random.key(0), 1000, params)
    result = run(jax.random.key(1), kernel, state, params, len(times) - 1)

    weights = jnp.exp(result.state.log_weights)[:, None]
    mean = jnp.sum(weights * result.state.particles, axis=0)
    assert mean.shape == (2,)
    # The tilt pulls the population towards the potential's centre.
    assert jnp.all(mean > 0.2)
    assert jnp.isfinite(result.state.log_evidence)


def test_pretrained_model_runs_under_jit_and_vmap():
    model = VP(_ShrinkNet(), event_spec=2, num_steps=8)
    guide, params = diffusion_guide(model, _gaussian_potential(1.0))
    times = model.solver_cfg.solve_schedule(
        model.train_cfg.t_min, model.train_cfg.t_max, 8
    )
    kernel = diffusion_smc(guide, times)

    def evidence(key):
        init_key, run_key = jax.random.split(key)
        state = kernel.init(init_key, 64, params)
        return run(run_key, kernel, state, params, len(times) - 1).state.log_evidence

    estimates = jax.jit(jax.vmap(evidence))(jax.random.split(jax.random.key(0), 3))
    assert estimates.shape == (3,)
    assert jnp.all(jnp.isfinite(estimates))


def _standard_normal_logpdf(x):
    return -0.5 * jnp.sum(x**2) - 0.5 * jnp.log(2.0 * jnp.pi) * jnp.size(x)


def test_score_estimate_matches_gaussian_truth():
    # Target N(0, 1) with alpha=0.8, sigma=0.6 keeps the marginal N(0, 1).
    score, log_marginal = _score_and_marginal(
        jax.random.key(0),
        jnp.array([0.5]),
        _standard_normal_logpdf,
        jnp.asarray(0.8),
        jnp.asarray(0.6),
        4000,
        None,
    )
    assert score[0] == pytest.approx(-0.5, abs=0.06)
    assert float(log_marginal) == pytest.approx(
        float(_standard_normal_logpdf(jnp.array([0.5]))), abs=0.06
    )


def _rdsmc(key, log_target, num_particles, num_steps, num_inner, shape=(1,)):
    schedule = VPNoiseSchedule(t_min=1e-3, t_max=1.0)
    times = BaseSolverConfig(schedule=schedule, num_steps=num_steps).solve_schedule(
        t_min=1e-3, t_max=1.0
    )
    guide = score_guide(log_target, schedule, shape, num_inner=num_inner)
    return _sample(key, guide, times, num_particles)


def test_rdsmc_gaussian_target_is_not_overconcentrated():
    # Weighting by the twist ratio alone, without the forward kernel, halves
    # this variance and biases log Z by about +0.5.
    def log_target(x):
        return _standard_normal_logpdf(x - 2.0)

    def one_run(key):
        result = _rdsmc(key, log_target, 256, 16, 64)
        mean, var = _weighted_stats(result.state)
        return mean, var, result.state.log_evidence

    mean, var, log_z = jax.jit(jax.vmap(one_run))(
        jax.random.split(jax.random.key(0), 8)
    )
    assert float(mean.mean()) == pytest.approx(2.0, abs=0.1)
    assert float(var.mean()) == pytest.approx(1.0, abs=0.1)
    assert float(log_z.mean()) == pytest.approx(0.0, abs=0.1)


def test_rdsmc_finds_both_modes():
    def log_target(x):
        component = lambda m: jax.scipy.stats.norm.logpdf(x[0], m, 0.5)
        return jnp.logaddexp(component(-2.0), component(2.0)) - jnp.log(2.0)

    result = _rdsmc(jax.random.key(7), log_target, 300, 10, 48)
    weights = jnp.exp(result.state.log_weights)
    left = float(jnp.sum(weights[result.state.particles[:, 0] < 0]))
    assert 0.25 < left < 0.75
    assert float(result.state.log_evidence) == pytest.approx(0.0, abs=0.5)
