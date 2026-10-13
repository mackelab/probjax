"""Tests for training-free reverse-diffusion SMC (RDSMC)."""

import jax
import jax.numpy as jnp
import pytest

from probjax.inference.smc.rdsmc import run_rdsmc
from probjax.nn.generative.diffusion.score_estimation import is_score_and_marginal

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


def _standard_normal_logpdf(x):
    return -0.5 * jnp.sum(x**2) - 0.5 * _LOG_2PI * jnp.size(x)


def test_is_estimator_matches_gaussian_truth():
    # Target N(0, 1) with alpha=0.8, sigma=0.6 keeps the marginal N(0, 1).
    score, log_marginal, info = is_score_and_marginal(
        jax.random.key(0),
        jnp.array([0.5]),
        jnp.asarray(0.5),
        log_target_fn=_standard_normal_logpdf,
        alpha=jnp.asarray(0.8),
        sigma=jnp.asarray(0.6),
        num_inner=4000,
    )
    assert score[0] == pytest.approx(-0.5, abs=0.06)
    assert float(log_marginal) == pytest.approx(-0.5 * 0.25 - 0.5 * _LOG_2PI, abs=0.06)
    assert float(info.ess) > 2000


def test_is_estimator_rejects_degenerate_noise():
    with pytest.raises(ValueError, match="sigma must be positive"):
        is_score_and_marginal(
            jax.random.key(0),
            jnp.array([0.5]),
            jnp.asarray(0.0),
            log_target_fn=_standard_normal_logpdf,
            alpha=jnp.asarray(1.0),
            sigma=jnp.asarray(0.0),
            num_inner=8,
        )


def test_rdsmc_gaussian_target():
    def log_target(x):
        return -0.5 * jnp.sum((x - 2.0) ** 2) - 0.5 * _LOG_2PI

    result = run_rdsmc(
        jax.random.key(0),
        log_target_fn=log_target,
        event_spec=(1,),
        num_particles=200,
        num_steps=8,
        num_inner=32,
    )
    weights = jnp.exp(result.state.log_weights)
    flat = jnp.ravel(result.state.particles)
    mean = jnp.sum(weights * flat)
    var = jnp.sum(weights * (flat - mean) ** 2)
    assert float(mean) == pytest.approx(2.0, abs=0.25)
    assert float(var) == pytest.approx(1.0, abs=0.3)
    assert float(result.log_evidence) == pytest.approx(0.0, abs=0.5)


def test_rdsmc_finds_both_modes():
    def log_target(x):
        x0 = jnp.ravel(x)[0]
        comp = lambda m: jax.scipy.stats.norm.logpdf(x0, m, 0.5)
        return jnp.logaddexp(comp(-2.0), comp(2.0)) - jnp.log(2.0)

    result = run_rdsmc(
        jax.random.key(7),
        log_target_fn=log_target,
        event_spec=(1,),
        num_particles=300,
        num_steps=10,
        num_inner=48,
    )
    weights = jnp.exp(result.state.log_weights)
    flat = jnp.ravel(result.state.particles)
    frac_left = float(jnp.sum(weights[flat < 0]))
    assert 0.25 < frac_left < 0.75
    assert float(result.log_evidence) == pytest.approx(0.0, abs=0.5)


def test_rdsmc_info_shapes_and_normalized_weights():
    result = run_rdsmc(
        jax.random.key(3),
        log_target_fn=_standard_normal_logpdf,
        event_spec=(2,),
        num_particles=64,
        num_steps=5,
        num_inner=16,
    )
    assert result.state.particles.shape == (64, 2)
    assert float(
        jax.scipy.special.logsumexp(result.state.log_weights)
    ) == pytest.approx(0.0, abs=1e-5)
    assert result.info.ess.shape == (4,)
    assert result.info.inner_ess.shape == (5,)
    assert result.info.ancestors.shape == (4, 64)
    assert result.num_steps == 4
    assert result.completed is True


def test_rdsmc_deterministic_under_fixed_key():
    kwargs = dict(
        log_target_fn=_standard_normal_logpdf,
        event_spec=(1,),
        num_particles=64,
        num_steps=4,
        num_inner=16,
    )
    first = run_rdsmc(jax.random.key(4), **kwargs)
    second = run_rdsmc(jax.random.key(4), **kwargs)
    assert jnp.array_equal(first.state.particles, second.state.particles)
    assert jnp.array_equal(first.log_evidence, second.log_evidence)


def test_rdsmc_invalid_args_raise():
    with pytest.raises(ValueError, match="num_particles"):
        run_rdsmc(
            jax.random.key(0),
            log_target_fn=_standard_normal_logpdf,
            event_spec=(1,),
            num_particles=0,
        )
    with pytest.raises(ValueError, match="t_min"):
        run_rdsmc(
            jax.random.key(0),
            log_target_fn=_standard_normal_logpdf,
            event_spec=(1,),
            t_min=0.0,
        )
