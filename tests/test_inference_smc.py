"""Tempered SMC: schedules, population policies and adaptation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from probjax.inference import (
    acceptance_rate_adaptor,
    hmc,
    mh,
    particle_adaptor,
    run,
    smc,
)

KEY = jax.random.key(0)
OBSERVATION = jnp.array([1.5, -0.75])
NOISE = jnp.array([[0.4, 0.15], [0.15, 0.7]])
POSTERIOR_COV = jnp.linalg.inv(jnp.eye(2) + jnp.linalg.inv(NOISE))
POSTERIOR_MEAN = POSTERIOR_COV @ jnp.linalg.solve(NOISE, OBSERVATION)
LOG_EVIDENCE = multivariate_normal.logpdf(OBSERVATION, jnp.zeros(2), jnp.eye(2) + NOISE)


def logprior(x):
    return norm.logpdf(x).sum()


def loglikelihood(x):
    return multivariate_normal.logpdf(OBSERVATION, x, NOISE)


def _kernel(**options):
    options.setdefault("num_mcmc_steps", 3)
    return smc(logprior, loglikelihood, move=hmc(num_integration_steps=4), **options)


def _run(kernel, num_steps, *, args=None, adaptor=None, num_particles=1024):
    init_key, run_key = jax.random.split(KEY)
    state = kernel.init(init_key, jax.random.normal(init_key, (num_particles, 2)))
    params = kernel.init_params(state, step_size=0.3)
    return run(
        run_key,
        kernel,
        state,
        params,
        num_steps,
        args=args,
        adaptor=adaptor,
        collect=("temperature", "ess", "ancestors", "acceptance_rate"),
    )


def _assert_posterior(state, mean_tol=0.08, evidence_tol=0.08):
    np.testing.assert_allclose(state.particles.mean(0), POSTERIOR_MEAN, atol=mean_tol)
    np.testing.assert_allclose(jnp.cov(state.particles.T), POSTERIOR_COV, atol=mean_tol)
    assert float(state.log_evidence) == pytest.approx(
        float(LOG_EVIDENCE), abs=evidence_tol
    )


def test_fixed_schedule_recovers_posterior_and_evidence():
    schedule = jnp.linspace(0.0, 1.0, 21)[1:]
    result = _run(_kernel(), 20, args=(schedule,))
    np.testing.assert_allclose(result.trace["temperature"], schedule)
    assert jnp.all(result.trace["acceptance_rate"] > 0.3)
    _assert_posterior(result.state)


def test_adaptive_schedule_keeps_target_ess_and_stops_at_one():
    result = _run(_kernel(target_ess=0.7), 30)
    temperatures = result.trace["temperature"]
    assert float(temperatures[-1]) == 1.0
    finished = int(jnp.argmax(temperatures >= 1.0))
    assert 0 < finished < 10
    # Every step but the last holds the ESS at the target.
    np.testing.assert_allclose(result.trace["ess"][:finished], 0.7, atol=0.01)
    # Steps after the end leave the population alone.
    assert jnp.all(result.trace["ancestors"][finished + 1 :] == jnp.arange(1024))
    _assert_posterior(result.state)


def test_waste_free_keeps_every_chain_state():
    result = _run(_kernel(target_ess=0.7, waste_free=4), 30)
    assert result.state.particles.shape == (1024, 2)
    ancestors = result.trace["ancestors"][0]
    # Each resampled parent seeds one chain of four kept states.
    assert jnp.all(ancestors.reshape(256, 4) == ancestors[::4, None])
    # Fewer independent chains make the evidence noisier.
    _assert_posterior(result.state, evidence_tol=0.15)


def test_batched_moves_match_moving_all_at_once():
    schedule = jnp.array([0.3, 1.0])
    full = _run(_kernel(), 2, args=(schedule,), num_particles=64)
    batched = _run(_kernel(batch_size=16), 2, args=(schedule,), num_particles=64)
    np.testing.assert_allclose(full.state.particles, batched.state.particles, atol=1e-5)


def test_data_tempering_adds_one_observation_per_step():
    data = jnp.array([1.0, -0.5]) + 0.7 * jax.random.normal(KEY, (8, 2))

    def per_observation(x):
        return norm.logpdf(data, x, 0.7).sum(axis=1)

    kernel = smc(
        logprior, per_observation, move=hmc(num_integration_steps=4), num_mcmc_steps=3
    )
    state = kernel.init(
        KEY, jax.random.normal(KEY, (1024, 2)), temperature=jnp.zeros(8)
    )
    params = kernel.init_params(state, step_size=0.3)
    schedule = jnp.tril(jnp.ones((8, 8)))
    result = run(KEY, kernel, state, params, 8, args=(schedule,))

    variance = 1 / (1 + 8 / 0.7**2)
    np.testing.assert_allclose(
        result.state.particles.mean(0), variance * data.sum(0) / 0.7**2, atol=0.05
    )
    np.testing.assert_allclose(result.state.particles.var(0), variance, atol=0.02)
    # Each coordinate's observations are jointly N(0, 0.49 I + 1 1^T).
    marginal_cov = 0.49 * jnp.eye(8) + jnp.ones((8, 8))
    exact = jnp.sum(
        jax.vmap(lambda y: multivariate_normal.logpdf(y, jnp.zeros(8), marginal_cov))(
            data.T
        )
    )
    assert float(result.state.log_evidence) == pytest.approx(float(exact), abs=0.1)


def test_adaptive_tempering_needs_a_scalar_temperature():
    kernel = _kernel(target_ess=0.5)
    state = kernel.init(KEY, jnp.zeros((8, 2)), temperature=jnp.zeros(2))
    with pytest.raises(ValueError, match="scalar temperature"):
        kernel.step(KEY, state, kernel.init_params(state))


def test_default_adaptation_fits_the_move_to_the_population():
    kernel = _kernel(target_ess=0.7)
    result = _run(kernel, 30, adaptor=kernel.default_adaptation())
    np.testing.assert_allclose(
        result.params.inverse_mass_matrix, jnp.diag(POSTERIOR_COV), atol=0.05
    )
    # The moves accept nearly everything at first, so the step grows.
    initial = kernel.init_params(kernel.init(KEY, jnp.zeros((1, 2))), step_size=0.3)
    assert result.params.step_size > 1.5 * initial.step_size
    _assert_posterior(result.state)


def test_default_adaptation_needs_something_to_adapt():
    kernel = smc(logprior, loglikelihood, move=mh)
    with pytest.raises(ValueError, match="pass an adaptor explicitly"):
        kernel.default_adaptation()


def test_independent_runs_vectorise():
    kernel = _kernel(target_ess=0.7)

    def evidence(key):
        init_key, run_key = jax.random.split(key)
        state = kernel.init(init_key, jax.random.normal(init_key, (256, 2)))
        params = kernel.init_params(state, step_size=0.3)
        return run(run_key, kernel, state, params, 15).state.log_evidence

    estimates = jax.jit(jax.vmap(evidence))(jax.random.split(KEY, 4))
    np.testing.assert_allclose(estimates, LOG_EVIDENCE, atol=0.2)


class _Info(NamedTuple):
    acceptance_rate: jax.Array


def test_acceptance_rate_adaptor_scales_the_step():
    adaptor = acceptance_rate_adaptor(target=0.5)
    params = {"step_size": jnp.array(0.1)}
    _, grown, _ = adaptor.update(None, _Info(jnp.array(0.9)), (), params)
    _, kept, _ = adaptor.update(None, _Info(jnp.array(jnp.nan)), (), params)
    assert float(grown["step_size"]) == pytest.approx(0.1 * np.exp(0.4))
    assert float(kept["step_size"]) == pytest.approx(0.1)


class _State(NamedTuple):
    particles: jax.Array


def test_particle_adaptor_matches_the_parameter_shape():
    particles = jax.random.normal(KEY, (4096, 2)) * jnp.array([1.0, 3.0])
    adaptor = particle_adaptor()
    for current, expected in (
        (jnp.ones(2), jnp.array([1.0, 9.0])),
        (jnp.eye(2), jnp.diag(jnp.array([1.0, 9.0]))),
    ):
        _, params, _ = adaptor.update(
            _State(particles), None, (), {"inverse_mass_matrix": current}
        )
        np.testing.assert_allclose(params["inverse_mass_matrix"], expected, atol=0.4)
