"""Upstream parity and correctness checks for the shared SMC extensions."""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from jax.scipy.stats import norm

from probjax.inference import SMC, adaptive_smc_kernel, hmc, persistent_smc_kernel, smc
from probjax.inference.filtering.joint import sample_joint_paths
from probjax.inference.smc import PartialPosteriorsPath
from probjax.inference.smc.base import make_mcmc_adapter
from probjax.inference.smc.ports import pretuned_smc, tuned_smc, waste_free_strategy
from probjax.inference.smc.temporal import run_temporal_smc, temporal_smc
from probjax.inference.smc.temporal_utils import (
    population_diagnostics,
    summarize_replicates,
)
from tests.test_temporal_smc import (
    MODELS,
    THETA,
    TS,
    YS,
    assert_tree_close,
    exact_joint,
    gaussian_model,
    kalman_log_likelihood,
)

KEY = jax.random.key(31)
PARTICLES = jax.random.normal(KEY, (32, 2))
PRIOR = lambda x: norm.logpdf(x).sum()
LIKELIHOOD = lambda x: norm.logpdf(x, 1.0, 0.7).sum()
PARAMS = {'step_size': 0.15, 'inverse_mass_matrix': jnp.ones(2)}
BATCHED = {'step_size': jnp.array([0.15]), 'inverse_mass_matrix': jnp.ones((1, 2))}


def factory(batch_size=0, **kw):
    return smc(
        logprior_fn=PRIOR,
        loglikelihood_fn=LIKELIHOOD,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=3,
        batch_size=batch_size,
        **kw,
    )


def test_batching_matches_upstream_and_full_population():
    full, batched = factory(), factory(8)
    state = full.init(PARTICLES)
    a = jax.jit(full.step)(KEY, state, tempering_param=0.3, mcmc_parameters=BATCHED)
    b = jax.jit(batched.step)(KEY, state, tempering_param=0.3, mcmc_parameters=BATCHED)
    assert_tree_close(a, b, atol=2e-6)


def test_adaptive_path_guard():
    with pytest.raises(ValueError, match='geometric'):
        adaptive_smc_kernel(
            path=PartialPosteriorsPath(),
            logprior_fn=PRIOR,
            loglikelihood_fn=LIKELIHOOD,
            mcmc_kernel=hmc,
        )


def test_persistent_likelihoods_are_bound_to_instances():
    a = persistent_smc_kernel(
        logprior_fn=PRIOR, loglikelihood_fn=lambda x: jnp.array(-2.0), mcmc_kernel=hmc
    )
    b = persistent_smc_kernel(
        logprior_fn=PRIOR, loglikelihood_fn=lambda x: jnp.array(-7.0), mcmc_kernel=hmc
    )
    first, second = a.init(PARTICLES, n_schedule=3), b.init(PARTICLES, n_schedule=3)
    np.testing.assert_allclose(first.persistent_log_likelihoods[0], -2.0)
    np.testing.assert_allclose(second.persistent_log_likelihoods[0], -7.0)


def test_waste_free_is_upstream_strategy_and_preserves_population_size():
    strategy = waste_free_strategy(32, 4)
    kernel = smc(
        logprior_fn=PRIOR,
        loglikelihood_fn=LIKELIHOOD,
        mcmc_kernel=hmc,
        num_integration_steps=3,
        num_mcmc_steps=None,
        update_strategy=strategy,
    )
    state, info = jax.jit(kernel.step)(
        KEY, kernel.init(PARTICLES), tempering_param=0.2, mcmc_parameters=BATCHED
    )
    assert state.particles.shape == PARTICLES.shape
    assert jnp.isfinite(state.particles).all()
    assert info.ancestors.shape == (8,)
    with pytest.raises(ValueError):
        waste_free_strategy(31, 4)


def test_runner_evidence_without_history_and_chunk_continuation():
    kernel = smc(
        logprior_fn=PRIOR,
        loglikelihood_fn=lambda x: jnp.array(-2.0),
        mcmc_kernel=hmc,
        num_mcmc_steps=1,
        num_integration_steps=2,
    )
    state = kernel.init(PARTICLES)
    runner = SMC(kernel)
    schedule = jnp.array([0.2, 0.6, 1.0])
    whole = runner.run(KEY, state, schedule, BATCHED)
    assert whole.info is None and whole.final_info is not None
    np.testing.assert_allclose(whole.log_evidence, -2.0, atol=1e-6)
    first = runner.run(KEY, state, schedule[:1], BATCHED)
    last = runner.run(
        first.key,
        first.state,
        schedule[1:],
        BATCHED,
        initial_log_evidence=first.log_evidence,
    )
    assert_tree_close(whole.state, last.state, atol=2e-6)
    np.testing.assert_allclose(last.log_evidence, whole.log_evidence)
    assert whole.completed and whole.num_steps == 3
    empty = runner.run(KEY, state, schedule[:0], BATCHED)
    assert empty.num_steps == 0 and empty.final_info is None


def test_adaptive_runner_reaches_target_and_reports_cap():
    kernel = adaptive_smc_kernel(
        logprior_fn=PRIOR,
        loglikelihood_fn=LIKELIHOOD,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=3,
    )
    result = SMC(kernel).run_adaptive(KEY, kernel.init(PARTICLES), BATCHED)
    assert result.completed and result.state.tempering_param == 1.0
    assert jnp.isfinite(result.log_evidence)
    capped = SMC(kernel).run_adaptive(KEY, kernel.init(PARTICLES), BATCHED, max_steps=1)
    assert capped.num_steps == 1 and not capped.completed


def test_tuned_wrapper_matches_blackjax():
    update = lambda key, state, info: dict(BATCHED)
    kernel = tuned_smc(
        PRIOR,
        LIKELIHOOD,
        mcmc_kernel=hmc,
        mcmc_parameters=PARAMS,
        parameter_update_fn=update,
        num_mcmc_steps=2,
        num_integration_steps=3,
        adaptive=False,
        batch_size=8,
    )
    init, step = make_mcmc_adapter(hmc, num_integration_steps=3)
    upstream = blackjax.smc.inner_kernel_tuning.as_top_level_api(
        blackjax.tempered_smc,
        PRIOR,
        LIKELIHOOD,
        step,
        init,
        blackjax.smc.resampling.systematic,
        update,
        BATCHED,
        num_mcmc_steps=2,
        batch_size=8,
    )
    ours = jax.jit(kernel.step)(KEY, kernel.init(PARTICLES), tempering_param=0.3)
    theirs = jax.jit(upstream.step)(KEY, upstream.init(PARTICLES), tempering_param=0.3)
    assert_tree_close(ours, theirs, atol=2e-6)


def test_pretuning_reuses_blackjax_pilot():
    params = {
        'step_size': jnp.full((32,), 0.15),
        'inverse_mass_matrix': jnp.eye(2)[None],
    }
    kernel = pretuned_smc(
        PRIOR,
        LIKELIHOOD,
        mcmc_kernel=hmc,
        mcmc_parameters=params,
        num_particles=32,
        sigma_parameters={'step_size': 0.01},
        positive_parameters=['step_size'],
        num_mcmc_steps=2,
        num_integration_steps=3,
        adaptive=False,
        batch_size=8,
    )
    initial = kernel.init(PARTICLES)
    result = SMC(kernel).run(KEY, initial, jnp.array([0.3, 1.0]), params)
    assert isinstance(result.state, StateWithParameterOverride)
    assert result.params['step_size'].shape == (32,)
    assert (result.params['step_size'] > 0).all()
    assert jnp.isfinite(result.log_evidence)
    np.testing.assert_allclose(initial.parameter_override['step_size'], 0.15)


@pytest.mark.parametrize('factory,num_particles', MODELS)
def test_default_random_walk_recovers_reference_posterior(factory, num_particles):
    grid = jnp.linspace(-5.0, 3.0, 801)
    log_post = norm.logpdf(grid) + jax.vmap(kalman_log_likelihood)(grid)
    weights = jax.nn.softmax(log_post)
    mean = jnp.sum(weights * grid)
    sd = jnp.sqrt(jnp.sum(weights * (grid - mean) ** 2))
    kernel = temporal_smc(
        factory(),
        norm.logpdf,
        num_particles=num_particles,
        ess_threshold=1.0,
        num_moves=3,
        batch_size=64,
    )
    init_key, run_key = jax.random.split(KEY)
    state = kernel.init(init_key, jax.random.normal(init_key, (512,)))
    result = jax.jit(
        lambda key: run_temporal_smc(kernel, key, state, TS, YS, history='full')
    )(run_key)
    assert result.trace.infos.valid.all()
    assert (result.trace.infos.acceptance_rate > 0).any()
    w = jnp.exp(result.state.log_weights)
    smc_mean = jnp.sum(w * result.state.parameters)
    smc_sd = jnp.sqrt(jnp.sum(w * (result.state.parameters - smc_mean) ** 2))
    np.testing.assert_allclose(smc_mean, mean, atol=0.1)
    np.testing.assert_allclose(smc_sd, sd, atol=0.1)
    assert result.state.proposal_scale > 0.0


def test_joint_sampler_matches_gaussian_conditional_moments():
    model = gaussian_model()
    kernel = temporal_smc(model, norm.logpdf)
    state = kernel.init(KEY, jnp.full((4,), THETA))
    state = run_temporal_smc(kernel, KEY, state, TS, YS).state
    sample = sample_joint_paths(
        KEY, state, model, TS, YS, num_samples=4000, batch_size=64
    )
    mean, cov, _ = exact_joint()
    np.testing.assert_allclose(sample.paths[:, :, 0].mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(sample.paths[:, :, 0].T), cov, atol=0.02)


def test_diagnostics_distinguish_uniform_duplicates():
    d = population_diagnostics(
        jnp.array([[1.0], [1.0], [2.0], [2.0]]),
        jnp.full(4, -jnp.log(4)),
        jnp.array([0, 0, 0, 0]),
    )
    assert d.ess == 4 and d.unique_particles == 2 and d.unique_ancestors == 1
    stats = summarize_replicates(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(stats.standard_error, 1 / jnp.sqrt(3), atol=1e-6)


def test_waste_free_preset_runs_adaptively():
    from probjax.inference import waste_free_smc

    kernel = waste_free_smc(
        PRIOR,
        LIKELIHOOD,
        num_particles=32,
        p=4,
        adaptive=True,
        mcmc_kernel=hmc,
        num_integration_steps=3,
        batch_size=8,
    )
    result = SMC(kernel).run_adaptive(KEY, kernel.init(PARTICLES), BATCHED)
    assert result.completed and result.state.particles.shape == PARTICLES.shape
