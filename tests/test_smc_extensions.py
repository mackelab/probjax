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
from probjax.inference.filtering.streaming import (
    append_streaming_window,
    init_streaming_window,
    streaming_window_trace,
)
from probjax.inference.smc import PartialPosteriorsPath
from probjax.inference.smc.base import make_mcmc_adapter
from probjax.inference.smc.particle_adaptation import (
    adapt_particle_count,
    exchange_filter_population,
    recommend_particle_count,
)
from probjax.inference.smc.ports import pretuned_smc, tuned_smc, waste_free_strategy
from probjax.inference.smc.temporal import ReplayData, run_temporal_smc, temporal_smc
from probjax.inference.smc.temporal_utils import (
    likelihood_diagnostics,
    population_diagnostics,
    summarize_replicates,
)
from tests.test_temporal_smc import (
    THETA,
    TS,
    YS,
    assert_tree_close,
    gaussian_backend,
    pf_backend,
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


@pytest.mark.parametrize('backend_factory', [gaussian_backend, pf_backend])
def test_tempered_observation_bridge_reaches_endpoint(backend_factory):
    backend = backend_factory()
    kernel = temporal_smc(
        backend,
        lambda p: norm.logpdf(p),
        tempering_ess=0.95,
        adaptive_proposal=True,
        batch_size=8,
    )
    state = kernel.init(KEY, jnp.linspace(-4.0, 2.0, 32), 0.0)
    result = jax.jit(
        lambda k: run_temporal_smc(kernel, k, state, TS, YS, history='full')
    )(KEY)
    assert result.trace.infos.valid.all()
    assert (result.trace.infos.tempering_param == 1.0).all()
    assert (result.trace.infos.num_tempering_steps > 1).any()
    assert jnp.isfinite(result.state.log_evidence)
    assert result.state.proposal_scale > 0.0


def test_tempering_without_moves_preserves_endpoint_importance_weights():
    backend = gaussian_backend()
    # No stage resampling in a one-stage mild bridge: directly matches the base step.
    parameters = jnp.linspace(-1.1, -1.0, 16)
    a = temporal_smc(backend, lambda p: norm.logpdf(p), ess_threshold=0.0)
    b = temporal_smc(
        backend, lambda p: norm.logpdf(p), ess_threshold=0.0, tempering_ess=0.1
    )
    state = a.init(KEY, parameters, 0.0)
    r1 = run_temporal_smc(a, KEY, state, TS, YS)
    r2 = run_temporal_smc(b, KEY, state, TS, YS)
    assert_tree_close(r1.state, r2.state, atol=3e-6)


def test_tempering_cap_is_atomic():
    backend = gaussian_backend()
    kernel = temporal_smc(
        backend, lambda p: norm.logpdf(p), tempering_ess=0.9999, max_tempering_steps=1
    )
    state = kernel.init(KEY, jnp.linspace(-5.0, 3.0, 32), 0.0)
    result, info = jax.jit(kernel.step)(KEY, state, 1.0, jnp.array([4.0]))
    assert not info.valid and info.tempering_param < 1.0
    assert_tree_close(result, state)


def test_exact_temporal_hmc_and_rejection_of_noisy_gradient_moves():
    backend = gaussian_backend()
    kernel = temporal_smc(
        backend,
        lambda p: norm.logpdf(p),
        ess_threshold=1.0,
        mcmc_kernel=hmc,
        mcmc_parameters={'step_size': 0.05, 'inverse_mass_matrix': jnp.ones(1)},
        mcmc_kernel_kwargs={'num_integration_steps': 2},
    )
    state = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 8), 0.0)
    result = jax.jit(
        lambda k: run_temporal_smc(kernel, k, state, TS, YS, history='full')
    )(KEY)
    assert result.trace.infos.valid.all()
    assert (result.trace.infos.acceptance_rate > 0).any()
    with pytest.raises(ValueError, match='exact'):
        temporal_smc(
            pf_backend(), lambda p: norm.logpdf(p), mcmc_kernel=hmc, mcmc_parameters={}
        )


def test_streaming_window_survives_chunks_and_reports_padding():
    backend = gaussian_backend()
    initial = backend.init(KEY, THETA, 0.0)
    shape = jax.eval_shape(backend.step, KEY, initial, THETA, TS[0], YS[0], True)[1]
    window = init_streaming_window(initial, shape, 3)
    state = initial
    for i in range(4):
        state, info = backend.step(KEY, state, THETA, TS[i], YS[i], True)
        window = jax.jit(append_streaming_window)(window, state, info)
        trace, mask = streaming_window_trace(window)
        assert mask.sum() == min(i + 1, 3)
    np.testing.assert_allclose(trace.ts, TS[1:])
    assert trace.initial_state.t == TS[0]
    assert trace.states.mean.shape == (3, 1)


def test_joint_sampler_matches_gaussian_conditional_moments():
    backend = gaussian_backend()
    kernel = temporal_smc(backend, lambda p: norm.logpdf(p))
    state = kernel.init(KEY, jnp.full((4,), THETA), 0.0)
    state = run_temporal_smc(kernel, KEY, state, TS, YS).state
    data = ReplayData(TS, YS, jnp.ones(4, bool))
    sample = jax.jit(
        lambda key: sample_joint_paths(
            key,
            state,
            backend,
            data,
            num_samples=4000,
            transition_fn=lambda p, a, b: jnp.eye(1),
            batch_size=64,
        )
    )(KEY)
    from tests.test_temporal_smc import exact_joint

    mean, cov, _ = exact_joint()
    np.testing.assert_allclose(sample.paths[:, :, 0].mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(sample.paths[:, :, 0].T), cov, atol=0.02)
    assert sample.valid


def test_diagnostics_distinguish_uniform_duplicates_and_noise():
    d = population_diagnostics(
        jnp.array([[1.0], [1.0], [2.0], [2.0]]),
        jnp.full(4, -jnp.log(4)),
        jnp.array([0, 0, 0, 0]),
    )
    assert d.ess == 4 and d.unique_particles == 2 and d.unique_ancestors == 1
    data = ReplayData(TS, YS, jnp.ones(4, bool))
    exact = likelihood_diagnostics(gaussian_backend(), KEY, THETA, 0.0, data)
    noisy = likelihood_diagnostics(pf_backend(16), KEY, THETA, 0.0, data)
    assert exact.log_variance < 1e-10 and noisy.log_variance > 0
    stats = summarize_replicates(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(stats.standard_error, 1 / jnp.sqrt(3), atol=1e-6)


def test_particle_count_exchange_uses_importance_correction():
    old, new = pf_backend(16), pf_backend(64)
    kernel = temporal_smc(old, lambda p: norm.logpdf(p))
    state = kernel.init(KEY, jnp.linspace(-2.0, 0.0, 16), 0.0)
    state = run_temporal_smc(kernel, KEY, state, TS, YS).state
    data = ReplayData(TS, YS, jnp.ones(4, bool))
    changed, info = jax.jit(lambda k, s: exchange_filter_population(k, s, new, data))(
        KEY, state
    )
    assert info.valid and changed.filter_states.particles.shape == (16, 64, 1)
    expected = state.log_weights + changed.log_likelihoods - state.log_likelihoods
    np.testing.assert_allclose(
        changed.log_weights, jax.nn.log_softmax(expected), atol=1e-6
    )
    assert recommend_particle_count(16, 4.0, (16, 32, 64)).recommended_count == 64
    assert recommend_particle_count(16, 100.0, (16, 32, 64)).at_capacity
    adapted = adapt_particle_count(
        KEY,
        state,
        pf_backend,
        16,
        data,
        buckets=(16, 32, 64),
        target_variance=1e-4,
        num_parameters=2,
        num_replicates=4,
    )
    assert adapted.particle_count > 16
    assert adapted.exchange_info.valid


def test_extended_tempering_preserves_discrete_parameter_and_auxiliary_targets():
    from typing import NamedTuple

    from probjax.inference import TemporalFilter

    class FilterState(NamedTuple):
        value: object
        t: object

    class Info(NamedTuple):
        log_likelihood: object

    def init(key, p, t):
        return FilterState(jnp.where(jax.random.bernoulli(key), 1.75, 0.25), t)

    backend = TemporalFilter(
        init,
        lambda k, s, p, t, y, m: (
            FilterState(s.value, t),
            Info(jnp.log(s.value) + p * jnp.log(3.0)),
        ),
        'unbiased',
    )
    kernel = temporal_smc(
        backend,
        lambda p: -jnp.log(2.0),
        proposal_fn=lambda key, p: (1 - p, 0.0),
        tempering_ess=0.9,
        ess_threshold=1.0,
        num_rejuvenation_steps=2,
    )
    init_key, run_key = jax.random.split(KEY)
    state = kernel.init(init_key, jnp.tile(jnp.array([0.0, 1.0]), 2048), 0.0)
    result = jax.jit(
        lambda key: run_temporal_smc(
            kernel, key, state, jnp.array([1.0]), jnp.zeros((1, 1)), history='full'
        )
    )(run_key)
    weights = jnp.exp(result.state.log_weights)
    np.testing.assert_allclose(
        jnp.sum(weights * result.state.parameters), 0.75, atol=0.025
    )
    np.testing.assert_allclose(
        jnp.sum(weights * (result.state.filter_states.value == 1.75)), 0.875, atol=0.025
    )
    assert result.trace.infos.num_tempering_steps[0] > 1
    assert result.trace.infos.valid[0]


def test_adaptive_move_count_is_bounded_and_persists():
    backend = gaussian_backend()
    kernel = temporal_smc(
        backend,
        lambda p: norm.logpdf(p),
        proposal_fn=lambda key, p: (p + 1.0, -jnp.inf),
        adaptive_num_steps=True,
        max_rejuvenation_steps=5,
        ess_threshold=1.0,
    )
    state = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 8), 0.0)
    result = run_temporal_smc(kernel, KEY, state, TS, YS, history='full')
    assert result.state.move_steps == 5
    assert result.trace.infos.valid.all()


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
