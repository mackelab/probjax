"""Temporal inference checked against Gaussian joint laws and streaming execution."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from probjax.inference import (
    ReplayData,
    kalman_backend,
    particle_backend,
    particle_gibbs,
    run_temporal_filter,
    run_temporal_smc,
    sample_gaussian_paths,
    sample_particle_paths,
    smooth_gaussian_path,
    temporal_smc,
)
from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.utils.linear_operator import LinearOperator

TS = jnp.array([0.2, 0.7, 1.4, 2.0])
YS = jnp.array([[0.4], [-0.1], [0.8], [0.5]])
THETA = jnp.array(jnp.log(0.3))
KEY = jax.random.key(13)


def transition(theta, old, new):
    return jnp.eye(1), jnp.eye(1) * jnp.exp(theta) * (new - old)


def gaussian_backend():
    return kalman_backend(
        lambda theta, t: (jnp.zeros(1), jnp.eye(1)),
        transition,
        lambda theta, t: (jnp.eye(1), jnp.eye(1) * 0.2),
    )


def initial_particles(key, theta, t, n=64):
    return jax.random.normal(key, (n, 1))


def particle_transition(key, theta, particles, old, new):
    return particles + jnp.sqrt(jnp.exp(theta) * (new - old)) * jax.random.normal(
        key, particles.shape
    )


def transition_density(theta, new, old, t0, t1):
    return norm.logpdf(new, old, jnp.sqrt(jnp.exp(theta) * (t1 - t0))).sum(-1)


def likelihood(theta, particles, y, t):
    return norm.logpdf(y, particles, jnp.sqrt(0.2)).sum(-1)


def pf_backend(n=64, ess_threshold=0.5):
    return particle_backend(
        partial(initial_particles, n=n),
        particle_transition,
        likelihood,
        ess_threshold=ess_threshold,
    )


def exact_joint(mask=None):
    times = np.r_[0.0, np.asarray(TS)]
    prior_cov = 1 + 0.3 * np.minimum(times[:, None], times[None, :])
    idx = np.arange(1, len(times))
    if mask is not None:
        idx = idx[np.asarray(mask)]
    cross = prior_cov[:, idx]
    obs_cov = prior_cov[np.ix_(idx, idx)] + 0.2 * np.eye(len(idx))
    obs = np.asarray(YS)[idx - 1, 0]
    mean = cross @ np.linalg.solve(obs_cov, obs)
    cov = prior_cov - cross @ np.linalg.solve(obs_cov, cross.T)
    logz = multivariate_normal.logpdf(
        jnp.asarray(obs), jnp.zeros(len(idx)), jnp.asarray(obs_cov)
    )
    return mean, cov, logz


def assert_tree_close(left, right, **kwargs):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if hasattr(a, "dtype") and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_allclose(a, b, **kwargs)


@pytest.mark.parametrize('mask', [None, jnp.array([True, False, True, False])])
def test_gaussian_likelihood_and_smoothing_match_joint(mask):
    backend = gaussian_backend()
    state = backend.init(KEY, THETA, 0.0)
    result = jax.jit(
        lambda key: run_temporal_filter(
            backend, key, state, THETA, TS, YS, observed=mask
        )
    )(KEY)
    mean, cov, logz = exact_joint(mask)
    smoothed_mean, smoothed_cov = jax.jit(
        lambda trace: smooth_gaussian_path(trace, lambda a, b: jnp.eye(1))
    )(result.trace)
    np.testing.assert_allclose(result.log_likelihood, logz, rtol=2e-6)
    np.testing.assert_allclose(smoothed_mean[:, 0], mean, atol=2e-6)
    np.testing.assert_allclose(smoothed_cov[:, 0, 0], np.diag(cov), atol=2e-6)
    samples = jax.jit(
        lambda key: sample_gaussian_paths(
            key, result.trace, lambda a, b: jnp.eye(1), num_samples=16000
        )
    )(KEY)[:, :, 0]
    np.testing.assert_allclose(samples.mean(1), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(samples), cov, atol=0.012)


@pytest.mark.parametrize('backend_factory', [gaussian_backend, pf_backend])
@pytest.mark.parametrize('history', ['none', 'full', 1, 3, 10])
def test_streaming_and_bounded_history(backend_factory, history):
    backend = backend_factory()
    initial = backend.init(KEY, THETA, 0.0)
    mask = jnp.array([True, False, True, True])
    y = YS.at[1].set(jnp.nan)
    result = jax.jit(
        lambda key, state: run_temporal_filter(
            backend, key, state, THETA, TS, y, observed=mask, history=history
        )
    )(KEY, initial)
    key, state, logz = KEY, initial, 0.0
    streaming = []
    for t, obs, present in zip(TS, y, mask, strict=True):
        key, step_key = jax.random.split(key)
        state, info = jax.jit(backend.step)(step_key, state, THETA, t, obs, present)
        logz += info.log_likelihood
        streaming.append(state)
    assert_tree_close(result.state, state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.log_likelihood, logz, rtol=1e-5)
    if history == 'none':
        assert result.trace is None
    else:
        count = 4 if history == 'full' else min(history, 4)
        full = jax.tree.map(lambda *xs: jnp.stack(xs), *streaming[-count:])
        assert_tree_close(result.trace.states, full, atol=1e-6)
        assert_tree_close(
            result.trace.initial_state, initial if count == 4 else streaming[-count - 1]
        )


def test_pf_increment_uses_normalized_weights():
    kernel = ParticleFilter(lambda x, y, t: jnp.full((8,), -2.0), lambda key, x, t: x)
    state = kernel.init(jnp.arange(8.0)[:, None], t=0.0)
    _, info = kernel(state, 1.0, jnp.zeros(1), KEY)
    np.testing.assert_allclose(info.log_likelihood, -2.0, atol=1e-6)


@pytest.mark.parametrize('threshold', [0.0, 1.0])
def test_pf_likelihood_estimate_is_unbiased(threshold):
    backend = pf_backend(64, threshold)

    def estimate(key):
        k1, k2 = jax.random.split(key)
        return run_temporal_filter(
            backend, k2, backend.init(k1, THETA, 0.0), THETA, TS, YS, history='none'
        ).log_likelihood

    logz = jax.jit(jax.vmap(estimate))(jax.random.split(KEY, 512))
    estimated = jnp.exp(logz).mean()
    np.testing.assert_allclose(estimated, jnp.exp(exact_joint()[2]), rtol=0.045)


def test_linear_operator_matches_dense_kalman():
    c = jnp.array([[1.0, 0.4]])
    operator = LinearOperator(lambda x: c @ x, 2, 1)

    def make(obs):
        return kalman_filter(
            lambda old, new: (jnp.eye(2), jnp.eye(2) * 0.1),
            lambda t: (obs, jnp.eye(1) * 0.2),
        )

    dense, lazy = make(c), make(operator)
    initial = dense.init(
        jnp.array([0.2, -0.1]), jnp.array([[2.0, 0.3], [0.3, 0.5]]), 0.0
    )
    assert_tree_close(
        dense(initial, 1.0, jnp.array([0.4])),
        lazy(initial, 1.0, jnp.array([0.4])),
        atol=1e-6,
    )


def test_ancestry_and_backward_paths():
    backend = pf_backend(256, 1.0)
    initial = backend.init(KEY, THETA, 0.0)
    result = run_temporal_filter(backend, KEY, initial, THETA, TS, YS)
    for method in ('ancestry', 'backward'):
        paths = jax.jit(
            lambda key, method=method: sample_particle_paths(
                key,
                result.trace,
                num_samples=500,
                method=method,
                transition_logdensity_fn=lambda new, old, a, b: transition_density(
                    THETA, new, old, a, b
                ),
            )
        )(KEY)
        assert paths.shape == (5, 500, 1)
        np.testing.assert_allclose(paths[:, :, 0].mean(1), exact_joint()[0], atol=0.12)
    # Deterministic translations identify ancestry exactly, including resampling.
    b = particle_backend(
        lambda key, p, t: jnp.arange(32.0)[:, None],
        lambda key, p, x, a, b: x + (b - a),
        lambda p, x, y, t: -0.1 * (x[:, 0] - y[0]) ** 2,
        ess_threshold=1.0,
    )
    r = run_temporal_filter(b, KEY, b.init(KEY, None, 0.0), None, TS, YS)
    paths = sample_particle_paths(KEY, r.trace, num_samples=100, method='ancestry')
    np.testing.assert_allclose(
        np.diff(paths[:, :, 0], axis=0),
        np.broadcast_to(np.diff(np.r_[0.0, TS])[:, None], (4, 100)),
        atol=1e-6,
    )


@pytest.mark.parametrize('ancestor_sampling', [True, False])
def test_conditional_smc_preserves_gaussian_smoothing(ancestor_sampling):
    mean, cov, _ = exact_joint()
    reference_key, update_key = jax.random.split(KEY)
    keys = jax.random.split(update_key, 3000)
    references = jax.random.multivariate_normal(
        reference_key, jnp.array(mean), jnp.array(cov), (3000,)
    )[:, :, None]
    update = partial(
        particle_gibbs,
        theta=THETA,
        t0=0.0,
        ts=TS,
        observations=YS,
        initial_fn=partial(initial_particles, n=8),
        transition_fn=particle_transition,
        transition_logdensity_fn=transition_density,
        log_likelihood_fn=likelihood,
        num_particles=8,
        ancestor_sampling=ancestor_sampling,
    )
    paths = jax.jit(jax.vmap(lambda k, ref: update(k, ref)))(keys, references)[:, :, 0]
    np.testing.assert_allclose(paths.mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(paths.T), cov, atol=0.02)


def test_outer_weights_equal_exact_parameter_mixture():
    backend = gaussian_backend()
    theta = jnp.linspace(-3.0, 1.0, 40)
    kernel = temporal_smc(backend, lambda p: norm.logpdf(p), ess_threshold=0.0)
    state = kernel.init(KEY, theta, 0.0)
    result = jax.jit(
        lambda key, s: run_temporal_smc(kernel, key, s, TS, YS, history='full')
    )(KEY, state)
    likelihoods = jax.vmap(
        lambda p: (
            run_temporal_filter(
                backend, KEY, backend.init(KEY, p, 0.0), p, TS, YS, history='none'
            ).log_likelihood
        )
    )(theta)
    expected = jax.nn.log_softmax(likelihoods)
    np.testing.assert_allclose(result.state.log_weights, expected, atol=2e-6)
    np.testing.assert_allclose(
        result.state.log_evidence,
        jax.scipy.special.logsumexp(likelihoods) - jnp.log(len(theta)),
        atol=2e-6,
    )
    assert result.trace.infos.valid.all()


@pytest.mark.parametrize('factory', [gaussian_backend, pf_backend])
def test_rejuvenation_rejections_preserve_likelihood_and_filter_state(factory):
    backend = factory()
    plain = temporal_smc(backend, lambda p: norm.logpdf(p), ess_threshold=1.0)
    reject = temporal_smc(
        backend,
        lambda p: norm.logpdf(p),
        ess_threshold=1.0,
        proposal_fn=lambda key, p: (p + 1.0, -jnp.inf),
        num_rejuvenation_steps=2,
    )
    initial = plain.init(KEY, jnp.linspace(-3.0, 1.0, 16), 0.0)
    run = lambda kernel: jax.jit(
        lambda key: run_temporal_smc(kernel, key, initial, TS, YS, history='full')
    )(KEY)
    baseline, rejected = run(plain), run(reject)
    assert_tree_close(baseline.state, rejected.state, rtol=1e-6, atol=1e-6)
    assert rejected.trace.infos.resampled.any()
    assert (rejected.trace.infos.acceptance_rate == 0).all()


def test_parameter_replay_matches_accepted_parameters_and_streaming():
    backend = gaussian_backend()
    # Exercise PyTrees by adapting only the model's parameter argument.
    wrapped = backend._replace(
        init=lambda k, p, t: backend.init(k, p['scale'], t),
        step=lambda k, s, p, t, y, m: backend.step(k, s, p['scale'], t, y, m),
    )
    kernel = temporal_smc(
        wrapped,
        lambda p: norm.logpdf(p['scale']),
        ess_threshold=1.0,
        proposal_fn=lambda k, p: (
            {'scale': p['scale'] + 0.3 * jax.random.normal(k)},
            0.0,
        ),
    )
    initial = kernel.init(KEY, {'scale': jnp.linspace(-3.0, 1.0, 16)}, 0.0)
    replay = ReplayData(TS, YS, jnp.ones(4, bool))
    result = jax.jit(
        lambda key: run_temporal_smc(kernel, key, initial, TS, YS, history='full')
    )(KEY)
    likelihoods = jax.vmap(
        lambda p: (
            run_temporal_filter(
                wrapped, KEY, wrapped.init(KEY, p, 0.0), p, TS, YS, history='none'
            ).log_likelihood
        )
    )(result.state.parameters)
    np.testing.assert_allclose(result.state.log_likelihoods, likelihoods, atol=2e-6)
    assert result.trace.infos.acceptance_rate.max() > 0.0
    first = run_temporal_smc(kernel, KEY, initial, TS[:2], YS[:2], replay=replay)
    second = run_temporal_smc(
        kernel, first.key, first.state, TS[2:], YS[2:], replay=replay
    )
    assert_tree_close(result.state, second.state, atol=2e-6)
    with pytest.raises(ValueError, match='full-prefix'):
        run_temporal_smc(kernel, first.key, first.state, TS[2:], YS[2:])
    bad = jax.jit(
        lambda s: run_temporal_smc(kernel, KEY, s, TS[2:], YS[2:], history='full')
    )(first.state)
    assert not bad.trace.infos.valid.any()
    assert_tree_close(bad.state, first.state)


def test_invalid_shapes_proposals_and_approximate_backend():
    backend = gaussian_backend()
    state = backend.init(KEY, THETA, 0.0)
    with pytest.raises(ValueError, match='time dimension'):
        run_temporal_filter(backend, KEY, state, THETA, TS, YS[:2])
    with pytest.raises(ValueError, match='boolean'):
        run_temporal_filter(backend, KEY, state, THETA, TS, YS, observed=jnp.ones(4))
    with pytest.raises(ValueError, match='history'):
        run_temporal_filter(backend, KEY, state, THETA, TS, YS, history=0)
    with pytest.raises(ValueError, match='proposal'):
        particle_backend(
            initial_particles,
            particle_transition,
            likelihood,
            proposal_fn=particle_transition,
        )
    with pytest.raises(ValueError, match='exact or unbiased'):
        temporal_smc(backend._replace(likelihood_kind='approximate'), lambda p: 0.0)


def test_empty_run_and_single_state_paths():
    backend = gaussian_backend()
    state = backend.init(KEY, THETA, 0.0)
    result = run_temporal_filter(backend, KEY, state, THETA, TS[:0], YS[:0])
    assert_tree_close(result.state, state)
    assert result.log_likelihood == 0.0
    assert sample_gaussian_paths(KEY, result.trace, lambda a, b: jnp.eye(1)).shape == (
        1,
        1,
        1,
    )


def test_pseudo_marginal_moves_preserve_size_biased_auxiliary_law():
    """An unbiased estimator U must have law proportional to U after weighting/MH."""
    from typing import NamedTuple

    from probjax.inference import TemporalFilter

    class State(NamedTuple):
        value: object
        t: object

    class Info(NamedTuple):
        log_likelihood: object

    def init(key, theta, t):
        return State(jnp.where(jax.random.bernoulli(key), 1.75, 0.25), t)

    backend = TemporalFilter(
        init,
        lambda key, s, p, t, y, mask: (State(s.value, t), Info(jnp.log(s.value))),
        'unbiased',
    )
    kernel = temporal_smc(
        backend,
        lambda p: jnp.array(0.0),
        ess_threshold=1.0,
        proposal_fn=lambda key, p: (p, 0.0),
        num_rejuvenation_steps=3,
    )
    init_key, run_key = jax.random.split(KEY)
    initial = kernel.init(init_key, jnp.zeros(4096), 0.0)
    result = jax.jit(
        lambda key: run_temporal_smc(
            kernel, key, initial, jnp.array([1.0]), jnp.zeros((1, 1)), history='full'
        )
    )(run_key)
    # q(U=1.75)*1.75 / E_q[U] = 0.875, not the proposal probability 0.5.
    frequency = jnp.mean(result.state.filter_states.value == 1.75)
    np.testing.assert_allclose(frequency, 0.875, atol=0.025)
    np.testing.assert_allclose(
        result.state.log_likelihoods, jnp.log(result.state.filter_states.value)
    )
    assert result.trace.infos.resampled[0]


def test_replay_does_not_use_future_observations():
    backend = gaussian_backend()
    kernel = temporal_smc(
        backend,
        lambda p: norm.logpdf(p),
        ess_threshold=1.0,
        proposal_fn=lambda k, p: (p + 0.1 * jax.random.normal(k), 0.0),
    )
    initial = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 16), 0.0)

    def run(future):
        replay = ReplayData(TS, YS.at[2:].set(future), jnp.ones(4, bool))
        return run_temporal_smc(kernel, KEY, initial, TS[:2], YS[:2], replay=replay)

    assert_tree_close(jax.jit(run)(1.0), jax.jit(run)(10000.0))


def test_observation_conditioned_proposal_and_missing_data():
    q = jnp.exp(THETA)

    def proposal(key, theta, x, previous, current, y):
        var = jnp.exp(theta) * (current - previous)
        mean = (0.2 * x + var * y) / (0.2 + var)
        std = jnp.sqrt(var * 0.2 / (var + 0.2))
        return mean + std * jax.random.normal(key, x.shape)

    def proposal_density(theta, new, old, previous, current, y):
        var = jnp.exp(theta) * (current - previous)
        mean = (0.2 * old + var * y) / (0.2 + var)
        return norm.logpdf(new, mean, jnp.sqrt(var * 0.2 / (var + 0.2))).sum(-1)

    backend = particle_backend(
        partial(initial_particles, n=512),
        particle_transition,
        likelihood,
        proposal_fn=proposal,
        proposal_logdensity_fn=proposal_density,
        transition_logdensity_fn=transition_density,
        ess_threshold=0.0,
    )
    initial = backend.init(KEY, THETA, 0.0)
    result, info = jax.jit(backend.step)(KEY, initial, THETA, TS[0], YS[0], True)
    expected = norm.logpdf(YS[0], initial.particles, jnp.sqrt(q * TS[0] + 0.2)).sum(-1)
    np.testing.assert_allclose(
        info.log_likelihood,
        jax.scipy.special.logsumexp(expected) - jnp.log(512),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.log_weights, jax.nn.log_softmax(expected), rtol=1e-6, atol=3e-6
    )
    predicted, missing = jax.jit(backend.step)(
        KEY, initial, THETA, TS[0], jnp.array([jnp.nan]), False
    )
    assert jnp.isfinite(predicted.particles).all()
    assert missing.log_likelihood == 0.0


def test_all_impossible_observations_report_failure():
    backend = particle_backend(
        initial_particles,
        particle_transition,
        lambda p, x, y, t: jnp.full((x.shape[0],), -jnp.inf),
    )
    kernel = temporal_smc(backend, lambda p: norm.logpdf(p))
    state = kernel.init(KEY, jnp.zeros(4), 0.0)
    _, info = jax.jit(kernel.step)(KEY, state, 1.0, jnp.zeros(1))
    assert not info.valid
    assert not info.resampled


@pytest.mark.parametrize('factory', [gaussian_backend, pf_backend])
def test_integer_initial_time_with_fractional_intervals(factory):
    backend = factory()
    state = backend.init(KEY, THETA, 0)
    result = jax.jit(lambda s: run_temporal_filter(backend, KEY, s, THETA, TS, YS))(
        state
    )
    assert result.state.t == TS[-1]
    kernel = temporal_smc(backend, lambda p: norm.logpdf(p))
    state = kernel.init(KEY, jnp.zeros(4), 0)
    result, info = jax.jit(kernel.step)(KEY, state, 1, YS[0])
    assert result.t == 1.0
    assert info.valid
