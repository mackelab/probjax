"""Temporal inference checked against Gaussian joint laws and streaming execution."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from probjax.inference import (
    LinearGaussianModel,
    StateSpaceModel,
    TemporalFilter,
    particle_gibbs,
    run_temporal_filter,
    run_temporal_smc,
    sample_paths,
    smooth_gaussian_path,
    temporal_filter,
    temporal_smc,
)
from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.utils.linear_operator import Function

TS = jnp.array([0.2, 0.7, 1.4, 2.0])
YS = jnp.array([[0.4], [-0.1], [0.8], [0.5]])
THETA = jnp.array(jnp.log(0.3))
KEY = jax.random.key(13)


def gaussian_model():
    return LinearGaussianModel(
        lambda theta: (jnp.zeros(1), jnp.eye(1)),
        lambda theta, old, new: (jnp.eye(1), jnp.eye(1) * jnp.exp(theta) * (new - old)),
        lambda theta, t: (jnp.eye(1), jnp.eye(1) * 0.2),
    )


def particle_model():
    return StateSpaceModel(
        init=lambda key, theta: jax.random.normal(key, (1,)),
        step=lambda key, theta, x, old, new: x
        + jnp.sqrt(jnp.exp(theta) * (new - old)) * jax.random.normal(key, x.shape),
        log_likelihood=lambda theta, x, y, t: norm.logpdf(y, x, jnp.sqrt(0.2)).sum(),
        transition_logdensity=lambda theta, new, old, t0, t1: norm.logpdf(
            new, old, jnp.sqrt(jnp.exp(theta) * (t1 - t0))
        ).sum(),
    )


# (model, num_particles) pairs for tests run against both filters.
MODELS = [(gaussian_model, None), (particle_model, 64)]


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


def kalman_log_likelihood(theta):
    return run_temporal_filter(
        KEY, gaussian_model(), theta, TS, YS, history="none"
    ).log_likelihood


@pytest.mark.parametrize("mask", [None, jnp.array([True, False, True, False])])
def test_gaussian_likelihood_and_smoothing_match_joint(mask):
    model = gaussian_model()
    result = jax.jit(
        lambda key: run_temporal_filter(key, model, THETA, TS, YS, observed=mask)
    )(KEY)
    mean, cov, logz = exact_joint(mask)
    smoothed_mean, smoothed_cov = smooth_gaussian_path(result.trace, model, THETA)
    np.testing.assert_allclose(result.log_likelihood, logz, rtol=2e-6)
    np.testing.assert_allclose(smoothed_mean[:, 0], mean, atol=2e-6)
    np.testing.assert_allclose(smoothed_cov[:, 0, 0], np.diag(cov), atol=2e-6)
    samples = jax.jit(
        lambda key: sample_paths(key, result.trace, model, THETA, num_samples=16000)
    )(KEY)[:, :, 0]
    np.testing.assert_allclose(samples.mean(1), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(samples), cov, atol=0.012)


@pytest.mark.parametrize("factory,num_particles", MODELS)
@pytest.mark.parametrize("history", ["none", "full", 1, 3, 10])
def test_streaming_and_bounded_history(factory, num_particles, history):
    model = factory()
    mask = jnp.array([True, False, True, True])
    y = YS.at[1].set(jnp.nan)
    result = jax.jit(
        lambda key: run_temporal_filter(
            key,
            model,
            THETA,
            TS,
            y,
            num_particles=num_particles,
            observed=mask,
            history=history,
        )
    )(KEY)
    # The same run, one step at a time.
    filt = temporal_filter(model, num_particles)
    key, init_key = jax.random.split(KEY)
    initial = state = filt.init(init_key, THETA, 0.0)
    logz, streaming = 0.0, []
    for t, obs, present in zip(TS, y, mask, strict=True):
        key, step_key = jax.random.split(key)
        state, info = jax.jit(filt.step)(step_key, state, THETA, t, obs, present)
        logz += info.log_likelihood
        streaming.append(state)
    assert_tree_close(result.state, state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.log_likelihood, logz, rtol=1e-5)
    if history == "none":
        assert result.trace is None
    else:
        count = 4 if history == "full" else min(history, 4)
        full = jax.tree.map(lambda *xs: jnp.stack(xs), *streaming[-count:])
        assert_tree_close(result.trace.states, full, rtol=1e-5, atol=1e-5)
        assert_tree_close(
            result.trace.initial_state,
            initial if count == 4 else streaming[-count - 1],
            rtol=1e-5,
            atol=1e-6,
        )


def test_pf_increment_uses_normalized_weights():
    kernel = ParticleFilter(lambda x, y, t: jnp.full((8,), -2.0), lambda key, x, t: x)
    state = kernel.init(jnp.arange(8.0)[:, None], t=0.0)
    _, info = kernel(state, 1.0, jnp.zeros(1), KEY)
    np.testing.assert_allclose(info.log_likelihood, -2.0, atol=1e-6)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_pf_likelihood_estimate_is_unbiased(threshold):
    filt = temporal_filter(particle_model(), 64, ess_threshold=threshold)

    def estimate(key):
        return run_temporal_filter(key, filt, THETA, TS, YS, history="none")

    logz = jax.jit(jax.vmap(estimate))(jax.random.split(KEY, 512)).log_likelihood
    estimated = jnp.exp(logz).mean()
    np.testing.assert_allclose(estimated, jnp.exp(exact_joint()[2]), rtol=0.045)


def test_linear_operator_matches_dense_kalman():
    c = jnp.array([[1.0, 0.4]])
    operator = Function(lambda x: c @ x, 2)

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
    model = particle_model()
    filt = temporal_filter(model, 256, ess_threshold=1.0)
    result = run_temporal_filter(KEY, filt, THETA, TS, YS)
    for method in ("ancestry", "backward"):
        paths = jax.jit(
            lambda key, method=method: sample_paths(
                key, result.trace, model, THETA, num_samples=500, method=method
            )
        )(KEY)
        assert paths.shape == (5, 500, 1)
        np.testing.assert_allclose(paths[:, :, 0].mean(1), exact_joint()[0], atol=0.12)
    # Deterministic translations identify ancestry exactly, including resampling.
    shift = StateSpaceModel(
        init=lambda key, p: jax.random.normal(key, (1,)),
        step=lambda key, p, x, a, b: x + (b - a),
        log_likelihood=lambda p, x, y, t: -0.1 * (x[0] - y[0]) ** 2,
    )
    r = run_temporal_filter(KEY, temporal_filter(shift, 32, 1.0), None, TS, YS)
    paths = sample_paths(KEY, r.trace, shift, None, num_samples=100)
    np.testing.assert_allclose(
        np.diff(paths[:, :, 0], axis=0),
        np.broadcast_to(np.diff(np.r_[0.0, TS])[:, None], (4, 100)),
        atol=1e-6,
    )


@pytest.mark.parametrize("ancestor_sampling", [True, False])
def test_conditional_smc_preserves_gaussian_smoothing(ancestor_sampling):
    mean, cov, _ = exact_joint()
    reference_key, update_key = jax.random.split(KEY)
    keys = jax.random.split(update_key, 3000)
    references = jax.random.multivariate_normal(
        reference_key, jnp.array(mean), jnp.array(cov), (3000,)
    )[:, :, None]

    def update(key, reference):
        return particle_gibbs(
            key,
            reference,
            particle_model(),
            THETA,
            TS,
            YS,
            num_particles=8,
            ancestor_sampling=ancestor_sampling,
        )

    paths = jax.jit(jax.vmap(update))(keys, references)[:, :, 0]
    np.testing.assert_allclose(paths.mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(paths.T), cov, atol=0.02)


def test_outer_weights_equal_exact_parameter_mixture():
    theta = jnp.linspace(-3.0, 1.0, 40)
    kernel = temporal_smc(gaussian_model(), norm.logpdf, ess_threshold=0.0)
    state = kernel.init(KEY, theta)
    result = jax.jit(
        lambda key, s: run_temporal_smc(kernel, key, s, TS, YS, history="full")
    )(KEY, state)
    likelihoods = jax.vmap(kalman_log_likelihood)(theta)
    np.testing.assert_allclose(
        result.state.log_weights, jax.nn.log_softmax(likelihoods), atol=2e-6
    )
    np.testing.assert_allclose(
        result.state.log_evidence,
        jax.scipy.special.logsumexp(likelihoods) - jnp.log(len(theta)),
        atol=2e-6,
    )
    assert result.trace.infos.valid.all()


@pytest.mark.parametrize("factory,num_particles", MODELS)
def test_rejected_moves_preserve_likelihood_and_filter_state(factory, num_particles):
    options = dict(num_particles=num_particles, ess_threshold=1.0)
    plain = temporal_smc(factory(), norm.logpdf, num_moves=0, **options)
    reject = temporal_smc(
        factory(),
        norm.logpdf,
        num_moves=2,
        proposal_fn=lambda key, p: (p + 1.0, -jnp.inf),
        **options,
    )
    initial = plain.init(KEY, jnp.linspace(-3.0, 1.0, 16))
    baseline, rejected = (
        jax.jit(lambda key, k=k: run_temporal_smc(k, key, initial, TS, YS, history="full"))(
            KEY
        )
        for k in (plain, reject)
    )
    assert_tree_close(baseline.state, rejected.state, rtol=1e-6, atol=1e-6)
    assert rejected.trace.infos.resampled.any()
    assert (rejected.trace.infos.acceptance_rate == 0).all()


def test_moves_keep_exact_likelihoods_with_pytree_parameters():
    model = gaussian_model()
    # Parameters are a dict; the model reads its one entry.
    wrapped = LinearGaussianModel(
        lambda p: model.initial(p["scale"]),
        lambda p, a, b: model.transition(p["scale"], a, b),
        lambda p, t: model.observation(p["scale"], t),
    )
    kernel = temporal_smc(
        wrapped, lambda p: norm.logpdf(p["scale"]), ess_threshold=1.0, num_moves=2
    )
    initial = kernel.init(KEY, {"scale": jnp.linspace(-3.0, 1.0, 16)})
    result = jax.jit(
        lambda key: run_temporal_smc(kernel, key, initial, TS, YS, history="full")
    )(KEY)
    expected = jax.vmap(kalman_log_likelihood)(result.state.parameters["scale"])
    np.testing.assert_allclose(result.state.log_likelihoods, expected, atol=2e-6)
    assert result.trace.infos.acceptance_rate.max() > 0.0
    assert result.state.posterior_samples(KEY, 5)["scale"].shape == (5,)


@pytest.mark.parametrize("factory,num_particles", MODELS)
def test_resuming_with_longer_data_matches_one_run(factory, num_particles):
    kernel = temporal_smc(
        factory(), norm.logpdf, num_particles=num_particles, ess_threshold=1.0
    )
    initial = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 16))
    run = jax.jit(
        lambda key, state, ts, ys: run_temporal_smc(kernel, key, state, ts, ys)
    )
    batch = run(KEY, initial, TS, YS)
    first = run(KEY, initial, TS[:2], YS[:2])
    second = run(first.key, first.state, TS, YS)
    assert_tree_close(second.state, batch.state, atol=2e-6)
    np.testing.assert_allclose(
        first.log_likelihood + second.log_likelihood, batch.log_likelihood, atol=2e-5
    )
    # Data that does not extend what the state has seen is rejected.
    shifted = run(first.key, first.state, TS + 0.05, YS)
    assert_tree_close(shifted.state, first.state)


def test_moves_do_not_read_future_observations():
    kernel = temporal_smc(gaussian_model(), norm.logpdf, ess_threshold=1.0)
    initial = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 16))

    def two_steps(future):
        ys = YS.at[2:].set(future)
        state, _ = kernel.step(KEY, initial, TS, ys)
        state, _ = kernel.step(KEY, state, TS, ys)
        return state

    assert_tree_close(jax.jit(two_steps)(1.0), jax.jit(two_steps)(10000.0))


def test_pseudo_marginal_moves_preserve_size_biased_auxiliary_law():
    """An unbiased estimator U must have law proportional to U after weighting/MH."""

    class State(NamedTuple):
        value: object
        t: object

    class Info(NamedTuple):
        log_likelihood: object

    def init(key, theta, t):
        return State(jnp.where(jax.random.bernoulli(key), 1.75, 0.25), t)

    filt = TemporalFilter(
        init,
        lambda key, s, p, t, y, mask: (State(s.value, t), Info(jnp.log(s.value))),
        "unbiased",
    )
    kernel = temporal_smc(
        filt,
        lambda p: jnp.array(0.0),
        ess_threshold=1.0,
        proposal_fn=lambda key, p: (p, 0.0),
        num_moves=3,
    )
    init_key, run_key = jax.random.split(KEY)
    initial = kernel.init(init_key, jnp.zeros(4096))
    result = jax.jit(
        lambda key: run_temporal_smc(
            kernel, key, initial, jnp.array([1.0]), jnp.zeros((1, 1)), history="full"
        )
    )(run_key)
    # q(U=1.75)*1.75 / E_q[U] = 0.875, not the proposal probability 0.5.
    frequency = jnp.mean(result.state.filter_states.value == 1.75)
    np.testing.assert_allclose(frequency, 0.875, atol=0.025)
    np.testing.assert_allclose(
        result.state.log_likelihoods, jnp.log(result.state.filter_states.value)
    )
    assert result.trace.infos.resampled[0]


def test_pytree_latent_states():
    model = StateSpaceModel(
        init=lambda key, p: {"x": jax.random.normal(key, (1,)), "steps": jnp.zeros(())},
        step=lambda key, p, s, a, b: {
            "x": particle_model().step(key, p, s["x"], a, b),
            "steps": s["steps"] + 1,
        },
        log_likelihood=lambda p, s, y, t: particle_model().log_likelihood(
            p, s["x"], y, t
        ),
    )
    result = run_temporal_filter(KEY, model, THETA, TS, YS, num_particles=64)
    assert result.state.particles["x"].shape == (64, 1)
    assert (result.state.particles["steps"] == 4).all()
    np.testing.assert_allclose(result.log_likelihood, exact_joint()[2], atol=0.3)


def test_missing_observation_only_predicts():
    filt = temporal_filter(particle_model(), 512)
    initial = filt.init(KEY, THETA, 0.0)
    predicted, info = jax.jit(filt.step)(
        KEY, initial, THETA, TS[0], jnp.array([jnp.nan]), False
    )
    assert jnp.isfinite(predicted.particles).all()
    assert info.log_likelihood == 0.0


def test_invalid_inputs():
    model = gaussian_model()
    with pytest.raises(ValueError, match="time dimension"):
        run_temporal_filter(KEY, model, THETA, TS, YS[:2])
    with pytest.raises(ValueError, match="boolean"):
        run_temporal_filter(KEY, model, THETA, TS, YS, observed=jnp.ones(4))
    with pytest.raises(ValueError, match="history"):
        run_temporal_filter(KEY, model, THETA, TS, YS, history=0)
    with pytest.raises(ValueError, match="num_particles"):
        run_temporal_filter(KEY, particle_model(), THETA, TS, YS)
    with pytest.raises(ValueError, match="exact or unbiased"):
        temporal_smc(TemporalFilter(None, None, "approximate"), norm.logpdf)


def test_empty_run_and_single_state_paths():
    model = gaussian_model()
    result = run_temporal_filter(KEY, model, THETA, TS[:0], YS[:0])
    assert result.log_likelihood == 0.0
    assert result.state.t == 0.0
    assert sample_paths(KEY, result.trace, model, THETA).shape == (1, 1, 1)


def test_all_impossible_observations_report_failure():
    model = particle_model()._replace(
        log_likelihood=lambda p, x, y, t: jnp.array(-jnp.inf)
    )
    kernel = temporal_smc(model, norm.logpdf, num_particles=8)
    state = kernel.init(KEY, jnp.zeros(4))
    next_state, info = jax.jit(kernel.step)(KEY, state, TS, YS)
    assert not info.valid
    assert not info.resampled
    assert_tree_close(next_state, state)


def test_nan_likelihoods_get_zero_weight():
    # E.g. a simulator diverging for some parameters.
    model = particle_model()._replace(
        log_likelihood=lambda p, x, y, t: jnp.where(
            p > 0, jnp.nan, particle_model().log_likelihood(p, x, y, t)
        )
    )
    kernel = temporal_smc(model, norm.logpdf, num_particles=8, num_moves=0)
    state = kernel.init(KEY, jnp.array([-1.0, -0.5, 0.5, 1.0]))
    next_state, info = jax.jit(kernel.step)(KEY, state, TS, YS)
    assert info.valid
    assert jnp.isfinite(next_state.log_evidence)
    assert (next_state.parameters[jnp.isfinite(next_state.log_weights)] < 0).all()


@pytest.mark.parametrize("factory,num_particles", MODELS)
def test_integer_initial_time_with_fractional_intervals(factory, num_particles):
    model = factory()
    result = jax.jit(
        lambda key: run_temporal_filter(
            key, model, THETA, TS, YS, t0=0, num_particles=num_particles
        )
    )(KEY)
    assert result.state.t == TS[-1]
    kernel = temporal_smc(model, norm.logpdf, num_particles=num_particles)
    state = kernel.init(KEY, jnp.zeros(4), 0)
    result, info = jax.jit(kernel.step)(KEY, state, jnp.array([1]), YS[:1])
    assert result.t == 1.0
    assert info.valid
