"""Temporal inference checked against Gaussian joint laws and streaming execution."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from probjax.inference import (
    Kernel,
    LinearGaussianModel,
    StateSpaceModel,
    kalman_filter,
    particle_filter,
    particle_gibbs,
    run,
    sample_paths,
    smooth,
    temporal_smc,
)
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


def kalman():
    return kalman_filter(gaussian_model())


def particles(num_particles=64, **options):
    return particle_filter(particle_model(), num_particles, **options)


# Filter factories for tests run against both filters.
FILTERS = [kalman, particles]
# Kept for test_smc_extensions, which reuses these fixtures.
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


def filter_run(filt, theta=THETA, key=KEY, observed=None, ts=TS, ys=YS, t0=0.0):
    init_key, run_key = jax.random.split(key)
    initial = filt.init(init_key, theta, t0)
    args = (ts, ys) if observed is None else (ts, ys, observed)
    result = run(
        run_key, filt, initial, theta, len(ts), args=args, collect=("state", "info")
    )
    return initial, result


def kalman_log_likelihood(theta):
    return filter_run(kalman(), theta)[1].state.log_likelihood


def smc_run(kernel, state, key=KEY, ts=TS, ys=YS):
    return run(key, kernel, state, None, len(ts), args=(ts, ys), collect=("info",))


@pytest.mark.parametrize("mask", [None, jnp.array([True, False, True, False])])
def test_gaussian_likelihood_and_smoothing_match_joint(mask):
    model = gaussian_model()
    initial, result = jax.jit(lambda key: filter_run(kalman(), key=key, observed=mask))(
        KEY
    )
    states, infos = result.trace["state"], result.trace["info"]
    mean, cov, logz = exact_joint(mask)
    smoothed_mean, smoothed_cov = smooth(model, THETA, initial, states, infos)
    np.testing.assert_allclose(result.state.log_likelihood, logz, rtol=2e-6)
    np.testing.assert_allclose(smoothed_mean[:, 0], mean, atol=2e-6)
    np.testing.assert_allclose(smoothed_cov[:, 0, 0], np.diag(cov), atol=2e-6)
    samples = jax.jit(
        lambda key: sample_paths(
            key, model, THETA, initial, states, infos, num_samples=16000
        )
    )(KEY)[:, :, 0]
    np.testing.assert_allclose(samples.mean(1), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(samples), cov, atol=0.012)


@pytest.mark.parametrize("factory", FILTERS)
def test_run_matches_stepping_by_hand(factory):
    filt = factory()
    mask = jnp.array([True, False, True, True])
    ys = YS.at[1].set(jnp.nan)
    _, result = jax.jit(lambda key: filter_run(filt, key=key, observed=mask, ys=ys))(
        KEY
    )
    init_key, run_key = jax.random.split(KEY)
    state = filt.init(init_key, THETA, 0.0)
    keys = jax.random.split(run_key, (4, 1))[:, 0]
    for key, t, y, present in zip(keys, TS, ys, mask, strict=True):
        state, _ = jax.jit(filt.step)(key, state, THETA, t, y, present)
    assert_tree_close(result.state, state, rtol=1e-5, atol=1e-6)


def test_pf_increment_uses_normalized_weights():
    model = StateSpaceModel(
        init=lambda key, p: jnp.zeros(1),
        step=lambda key, p, x, a, b: x,
        log_likelihood=lambda p, x, y, t: jnp.array(-2.0),
    )
    filt = particle_filter(model, 8)
    _, info = filt.step(KEY, filt.init(KEY, None), None, 1.0, jnp.zeros(1))
    np.testing.assert_allclose(info.log_likelihood_increment, -2.0, atol=1e-6)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_pf_likelihood_estimate_is_unbiased(threshold):
    filt = particles(ess_threshold=threshold)

    def estimate(key):
        return filter_run(filt, key=key)[1].state.log_likelihood

    logz = jax.jit(jax.vmap(estimate))(jax.random.split(KEY, 512))
    np.testing.assert_allclose(
        jnp.exp(logz).mean(), jnp.exp(exact_joint()[2]), rtol=0.045
    )


def test_linear_operator_matches_dense_kalman():
    c = jnp.array([[1.0, 0.4]])
    operator = Function(lambda x: c @ x, 2)

    def make(obs):
        return kalman_filter(
            LinearGaussianModel(
                lambda p: (jnp.array([0.2, -0.1]), jnp.array([[2.0, 0.3], [0.3, 0.5]])),
                lambda p, old, new: (jnp.eye(2), jnp.eye(2) * 0.1),
                lambda p, t: (obs, jnp.eye(1) * 0.2),
            )
        )

    dense, lazy = make(c), make(operator)
    initial = dense.init(KEY, None)
    assert_tree_close(
        dense.step(KEY, initial, None, 1.0, jnp.array([0.4])),
        lazy.step(KEY, initial, None, 1.0, jnp.array([0.4])),
        atol=1e-6,
    )


def test_ancestry_and_backward_paths():
    model = particle_model()
    initial, result = filter_run(particles(256, ess_threshold=1.0))
    states, infos = result.trace["state"], result.trace["info"]
    for method in ("ancestry", "backward"):
        paths = jax.jit(
            lambda key, method=method: sample_paths(
                key, model, THETA, initial, states, infos, num_samples=500, method=method
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
    initial, r = filter_run(particle_filter(shift, 32, ess_threshold=1.0), None)
    paths = sample_paths(
        KEY, shift, None, initial, r.trace["state"], r.trace["info"], num_samples=100
    )
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


def _smc(filt, logprior=norm.logpdf, **options):
    return temporal_smc(filt, logprior, max_observations=len(TS), **options)


def test_outer_weights_equal_exact_parameter_mixture():
    theta = jnp.linspace(-3.0, 1.0, 40)
    kernel = _smc(kalman(), ess_threshold=0.0)
    result = jax.jit(lambda key, s: smc_run(kernel, s, key))(
        KEY, kernel.init(KEY, theta, YS[0])
    )
    likelihoods = jax.vmap(kalman_log_likelihood)(theta)
    np.testing.assert_allclose(
        result.state.log_weights, jax.nn.log_softmax(likelihoods), atol=2e-6
    )
    np.testing.assert_allclose(
        result.state.log_evidence,
        jax.scipy.special.logsumexp(likelihoods) - jnp.log(len(theta)),
        atol=2e-6,
    )
    assert result.trace["info"].valid.all()


@pytest.mark.parametrize("factory", FILTERS)
def test_rejected_moves_preserve_likelihood_and_filter_state(factory):
    plain = _smc(factory(), num_moves=0, ess_threshold=1.0)
    reject = _smc(
        factory(),
        num_moves=2,
        ess_threshold=1.0,
        proposal_fn=lambda key, p: (p + 1.0, -jnp.inf),
    )
    initial = plain.init(KEY, jnp.linspace(-3.0, 1.0, 16), YS[0])
    baseline, rejected = (
        jax.jit(lambda key, k=k: smc_run(k, initial, key))(KEY) for k in (plain, reject)
    )
    assert_tree_close(baseline.state, rejected.state, rtol=1e-6, atol=1e-6)
    assert rejected.trace["info"].resampled.any()
    assert (rejected.trace["info"].acceptance_rate == 0).all()


def test_moves_keep_exact_likelihoods_with_pytree_parameters():
    model = gaussian_model()
    # Parameters are a dict; the model reads its one entry.
    wrapped = LinearGaussianModel(
        lambda p: model.initial(p["scale"]),
        lambda p, a, b: model.transition(p["scale"], a, b),
        lambda p, t: model.observation(p["scale"], t),
    )
    kernel = _smc(
        kalman_filter(wrapped),
        lambda p: norm.logpdf(p["scale"]),
        ess_threshold=1.0,
        num_moves=2,
    )
    initial = kernel.init(KEY, {"scale": jnp.linspace(-3.0, 1.0, 16)}, YS[0])
    result = jax.jit(lambda key: smc_run(kernel, initial, key))(KEY)
    expected = jax.vmap(kalman_log_likelihood)(result.state.parameters["scale"])
    np.testing.assert_allclose(
        result.state.filter_states.log_likelihood, expected, atol=2e-6
    )
    assert result.trace["info"].acceptance_rate.max() > 0.0
    assert result.state.posterior_samples(KEY, 5)["scale"].shape == (5,)


def test_continuing_a_run_assimilates_the_rest():
    kernel = _smc(kalman(), ess_threshold=0.0)
    initial = kernel.init(KEY, jnp.linspace(-3.0, 1.0, 16), YS[0])
    batch = smc_run(kernel, initial)
    first = smc_run(kernel, initial, ts=TS[:2], ys=YS[:2])
    second = smc_run(kernel, first.state, ts=TS[2:], ys=YS[2:])
    np.testing.assert_allclose(
        second.state.log_weights, batch.state.log_weights, atol=2e-6
    )
    np.testing.assert_allclose(
        second.state.log_evidence, batch.state.log_evidence, atol=2e-6
    )
    assert int(second.state.num_observations) == 4
    # Times must advance: replaying old data is rejected.
    replay = smc_run(kernel, first.state, ts=TS[:2], ys=YS[:2])
    assert not replay.trace["info"].valid.any()
    assert_tree_close(replay.state, first.state)


def test_steps_beyond_the_buffer_are_rejected():
    kernel = temporal_smc(kalman(), norm.logpdf, max_observations=2)
    state = kernel.init(KEY, jnp.zeros(4), YS[0])
    result = smc_run(kernel, state)
    assert result.trace["info"].valid.tolist() == [True, True, False, False]
    assert int(result.state.num_observations) == 2


def test_pseudo_marginal_moves_preserve_size_biased_auxiliary_law():
    """An unbiased estimator U must have law proportional to U after weighting/MH."""

    class State(NamedTuple):
        value: object
        t: object
        log_likelihood: object

    class Info(NamedTuple):
        log_likelihood_increment: object

    def init(key, theta, t0):
        value = jnp.where(jax.random.bernoulli(key), 1.75, 0.25)
        return State(value, jnp.asarray(t0, float), jnp.zeros(()))

    def step(key, state, theta, t, y, observed=True):
        increment = jnp.log(state.value) - state.log_likelihood
        return State(state.value, t, jnp.log(state.value)), Info(increment)

    kernel = temporal_smc(
        Kernel(init, step, lambda s, p=None: p),
        lambda p: jnp.array(0.0),
        max_observations=1,
        ess_threshold=1.0,
        proposal_fn=lambda key, p: (p, 0.0),
        num_moves=3,
    )
    init_key, run_key = jax.random.split(KEY)
    initial = kernel.init(init_key, jnp.zeros(4096), jnp.zeros(1))
    result = jax.jit(
        lambda key: smc_run(kernel, initial, key, jnp.array([1.0]), jnp.zeros((1, 1)))
    )(run_key)
    # q(U=1.75)*1.75 / E_q[U] = 0.875, not the proposal probability 0.5.
    frequency = jnp.mean(result.state.filter_states.value == 1.75)
    np.testing.assert_allclose(frequency, 0.875, atol=0.025)
    assert result.trace["info"].resampled[0]


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
    _, result = filter_run(particle_filter(model, 64))
    assert result.state.particles["x"].shape == (64, 1)
    assert (result.state.particles["steps"] == 4).all()
    np.testing.assert_allclose(result.state.log_likelihood, exact_joint()[2], atol=0.3)


def test_missing_observation_only_predicts():
    filt = particles(512)
    initial = filt.init(KEY, THETA, 0.0)
    predicted, info = jax.jit(filt.step)(
        KEY, initial, THETA, TS[0], jnp.array([jnp.nan]), False
    )
    assert jnp.isfinite(predicted.particles).all()
    assert info.log_likelihood_increment == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError, match="time dimension"):
        particle_gibbs(KEY, jnp.zeros((5, 1)), particle_model(), THETA, TS, YS[:2])
    with pytest.raises(ValueError, match="boolean"):
        particle_gibbs(
            KEY, jnp.zeros((5, 1)), particle_model(), THETA, TS, YS, observed=jnp.ones(4)
        )
    with pytest.raises(ValueError, match="max_observations"):
        temporal_smc(kalman(), norm.logpdf, max_observations=0)
    with pytest.raises(ValueError, match="transition_logdensity"):
        particle_filter(
            particle_model()._replace(transition_logdensity=None),
            8,
            proposal=lambda *args: None,
        )


def test_single_state_paths():
    model = gaussian_model()
    filt = kalman()
    initial = filt.init(KEY, THETA)
    result = run(KEY, filt, initial, THETA, 0, args=(TS[:0], YS[:0]), collect=("state", "info"))
    assert result.state.log_likelihood == 0.0
    assert result.state.t == 0.0
    paths = sample_paths(
        KEY, model, THETA, initial, result.trace["state"], result.trace["info"]
    )
    assert paths.shape == (1, 1, 1)


def test_all_impossible_observations_report_failure():
    model = particle_model()._replace(
        log_likelihood=lambda p, x, y, t: jnp.array(-jnp.inf)
    )
    kernel = _smc(particle_filter(model, 8))
    state = kernel.init(KEY, jnp.zeros(4), YS[0])
    next_state, info = jax.jit(kernel.step)(KEY, state, None, TS[0], YS[0])
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
    kernel = _smc(particle_filter(model, 8), num_moves=0)
    state = kernel.init(KEY, jnp.array([-1.0, -0.5, 0.5, 1.0]), YS[0])
    next_state, info = jax.jit(kernel.step)(KEY, state, None, TS[0], YS[0])
    assert info.valid
    assert jnp.isfinite(next_state.log_evidence)
    assert (next_state.parameters[jnp.isfinite(next_state.log_weights)] < 0).all()


@pytest.mark.parametrize("factory", FILTERS)
def test_integer_initial_time_with_fractional_intervals(factory):
    filt = factory()
    _, result = jax.jit(lambda key: filter_run(filt, key=key, t0=0))(KEY)
    assert result.state.t == TS[-1]
    kernel = _smc(filt)
    state = kernel.init(KEY, jnp.zeros(4), YS[0], 0)
    result, info = jax.jit(kernel.step)(KEY, state, None, 1, YS[0])
    assert result.t == 1.0
    assert info.valid
