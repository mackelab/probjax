from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.inference.base import Kernel
from probjax.inference.mcmc import (
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    dynamic_hmc,
    elliptical_slice,
    gauss_rwmh,
    gaussian_imh,
    hmc,
    latent_slice,
    mala,
    mclmc,
    nuts,
    sghmc,
    sgld,
    sgnht,
    slice,
    step_size_adaptor,
)
from probjax.inference.mcmc.pmmcmc import pseudo_marginal
from probjax.inference.mcmc.sgmcmc import grad_estimator
from probjax.inference.runner import run, warmup

KERNELS = [
    hmc,
    nuts,
    mclmc,
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    gaussian_imh,
    gauss_rwmh,
    mala,
    dynamic_hmc,
    slice,
    latent_slice,
    elliptical_slice,
]
SHAPES = [(1,), (2,), (2, 3), (4, 5, 6)]


@pytest.mark.parametrize(
    "kernel_type,in_shape", [(kernel, s) for kernel in KERNELS for s in SHAPES]
)
def test_markov_kernel_vector_input(kernel_type, in_shape):
    if kernel_type is mclmc and in_shape == (1,):
        pytest.skip("mclmc does not support 1-d input")
    rng = np.random.default_rng(0)
    i = int(rng.integers(0, 2**16))
    xs = rng.standard_normal(in_shape)

    def logdensity(x):
        return -jnp.sum(x**2)

    if kernel_type is elliptical_slice:
        kernel = kernel_type(
            logdensity, cov_matrix=jnp.eye(xs.size), mean=jnp.zeros(xs.size)
        )
    else:
        kernel = kernel_type(logdensity)
    state = kernel.init(jax.random.PRNGKey(i), xs)
    params = kernel.init_params(state)

    assert hasattr(state, "position"), "State must have a position attribute"
    assert state.position.shape == in_shape, "Position shape must match input shape"

    next_state, next_info = kernel(jax.random.PRNGKey(i), state, params)

    assert hasattr(next_state, "position"), "State must have a position attribute"
    assert next_state.position.shape == in_shape, (
        "Position shape must match input shape after transition"
    )


@pytest.mark.parametrize("kernel_type", [hmc, nuts, mala, gauss_rwmh])
def test_markov_kernel_adaptation_sanity(kernel_type):
    key = jax.random.PRNGKey(0)
    x0 = jnp.array([0.1, -0.2])

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    if kernel_type is hmc:
        kernel = kernel_type(logdensity, num_integration_steps=5)
    elif kernel_type is nuts:
        kernel = kernel_type(logdensity, max_num_doublings=5)
    else:
        kernel = kernel_type(logdensity)
    state = kernel.init(None, x0)
    params = kernel.init_params(state)

    result = warmup(key, kernel, state, params, 10, policy=step_size_adaptor())
    new_state, new_params = result.state, result.params

    assert hasattr(new_state, "position")
    assert new_state.position.shape == x0.shape
    assert isinstance(new_params, type(params))


@pytest.mark.parametrize(
    "kernel_type,in_shape",
    [(kernel, s) for kernel in KERNELS for s in [(1,), (2,), (2, 2)]],
)
def test_markov_kernel_invariance(kernel_type, in_shape):
    if kernel_type is mclmc:
        pytest.skip("mclmc is not invariant in this setup")

    rng = np.random.default_rng(1)
    i = int(rng.integers(0, 2**16))
    key = jax.random.PRNGKey(i)

    N = 5000

    positions = rng.standard_normal((N, *in_shape))
    if kernel_type is elliptical_slice:
        # Elliptical slice uses a Gaussian prior; use a flat likelihood so
        # the target is the prior N(0, I).
        def logdensity(x):
            return jnp.array(0.0)

        kernel = kernel_type(
            logdensity,
            cov_matrix=jnp.eye(positions[0].size),
            mean=jnp.zeros(positions[0].size),
        )
    else:
        # Gaussian target, hence must leave the particle distribution invariant
        def logdensity(x):
            return jax.scipy.stats.norm.logpdf(x).sum()

        kernel = kernel_type(logdensity)
    states = jax.vmap(lambda x: kernel.init(key, x))(positions)
    state0 = jax.tree_util.tree_map(lambda x: x[0], states)
    if kernel_type is gaussian_imh:
        flat_position, _ = jax.flatten_util.ravel_pytree(state0.position)
        dim = flat_position.shape[0]
        params = kernel.init_params(
            state0,
            mean=jnp.zeros_like(flat_position),
            cov=jnp.eye(dim, dtype=flat_position.dtype),
        )
    else:
        params = kernel.init_params(state0)

    keys = jax.random.split(key, (N,))
    next_states, _ = jax.vmap(kernel, in_axes=(0, 0, None))(keys, states, params)

    new_positions = next_states.position

    assert new_positions.shape == positions.shape, "Output shape must match input shape"

    # Must be invariant to input
    mean_before = jnp.mean(positions, axis=0)
    mean_after = jnp.mean(new_positions, axis=0)

    assert jnp.allclose(mean_before, mean_after, atol=1e-1, rtol=1e-1), (
        "Mean must be invariant"
    )

    var_before = jnp.var(positions, axis=0)
    var_after = jnp.var(new_positions, axis=0)

    assert jnp.allclose(var_before, var_after, atol=1e-1, rtol=1e-1), (
        "Variance must be invariant"
    )


# ---------------------------------------------------------------------------
# SG-MCMC tests — minibatch via *args
# ---------------------------------------------------------------------------


def _make_sgmcmc_problem():
    """Simple Bayesian linear regression for SG-MCMC tests."""
    data_size = 100
    batch_size = 10

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (data_size, 2))
    y = X @ jnp.array([1.0, -0.5]) + 0.1 * jax.random.normal(k2, (data_size,))

    def logprior_fn(position):
        return -0.5 * jnp.sum(position**2)

    def loglikelihood_fn(position, minibatch):
        X_batch, y_batch = minibatch
        preds = X_batch @ position
        return -0.5 * jnp.sum((preds - y_batch) ** 2)

    ge = grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    # Pre-generate batches as stacked arrays
    indices = jnp.arange(data_size)
    batch_indices = indices.reshape(-1, batch_size)  # (10, 10)
    X_batches = X[batch_indices]  # (10, 10, 2)
    y_batches = y[batch_indices]  # (10, 10)
    # Wrap in a 1-tuple so scan slices correctly and *args unpacks to one
    # positional argument (the minibatch tuple) for the kernel.
    batches = ((X_batches, y_batches),)

    return ge, batches, data_size


def _get_position(state):
    """Extract position from state. All states now have .position."""
    return state.position


@pytest.mark.parametrize("kernel_fn", [sgld, sghmc, sgnht])
def test_sgmcmc_step_with_args(kernel_fn):
    """SG-MCMC step accepts minibatch via *args (not params)."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(0)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(key, position)
    elif kernel_fn is sghmc:
        sampler = kernel_fn(ge, num_integration_steps=2)
        state = sampler.init(None, position)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(None, position)

    params = sampler.init_params(state)

    # One step with a single minibatch: unwrap the outer 1-tuple, take first slice
    single_batch = jax.tree_util.tree_map(lambda x: x[0], batches[0])
    new_state, info = sampler.step(key, state, params, single_batch)

    new_pos = _get_position(new_state)
    old_pos = _get_position(state)
    assert new_pos.shape == position.shape
    assert not jnp.allclose(new_pos, old_pos), "Position should change after a step"


@pytest.mark.parametrize("kernel_fn", [sgld, sghmc, sgnht])
def test_sgmcmc_constructor_temperature_sets_default_params(kernel_fn):
    ge, _, _ = _make_sgmcmc_problem()
    sampler = kernel_fn(ge, temperature=0.5)
    state = sampler.init(jax.random.PRNGKey(0), jnp.zeros(2))

    assert sampler.init_params(state).temperature == 0.5
    assert sampler.init_params(state, temperature=2.0).temperature == 2.0


@pytest.mark.parametrize("kernel_fn", [sgld, sghmc, sgnht])
def test_sgmcmc_runner_run(kernel_fn):
    """run works with SG-MCMC kernels when args= is provided."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(1)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(key, position)
    elif kernel_fn is sghmc:
        sampler = kernel_fn(ge, num_integration_steps=2)
        state = sampler.init(None, position)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(None, position)

    params = sampler.init_params(state)
    num_steps = batches[0][0].shape[0]  # 10

    final_state = run(key, sampler, state, params, num_steps, args=batches).state

    final_pos = _get_position(final_state)
    assert final_pos.shape == position.shape
    assert not jnp.allclose(final_pos, position), "Position should change after running"


def test_runner_run_without_args_unchanged():
    """run still works for standard kernels without args."""
    key = jax.random.PRNGKey(0)
    position = jnp.array([0.1, -0.2])

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    kernel = mala(logdensity)
    state = kernel.init(None, position)
    params = kernel.init_params(state)

    final_state = run(key, kernel, state, params, 10).state

    assert final_state.position.shape == position.shape


class _KeyRecordingState(NamedTuple):
    position: jax.Array


class _KeyRecordingInfo(NamedTuple):
    pass


def test_runner_thinning_advances_key_past_inner_scan():
    """Thinned sampling must not reuse keys consumed by the previous sample."""

    def init(rng, position):
        return _KeyRecordingState(position)

    def step(rng, state, params, *args):
        return (
            _KeyRecordingState(jnp.stack([state.position[1], rng])),
            _KeyRecordingInfo(),
        )

    kernel = Kernel(
        init=init,
        step=step,
        init_params=lambda state: None,
    )

    initial_position = jnp.zeros((2, 2), dtype=jnp.uint32)
    state = kernel.init(None, initial_position)
    result = run(
        jax.random.PRNGKey(0), kernel, state, kernel.init_params(state), 8, thin=2
    )

    samples = result.trace["position"]
    assert not jnp.any(jnp.all(samples[:-1, 1] == samples[1:, 0], axis=1))


@pytest.mark.parametrize("kernel_fn", [sgld, sgnht])
def test_sgmcmc_runner_thinned(kernel_fn):
    """Thinned run works with SG-MCMC kernels when args= is provided."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(2)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(key, position)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(None, position)
    params = sampler.init_params(state)

    num_samples = 5
    thin = 2
    # Need num_samples * thin = 10 batches — slice the inner arrays
    total_batches = jax.tree_util.tree_map(lambda x: x[: num_samples * thin], batches)

    result = run(
        key, sampler, state, params, num_samples * thin, thin=thin, args=total_batches
    )

    assert result.trace["position"].shape == (num_samples, 2)
    assert result.state.position.shape == position.shape


# ---------------------------------------------------------------------------
# Pseudo-marginal MCMC tests
# ---------------------------------------------------------------------------


def _stochastic_logdensity(position, rng_key):
    """True Gaussian log-density + small additive noise."""
    true_logd = -0.5 * jnp.sum(position**2)
    noise = 0.1 * jax.random.normal(rng_key)
    return true_logd + noise


@pytest.mark.parametrize("inner_cls", [gauss_rwmh, mala])
def test_pseudo_marginal_step(inner_cls):
    """Pseudo-marginal step produces a new state with correct shape."""
    key = jax.random.PRNGKey(0)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(inner_cls, _stochastic_logdensity)
    state = kernel.init(key, position)
    params = kernel.init_params(state)

    new_state, info = kernel.step(key, state, params)

    assert new_state.position.shape == position.shape
    # logdensity is stored in state (noisy estimate)
    assert jnp.isfinite(new_state.logdensity)


def test_pseudo_marginal_hmc():
    """Pseudo-marginal with HMC (gradient-based, multiple logdensity evals)."""
    key = jax.random.PRNGKey(1)
    position = jnp.zeros(3)

    kernel = pseudo_marginal(hmc, _stochastic_logdensity, num_integration_steps=5)
    state = kernel.init(key, position)
    params = kernel.init_params(state)

    new_state, info = kernel.step(key, state, params)

    assert new_state.position.shape == position.shape
    assert jnp.isfinite(new_state.logdensity)


def test_pseudo_marginal_num_samples():
    """num_samples > 1 averages multiple stochastic estimates."""
    key = jax.random.PRNGKey(2)
    position = jnp.array([0.5, -0.3])

    kernel_1 = pseudo_marginal(gauss_rwmh, _stochastic_logdensity, num_samples=1)
    kernel_10 = pseudo_marginal(gauss_rwmh, _stochastic_logdensity, num_samples=10)

    state_1 = kernel_1.init(key, position)
    state_10 = kernel_10.init(key, position)

    # We can't assert exact values due to randomness, but both should be finite
    assert jnp.isfinite(state_1.logdensity)
    assert jnp.isfinite(state_10.logdensity)

    # Verify both can step
    params = kernel_10.init_params(state_10)
    new_state, _ = kernel_10.step(key, state_10, params)
    assert new_state.position.shape == position.shape


def test_pseudo_marginal_mcmc_runner():
    """Pseudo-marginal works with the runner (jax.lax.scan)."""
    key = jax.random.PRNGKey(3)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(gauss_rwmh, _stochastic_logdensity)
    state = kernel.init(key, position)
    params = kernel.init_params(state)

    final_state = run(key, kernel, state, params, 50).state

    assert final_state.position.shape == position.shape
    assert not jnp.allclose(final_state.position, position), (
        "Chain should move from the origin"
    )


def test_pseudo_marginal_mcmc_runner_hmc():
    """Pseudo-marginal HMC works with the runner."""
    key = jax.random.PRNGKey(4)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(hmc, _stochastic_logdensity, num_integration_steps=3)
    state = kernel.init(key, position)
    params = kernel.init_params(state)

    final_state = run(key, kernel, state, params, 20).state

    assert final_state.position.shape == position.shape
    assert jnp.isfinite(final_state.logdensity)


