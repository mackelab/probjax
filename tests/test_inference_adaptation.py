from typing import NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.adaptation import as_warmup, compose_adaptors
from probjax.inference.base import Adaptor, Warmup
from probjax.inference.mcmc import arms, gaussian_imh, gauss_rwmh, hmc, mala
from probjax.inference.mcmc.adaptation import (
    covariance_adaptor,
    mass_matrix_adaptor,
    mean_adaptor,
    step_size_adaptor,
    window_warmup,
)
from probjax.inference.mcmc_runner import MCMC
from probjax.inference.smc.tuning import acceptance_rate_adaptor


def _logdensity(position):
    return -0.5 * jnp.sum(position**2)


def test_step_size_adaptor_supports_mapping_params():
    class Info(NamedTuple):
        acceptance_rate: jax.Array

    adaptor = step_size_adaptor()
    params = {"step_size": jnp.array(0.1)}
    state = adaptor.init(None, params)
    state, params, info = adaptor.update(None, Info(jnp.array(0.9)), state, params)
    params, final_info = adaptor.finalize(state, params)

    assert params["step_size"].shape == ()
    assert jnp.isfinite(params["step_size"])
    assert info is None
    assert final_info is None


def test_local_adaptor_step_updates_the_next_transition_parameters():
    kernel = mala(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    adaptor = step_size_adaptor()
    adaptor_state = adaptor.init(state, params)

    state, params, adaptor_state, kernel_info, adaptor_info = MCMC.adapt_step(
        jax.random.key(1), kernel, adaptor, state, params, adaptor_state
    )

    assert state.position.shape == (2,)
    assert jnp.isfinite(params.step_size)
    assert kernel_info.acceptance_rate.shape == ()
    assert adaptor_info is None


def test_composed_adaptors_update_step_size_and_scale():
    kernel = gauss_rwmh(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    adaptor = compose_adaptors(
        step_size=step_size_adaptor(target=0.234),
        scale=covariance_adaptor(),
    )

    result = MCMC(kernel).adapt(jax.random.key(1), adaptor, state, params, 25)

    assert jnp.isfinite(result.params.step_size)
    assert result.params.scale.shape == (2,)
    assert jnp.all(jnp.isfinite(result.params.scale))
    assert result.trace is None
    assert result.final_info is None


def test_mass_matrix_adaptor_diagonal_and_dense_shapes():
    kernel = mala(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))

    class Params(NamedTuple):
        inverse_mass_matrix: jax.Array

    for diagonal, expected_shape in ((True, (2,)), (False, (2, 2))):
        params = Params(jnp.ones(2) if diagonal else jnp.eye(2))
        adaptor = mass_matrix_adaptor(diagonal=diagonal)
        adaptor_state = adaptor.init(state, params)
        for value in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])):
            state = state._replace(position=value)
            adaptor_state, params, _ = adaptor.update(
                state, None, adaptor_state, params
            )
        params, _ = adaptor.finalize(adaptor_state, params)
        assert params.inverse_mass_matrix.shape == expected_shape


def test_window_warmup_ports_blackjax_window_state_machine():
    kernel = hmc(_logdensity, num_integration_steps=3)
    state = kernel.init(jax.random.key(0), jnp.ones(2))
    params = kernel.init_params(state, step_size=0.1)

    result = MCMC(kernel).warmup(jax.random.key(1), window_warmup(), state, params, 30)

    assert jnp.isfinite(result.params.step_size)
    assert result.params.inverse_mass_matrix.shape == (2,)
    assert jnp.all(jnp.isfinite(result.params.inverse_mass_matrix))


def test_mean_adaptor_tracks_running_mean():
    class State(NamedTuple):
        position: jax.Array

    adaptor = mean_adaptor()
    params = {"mean": jnp.zeros(2)}
    adaptor_state = adaptor.init(State(jnp.zeros(2)), params)
    for value in (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])):
        adaptor_state, params, _ = adaptor.update(
            State(value), None, adaptor_state, params
        )
    params, _ = adaptor.finalize(adaptor_state, params)

    assert params["mean"].shape == (2,)
    assert jnp.allclose(params["mean"], jnp.array([2.0, 3.0]))


def test_warmup_accepts_bare_adaptor_like_as_warmup():
    kernel = mala(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    runner = MCMC(kernel, verbose=False)

    auto = runner.warmup(jax.random.key(1), step_size_adaptor(), state, params, 10)
    explicit = runner.warmup(
        jax.random.key(1), as_warmup(step_size_adaptor()), state, params, 10
    )

    assert isinstance(kernel.default_adaptation(), Adaptor)
    assert float(auto.params.step_size) == float(explicit.params.step_size)


def test_warmup_uses_kernel_default_when_omitted():
    kernel = hmc(_logdensity, num_integration_steps=3)
    state = kernel.init(jax.random.key(0), jnp.ones(2))
    params = kernel.init_params(state, step_size=0.1)
    runner = MCMC(kernel, verbose=False)

    assert isinstance(kernel.default_adaptation(), Warmup)
    defaulted = runner.warmup(jax.random.key(1), state=state, params=params, num_steps=30)
    explicit = runner.warmup(
        jax.random.key(1), window_warmup(), state, params, 30
    )

    assert float(defaulted.params.step_size) == float(explicit.params.step_size)


def test_warmup_without_default_raises_informatively():
    kernel = arms(lambda x: -0.5 * x**2)
    state = kernel.init(jax.random.key(0), jnp.array(0.0))
    params = kernel.init_params(state)

    assert kernel.default_adaptation is None
    try:
        MCMC(kernel, verbose=False).warmup(
            jax.random.key(1), state=state, params=params, num_steps=5
        )
    except ValueError as error:
        assert "no default adaptation" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_gaussian_imh_default_adaptation_tunes_mean_and_cov():
    kernel = gaussian_imh(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state, cov=50.0 * jnp.ones(2))

    result = MCMC(kernel, verbose=False).warmup(
        jax.random.key(1), state=state, params=params, num_steps=10
    )

    assert result.params.mean.shape == (2,)
    assert result.params.cov.shape == (2,)
    assert jnp.all(jnp.isfinite(result.params.mean))
    assert jnp.all(jnp.isfinite(result.params.cov))


def test_smc_acceptance_adaptor_preserves_shared_shape():
    class UpdateInfo(NamedTuple):
        acceptance_rate: jax.Array

    class Info(NamedTuple):
        update_info: UpdateInfo

    adaptor = acceptance_rate_adaptor()
    params = {"step_size": jnp.array([[0.1]])}
    state = adaptor.init(None, params)
    state, params, _ = adaptor.update(
        None,
        Info(UpdateInfo(jnp.array([0.5, 0.6]))),
        state,
        params,
    )

    assert params["step_size"].shape == (1, 1)
