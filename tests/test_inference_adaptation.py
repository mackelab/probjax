from typing import NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.adaptation import compose_adaptors
from probjax.inference.mcmc import gauss_rwmh, hmc, mala
from probjax.inference.mcmc.adaptation import (
    covariance_adaptor,
    mass_matrix_adaptor,
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
