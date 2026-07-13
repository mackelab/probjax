from typing import NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference import (
    MCMC,
    SMC,
    Kernel,
    mala,
    step_size_adaptor,
)


def test_mcmc_runners_collect_requested_outputs():
    kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)

    runner = MCMC(
        kernel,
        collect_state=("position",),
        collect_info=("acceptance_rate",),
    )
    run = runner.run(jax.random.key(1), state, 3, params)
    sampler = MCMC(
        kernel,
        collect_info=("acceptance_rate",),
    )
    samples = sampler.sample(jax.random.key(2), state, 4, params, thin=2)

    assert run.samples["position"].shape == (3, 2)
    assert run.info["acceptance_rate"].shape == (3,)
    assert samples.samples.shape == (4, 2)
    assert samples.info["acceptance_rate"].shape == (4,)


def test_mcmc_static_runner_primitive():
    kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)

    result = MCMC.run_kernel(
        jax.random.key(1),
        kernel,
        state,
        3,
        params,
        collect_info=("acceptance_rate",),
    )

    assert result.state.position.shape == (2,)
    assert result.info["acceptance_rate"].shape == (3,)


def test_step_size_adaptor_is_a_compiled_state_machine():
    kernel = mala(lambda x: -0.5 * jnp.sum(x**2))
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)

    result = MCMC(kernel).adapt(
        jax.random.key(1), step_size_adaptor(), state, params, 5
    )

    assert result.state.position.shape == (2,)
    assert jnp.isfinite(result.params.step_size)


class _SMCState(NamedTuple):
    particles: jax.Array
    weights: jax.Array


class _SMCInfo(NamedTuple):
    increment: jax.Array


def test_smc_static_runner_primitive():
    def step(key, state, tempering_param, mcmc_parameters):
        del key
        increment = tempering_param * mcmc_parameters["scale"]
        state = _SMCState(state.particles + increment, state.weights)
        return state, _SMCInfo(increment)

    kernel = Kernel(lambda particles: particles, step, lambda state: {})
    state = _SMCState(jnp.zeros((4, 1)), jnp.full(4, 0.25))
    schedule = jnp.array([0.25, 0.5, 1.0])

    result = SMC.run_kernel(
        jax.random.key(0),
        kernel,
        state,
        schedule,
        {"scale": jnp.array(2.0)},
        collect=True,
    )

    states, info = result.info
    assert states.particles.shape == (3, 4, 1)
    assert jnp.array_equal(info.increment, schedule * 2.0)
