from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from probjax.inference import (
    SMC,
    Kernel,
    gauss_rwmh,
    mala,
    pooled,
    run,
    step,
    step_size_adaptor,
    warmup,
)
from probjax.inference.mcmc.adaptation import covariance_adaptor


def _standard_normal(x):
    return -0.5 * jnp.sum(x**2)


def _mala_setup():
    kernel = mala(_standard_normal)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    return kernel, state, kernel.init_params(state)


def test_run_collects_state_and_info_fields():
    kernel, state, params = _mala_setup()

    result = run(
        jax.random.key(1),
        kernel,
        state,
        params,
        8,
        collect=("position", "acceptance_rate"),
        thin=2,
    )

    assert result.trace["position"].shape == (4, 2)
    assert result.trace["acceptance_rate"].shape == (4,)
    assert result.params == params


def test_run_rejects_steps_not_divisible_by_thin():
    kernel, state, params = _mala_setup()

    with pytest.raises(ValueError, match="multiple of thin"):
        run(jax.random.key(1), kernel, state, params, 5, thin=2)


def test_run_reports_unknown_collect_names():
    kernel, state, params = _mala_setup()

    with pytest.raises(AttributeError, match="not_a_field"):
        run(jax.random.key(1), kernel, state, params, 2, collect=("not_a_field",))


@pytest.mark.parametrize("progress", [True, ("acceptance_rate", "missing")])
def test_run_with_progress_matches_silent_run(progress, capsys):
    kernel, state, params = _mala_setup()

    silent = run(jax.random.key(1), kernel, state, params, 6)
    loud = run(jax.random.key(1), kernel, state, params, 6, progress=progress)
    jax.effects_barrier()

    assert jnp.array_equal(silent.trace["position"], loud.trace["position"])


def test_run_with_adaptor_returns_finalized_params():
    kernel, state, params = _mala_setup()

    result = run(
        jax.random.key(1), kernel, state, params, 50, adaptor=step_size_adaptor()
    )

    assert result.trace["position"].shape == (50, 2)
    assert jnp.isfinite(result.params.step_size)
    assert result.params.step_size != params.step_size


def test_step_applies_one_adaptor_update():
    kernel, state, params = _mala_setup()
    adaptor = step_size_adaptor()

    state, new_params, adaptor_state, info = step(
        jax.random.key(1),
        kernel,
        state,
        params,
        adaptor=adaptor,
        adaptor_state=adaptor.init(state, params),
    )

    assert info.acceptance_rate.shape == ()
    assert new_params.step_size != params.step_size


def test_warmup_accepts_adaptor_default_and_rejects_others():
    kernel, state, params = _mala_setup()

    explicit = warmup(jax.random.key(1), kernel, state, params, 10, step_size_adaptor())
    defaulted = warmup(jax.random.key(1), kernel, state, params, 10)

    assert jnp.isfinite(explicit.params.step_size)
    assert jnp.isfinite(defaulted.params.step_size)
    with pytest.raises(TypeError, match="Adaptor or a Warmup"):
        warmup(jax.random.key(1), kernel, state, params, 10, policy="nuts")


def test_pooled_adaptor_gives_all_chains_one_step_size():
    kernel, _, _ = _mala_setup()
    starts = jax.random.normal(jax.random.key(0), (4, 2)) * 3.0
    states = jax.vmap(lambda x: kernel.init(None, x))(starts)
    params = jax.vmap(kernel.init_params)(states)
    keys = jax.random.split(jax.random.key(1), 4)

    def adapt(adaptor):
        return jax.vmap(
            lambda key, state, params: run(
                key, kernel, state, params, 30, adaptor=adaptor, collect=()
            ),
            axis_name="chain",
        )(keys, states, params)

    independent = adapt(step_size_adaptor()).params.step_size
    shared = adapt(pooled(step_size_adaptor())).params.step_size

    assert not jnp.allclose(independent, independent[0])
    assert jnp.allclose(shared, shared[0])


def test_pooled_covariance_sees_spread_between_chains():
    # Chains frozen at their starts (zero step size): each chain alone has no
    # spread, so only pooling can recover the spread between them.
    kernel = gauss_rwmh(_standard_normal)
    starts = jnp.array([[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [2.0, 0.0]])
    states = jax.vmap(lambda x: kernel.init(None, x))(starts)
    params = jax.vmap(lambda s: kernel.init_params(s, step_size=0.0))(states)
    keys = jax.random.split(jax.random.key(1), 4)

    result = jax.vmap(
        lambda key, state, params: run(
            key,
            kernel,
            state,
            params,
            200,
            adaptor=pooled(covariance_adaptor(output="variance")),
            collect=(),
        ),
        axis_name="chain",
    )(keys, states, params)

    assert jnp.allclose(result.params.scale[:, 0], 4.0, rtol=0.05)
    assert jnp.allclose(result.params.scale[:, 1], 0.0, atol=1e-6)


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
