from typing import NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.adaptation import compose_adaptors
from probjax.inference.base import Adaptor, Warmup
from probjax.inference.mcmc import arms, gauss_rwmh, gaussian_imh, hmc, mala
from probjax.inference.mcmc.adaptation import (
    covariance_adaptor,
    mass_matrix_adaptor,
    mean_adaptor,
    step_size_adaptor,
    window_warmup,
)
from probjax.inference.runner import run, step, warmup


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

    state, params, adaptor_state, kernel_info = step(
        jax.random.key(1),
        kernel,
        state,
        params,
        adaptor=adaptor,
        adaptor_state=adaptor_state,
    )

    assert state.position.shape == (2,)
    assert jnp.isfinite(params.step_size)
    assert kernel_info.acceptance_rate.shape == ()


def test_composed_adaptors_update_step_size_and_scale():
    kernel = gauss_rwmh(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    adaptor = compose_adaptors(
        step_size=step_size_adaptor(target=0.234),
        scale=covariance_adaptor(),
    )

    result = warmup(jax.random.key(1), kernel, state, params, 25, adaptor)

    assert jnp.isfinite(result.params.step_size)
    assert result.params.scale.shape == (2,)
    assert jnp.all(jnp.isfinite(result.params.scale))
    assert result.trace is None


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

    result = warmup(jax.random.key(1), kernel, state, params, 30, window_warmup())

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


def test_warmup_with_adaptor_matches_adaptive_run():
    kernel = mala(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)

    warm = warmup(jax.random.key(1), kernel, state, params, 10, step_size_adaptor())
    adapted = run(
        jax.random.key(1), kernel, state, params, 10, adaptor=step_size_adaptor()
    )

    assert isinstance(kernel.default_adaptation(), Adaptor)
    assert float(warm.params.step_size) == float(adapted.params.step_size)


def test_warmup_uses_kernel_default_when_omitted():
    kernel = hmc(_logdensity, num_integration_steps=3)
    state = kernel.init(jax.random.key(0), jnp.ones(2))
    params = kernel.init_params(state, step_size=0.1)

    assert isinstance(kernel.default_adaptation(), Warmup)
    defaulted = warmup(jax.random.key(1), kernel, state, params, 30)
    explicit = warmup(jax.random.key(1), kernel, state, params, 30, window_warmup())

    assert float(defaulted.params.step_size) == float(explicit.params.step_size)


def test_warmup_without_default_raises_informatively():
    kernel = arms(lambda x: -0.5 * x**2)
    state = kernel.init(jax.random.key(0), jnp.array(0.0))
    params = kernel.init_params(state)

    assert kernel.default_adaptation is None
    try:
        warmup(jax.random.key(1), kernel, state, params, 5)
    except ValueError as error:
        assert "no default adaptation" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_gaussian_imh_default_adaptation_tunes_mean_and_cov():
    kernel = gaussian_imh(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state, cov=50.0 * jnp.ones(2))

    result = warmup(jax.random.key(1), kernel, state, params, 10)

    assert result.params.mean.shape == (2,)
    assert result.params.cov.shape == (2,)
    assert jnp.all(jnp.isfinite(result.params.mean))
    assert jnp.all(jnp.isfinite(result.params.cov))


def test_pathfinder_warmup_runs_through_warmup():
    import blackjax

    from probjax.inference import nuts, pathfinder_warmup

    kernel = nuts(_logdensity)
    state = kernel.init(jax.random.key(0), jnp.ones(2))
    params = kernel.init_params(state)

    result = warmup(
        jax.random.key(1),
        kernel,
        state,
        params,
        50,
        pathfinder_warmup(blackjax.nuts, _logdensity),
    )

    assert jnp.isfinite(result.params.step_size)
    assert result.params.inverse_mass_matrix.shape == (2, 2)


def test_mclmc_default_warmups_recover_the_target_scale():
    from probjax.inference import adjusted_mclmc, adjusted_mclmc_dynamic, mclmc

    scales = jnp.array([1.0, 3.0])

    def logdensity(x):
        return -0.5 * jnp.sum((x / scales) ** 2)

    for api in (mclmc, adjusted_mclmc, adjusted_mclmc_dynamic):
        kernel = api(logdensity)
        state = kernel.init(jax.random.key(0), jnp.ones(2))
        warm = warmup(jax.random.key(1), kernel, state, kernel.init_params(state), 500)
        draws = run(jax.random.key(2), kernel, warm.state, warm.params, 2000)
        std = draws.trace["position"].std(0)
        assert jnp.allclose(std, scales, rtol=0.25), (api.__name__, std)


def _banana(z):
    return jax.scipy.stats.norm.logpdf(z[0], 0.0, 2.0) + jax.scipy.stats.norm.logpdf(
        z[1], 0.25 * z[0] ** 2 - 1.0, 0.5
    )


def test_slice_default_adaptation_settles_on_a_finite_width():
    from probjax.inference import slice

    kernel = slice(_banana)
    state = kernel.init(jax.random.key(0), jnp.array([0.0, -1.0]))
    warm = warmup(jax.random.key(1), kernel, state, kernel.init_params(state), 1000)
    out = run(
        jax.random.key(2),
        kernel,
        warm.state,
        warm.params,
        1000,
        collect=("position", "num_evals", "num_expansions"),
    )

    assert 0.1 < warm.params.step_size < 20.0
    assert jnp.all(jnp.isfinite(out.trace["position"]))
    expanding = out.trace["num_expansions"].sum() / out.trace["num_evals"].sum()
    assert 0.35 < expanding < 0.65


def test_slice_rejects_nan_densities_and_accepts_integer_step_sizes():
    from probjax.inference import slice

    # NaN outside the unit disc, like an overflowing density would produce.
    def logdensity(x):
        return jnp.where(jnp.sum(x**2) < 1.0, 0.0, jnp.nan)

    kernel = slice(logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state, step_size=2)
    out = run(jax.random.key(1), kernel, state, params, 500)

    assert jnp.all(jnp.sum(out.trace["position"] ** 2, -1) < 1.0)


def test_latent_slice_keeps_its_log_density_when_shrinking_runs_out():
    from probjax.inference.mcmc.latent_slice import latent_slice

    kernel = latent_slice(_banana, max_evals=1)
    state = kernel.init(None, jnp.array([0.0, -1.0]))
    out = run(
        jax.random.key(1),
        kernel,
        state,
        kernel.init_params(state),
        200,
        collect=("position", "logdensity"),
    )

    recomputed = jax.vmap(_banana)(out.trace["position"])
    assert jnp.allclose(out.trace["logdensity"], recomputed, atol=1e-5)


def test_gauss_rwmh_default_warmup_hits_its_acceptance_target():
    kernel = gauss_rwmh(_banana)
    state = kernel.init(None, jnp.array([0.0, -1.0]))
    warm = warmup(jax.random.key(1), kernel, state, kernel.init_params(state), 4000)
    out = run(
        jax.random.key(2),
        kernel,
        warm.state,
        warm.params,
        4000,
        collect=("acceptance_rate",),
    )

    assert abs(float(out.trace["acceptance_rate"].mean()) - 0.234) < 0.06
