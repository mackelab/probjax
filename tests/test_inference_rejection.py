"""Exact samplers: ARS and rejection sampling on the kernel contract."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import scipy.stats as st

from probjax.inference import (
    Density,
    arms,
    ars,
    bound_adaptor,
    compose_adaptors,
    fit_adaptor,
    maximum_likelihood,
    rejection,
    run,
    warmup,
)
from probjax.inference.rejection import univariate


def _normal_logdensity(x):
    return -0.5 * ((x - 1.0) / 2.0) ** 2


def _gamma3_logdensity(x):
    return 2.0 * jnp.log(x) - x


def test_ars_draws_exact_samples_with_few_target_evaluations():
    kernel = ars(_normal_logdensity)
    state = kernel.init(jax.random.key(0), 0.0)

    result = run(
        jax.random.key(1),
        kernel,
        state,
        None,
        20_000,
        collect=("position", "num_evals"),
    )

    draws = np.asarray(result.trace["position"])
    assert st.kstest(draws, st.norm(1.0, 2.0).cdf).pvalue > 0.01
    # The envelope tightens, so almost every draw is accepted by the squeeze.
    assert int(result.trace["num_evals"].sum()) < 500


def test_ars_respects_a_finite_lower_bound():
    kernel = ars(_gamma3_logdensity)
    state = kernel.init(
        jax.random.key(0), 2.0, init_points=jnp.array([0.5, 2.0, 6.0]), lower=0.0
    )

    draws = np.asarray(
        run(jax.random.key(1), kernel, state, None, 20_000).trace["position"]
    )

    assert draws.min() > 0.0
    assert st.kstest(draws, st.gamma(3.0).cdf).pvalue > 0.01


def test_default_points_on_one_side_of_the_mode_are_moved_outwards():
    # All three default points lie at or left of the mode at 1.
    env = univariate.envelope(_normal_logdensity, jnp.array([-1.0, 0.0, 1.0]))

    assert float(env.grad[env.num_points - 1]) < 0.0
    assert jnp.isfinite(jax.scipy.special.logsumexp(env.log_mass))


def test_envelope_bounds_the_log_density_from_both_sides():
    env = univariate.envelope(_normal_logdensity, jnp.array([-3.0, 0.5, 4.0]))
    for x in (2.0, -1.0, 6.0):
        h, g = jax.value_and_grad(_normal_logdensity)(x)
        env = univariate.refine(env, jnp.asarray(x), h, g)

    grid = jnp.linspace(-10.0, 12.0, 400)
    target = _normal_logdensity(grid)
    upper = jax.vmap(lambda x: univariate.upper_hull(env, x))(grid)
    lower = jax.vmap(lambda x: univariate.squeeze(env, x))(grid)
    assert bool(jnp.all(upper >= target - 1e-4))
    assert bool(jnp.all(lower <= target + 1e-4))


def test_ars_runs_under_vmap():
    kernel = ars(_normal_logdensity)
    keys = jax.random.split(jax.random.key(0), 3)
    states = jax.vmap(lambda key: kernel.init(key, 0.0))(keys)

    result = jax.vmap(lambda key, state: run(key, kernel, state, None, 50))(
        keys, states
    )

    assert result.trace["position"].shape == (3, 50)


def test_arms_recovers_the_target_with_the_new_envelope():
    kernel = arms(_normal_logdensity)
    state = kernel.init(jax.random.key(0), 0.0)

    draws = run(jax.random.key(1), kernel, state, None, 20_000).trace["position"]

    assert float(draws.mean()) == pytest.approx(1.0, abs=0.15)
    assert float(draws.std()) == pytest.approx(2.0, abs=0.15)


def _gaussian_proposal(dim):
    def sample(key, params):
        return params["loc"] + params["scale"] * jax.random.normal(key, (dim,))

    def logpdf(x, params):
        return jax.scipy.stats.norm.logpdf(x, params["loc"], params["scale"]).sum()

    return Density(sample, logpdf)


MEAN = jnp.array([1.0, -1.0])
STD = jnp.array([1.0, 0.5])


def _target(x):
    return -0.5 * jnp.sum(((x - MEAN) / STD) ** 2)


def test_rejection_draws_independent_exact_samples_after_bound_warmup():
    proposal = _gaussian_proposal(2)
    kernel = rejection(_target, proposal=proposal)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(
        state, {"loc": jnp.zeros(2), "scale": 2.0 * jnp.ones(2)}
    )

    warm = warmup(jax.random.key(1), kernel, state, params, 2000)
    result = run(
        jax.random.key(2),
        kernel,
        warm.state,
        warm.params,
        10_000,
        collect=("position", "num_proposals"),
    )

    draws = result.trace["position"]
    assert jnp.allclose(draws.mean(0), MEAN, atol=0.05)
    assert jnp.allclose(draws.std(0), STD, atol=0.05)
    lag_one = jnp.corrcoef(draws[:-1, 0], draws[1:, 0])[0, 1]
    assert abs(float(lag_one)) < 0.05
    # Matches 1 / acceptance probability of the learned bound.
    assert 5.0 < float(result.trace["num_proposals"].mean()) < 20.0


def test_bound_adaptor_takes_the_largest_ratio_across_chains():
    proposal = _gaussian_proposal(2)
    kernel = rejection(_target, proposal=proposal)
    keys = jax.random.split(jax.random.key(0), 4)
    states = jax.vmap(lambda key: kernel.init(key, jnp.zeros(2)))(keys)
    params = kernel.init_params(
        states, {"loc": jnp.zeros(2), "scale": 2.0 * jnp.ones(2)}
    )

    result = jax.vmap(
        lambda key, state: warmup(
            key, kernel, state, params, 20, bound_adaptor(axis_name="chain")
        ),
        axis_name="chain",
    )(keys, states)

    bounds = result.params.log_bound
    assert jnp.all(bounds == bounds[0])


def test_rejection_trains_its_proposal_while_it_samples():
    proposal = _gaussian_proposal(2)
    kernel = rejection(_target, proposal=proposal)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(
        state, {"loc": jnp.zeros(2), "scale": 3.0 * jnp.ones(2)}
    )
    policy = compose_adaptors(
        proposal=fit_adaptor(
            maximum_likelihood(proposal),
            optax.adam(1e-2),
            field="proposal",
            batch_size=64,
        ),
        bound=bound_adaptor(),
    )

    warm = warmup(jax.random.key(1), kernel, state, params, 1500, policy)

    assert jnp.allclose(warm.params.proposal["loc"], MEAN, atol=0.3)
    assert float(warm.params.log_bound) > -jnp.inf
