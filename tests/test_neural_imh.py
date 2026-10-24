"""Learned proposals: the Density bridge, imh on a flow, and fit_adaptor."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probjax.inference import (
    fit_adaptor,
    imh,
    maximum_likelihood,
    mclmc,
    pseudo_marginal,
    run,
    warmup,
)
from probjax.inference.mcmc.imh import IMHInfo, imh_data
from probjax.inference.neural import density
from probjax.nn import LinearFlow, maf


def _normal_logdensity(position):
    return jax.scipy.stats.norm.logpdf(position, 1.5, 0.7).sum()


def _flow_imh():
    proposal, params = density(maf(1, 1, rngs=nnx.Rngs(0)))
    return proposal, params, imh(_normal_logdensity, proposal=proposal)


def test_imh_on_a_flow_runs_under_jit_and_vmap():
    _, params, kernel = _flow_imh()
    keys = jax.random.split(jax.random.key(0), 3)
    states = jax.vmap(lambda key: kernel.init(key, jnp.zeros(1)))(keys)

    result = jax.vmap(
        lambda key, state: run(
            key, kernel, state, params, 8, collect=("position", "acceptance_rate")
        )
    )(keys, states)

    assert result.trace["position"].shape == (3, 8, 1)
    assert jnp.all(jnp.isfinite(result.trace["position"]))


def test_default_adaptation_trains_the_proposal_towards_the_target():
    proposal, params, kernel = _flow_imh()
    state = kernel.init(jax.random.key(0), jnp.array([1.5]))

    warm = warmup(jax.random.key(1), kernel, state, params, 2000)

    def mean_acceptance(params):
        trace = run(
            jax.random.key(2),
            kernel,
            warm.state,
            params,
            500,
            collect=("acceptance_rate",),
        ).trace
        return float(trace["acceptance_rate"].mean())

    assert mean_acceptance(warm.params) > mean_acceptance(params) + 0.2
    draws = jax.vmap(proposal.sample, (0, None))(
        jax.random.split(jax.random.key(3), 4000), warm.params
    )
    assert float(draws.mean()) == pytest.approx(1.5, abs=0.25)
    assert float(draws.std()) == pytest.approx(0.7, abs=0.25)


def test_block_fitting_waits_for_every_transitions():
    proposal, params, kernel = _flow_imh()
    state = kernel.init(jax.random.key(0), jnp.array([1.5]))
    policy = fit_adaptor(
        maximum_likelihood(proposal),
        optax.adam(1e-2),
        data_fn=imh_data,
        batch_size=4,
        every=10,
        num_fit_steps=5,
    )

    before = warmup(jax.random.key(1), kernel, state, params, 9, policy).params
    after = warmup(jax.random.key(1), kernel, state, params, 10, policy).params

    unchanged = jax.tree.map(jnp.array_equal, before, params)
    changed = jax.tree.map(jnp.array_equal, after, params)
    assert all(jax.tree.leaves(unchanged))
    assert not all(jax.tree.leaves(changed))


def test_fit_adaptor_ring_buffer_keeps_the_most_recent_points():
    class State:
        def __init__(self, position):
            self.position = position

    def loss(params, points, weights):
        return jnp.sum(params * 0.0)

    adaptor = fit_adaptor(loss, optax.sgd(0.0), buffer_size=4, batch_size=None)
    fit_state = adaptor.init(State(jnp.zeros(1)), jnp.zeros(()))
    for value in range(6):
        fit_state, _, _ = adaptor.update(
            State(jnp.array([float(value)])), None, fit_state, jnp.zeros(())
        )

    assert sorted(fit_state.points[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0]
    assert int(fit_state.num_seen) == 6


def test_imh_data_rao_blackwellizes_the_accept_reject_outcome():
    info = IMHInfo(
        acceptance_rate=jnp.array(0.25),
        is_accepted=jnp.array(False),
        proposal=jnp.array([2.0]),
        previous_position=jnp.array([1.0]),
    )

    points, weights = imh_data(None, info)

    assert jnp.array_equal(points[:, 0], jnp.array([1.0, 2.0]))
    assert jnp.allclose(weights, jnp.array([0.75, 0.25]))


def test_axis_name_trains_one_proposal_shared_by_all_chains():
    proposal, params, kernel = _flow_imh()
    policy = fit_adaptor(
        maximum_likelihood(proposal),
        optax.adam(1e-2),
        data_fn=imh_data,
        batch_size=16,
        axis_name="chain",
    )
    keys = jax.random.split(jax.random.key(0), 4)
    states = jax.vmap(lambda key: kernel.init(key, jax.random.normal(key, (1,))))(keys)

    result = jax.vmap(
        lambda key, state: warmup(key, kernel, state, params, 50, policy),
        axis_name="chain",
    )(keys, states)

    pairs = list(
        zip(jax.tree.leaves(result.params), jax.tree.leaves(params), strict=True)
    )
    assert all(jnp.allclose(leaf, leaf[0]) for leaf, _ in pairs)
    assert not all(jnp.array_equal(leaf[0], initial) for leaf, initial in pairs)


def test_density_rejects_a_sample_only_model():
    class ZeroNet(nnx.Module):
        def __call__(self, _time, value, **_kwargs):
            return jnp.zeros_like(value)

    with pytest.raises(TypeError, match="tractable log-density"):
        density(LinearFlow(ZeroNet(), event_spec=1))


def test_pseudo_marginal_drops_a_default_that_needs_the_target():
    def noisy_logdensity(position, key):
        return -0.5 * jnp.sum(position**2) + 0.01 * jax.random.normal(key)

    # MCLMC's tuning needs one fixed log-density; a stochastic one has none.
    assert pseudo_marginal(mclmc, noisy_logdensity).default_adaptation is None


def test_pseudo_marginal_binds_the_imh_proposal_into_its_default():
    proposal, params = density(maf(1, 1, rngs=nnx.Rngs(0)))

    def noisy_logdensity(position, key):
        return _normal_logdensity(position) + 0.01 * jax.random.normal(key)

    kernel = pseudo_marginal(imh, noisy_logdensity, proposal=proposal)
    state = kernel.init(jax.random.key(0), jnp.array([1.5]))

    result = warmup(jax.random.key(1), kernel, state, params, 300)

    assert jax.tree.structure(result.params) == jax.tree.structure(params)
