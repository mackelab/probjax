import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.inference import MCMC, neural_imh, neural_imh_warmup
from probjax.inference.mcmc.imh import _limit_data, _rao_blackwellized_data
from probjax.nn import LinearFlow, maf


def _normal_logdensity(position):
    return jax.scipy.stats.norm.logpdf(position, 1.5, 0.7).sum()


def test_neural_imh_runs_in_compiled_mcmc():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(jnp.zeros(1))
    params = kernel.init_params(state)

    result = MCMC(kernel, collect_info=("acceptance_rate",)).sample(
        jax.random.key(1), state, 8, params
    )

    assert result.samples.shape == (8, 1)
    assert result.info["acceptance_rate"].shape == (8,)
    assert jnp.all(jnp.isfinite(result.samples))


def test_neural_imh_warmup_fits_chain_positions_without_data():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(jnp.array([1.5]))
    params = kernel.init_params(state)
    warmup = neural_imh_warmup(
        proposal,
        num_adaptations=2,
        fit_steps=3,
        batch_size=8,
    )

    result = MCMC(kernel).warmup(jax.random.key(1), warmup, state, params, num_steps=12)

    assert result.info.losses.shape == (2, 3)
    assert result.info.acceptance_rate.shape == (2,)
    assert jnp.all(jnp.isfinite(result.info.losses))

    samples = (
        MCMC(kernel).sample(jax.random.key(2), result.state, 8, result.params).samples
    )
    assert samples.shape == (8, 1)


def test_neural_imh_warmup_augments_chain_with_data():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    distribution = proposal.as_dist()
    seed_data = jax.random.normal(jax.random.key(0), (128, 1)) * 0.4 + 2.0
    density_before = distribution.logpdf(seed_data).mean()
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(jnp.array([1.5]))
    params = kernel.init_params(state)

    result = MCMC(kernel).warmup(
        jax.random.key(1),
        neural_imh_warmup(
            proposal,
            data=seed_data,
            num_adaptations=1,
            fit_steps=30,
            batch_size=64,
            learning_rate=1e-2,
        ),
        state,
        params,
        num_steps=8,
    )

    assert distribution.logpdf(seed_data).mean() > density_before
    assert result.info.losses[-1, -1] < result.info.losses[0, 0]


def test_neural_imh_warmup_resumes_from_kernel_params():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    proposal.fit_standardization(jnp.zeros((32, 1)))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(jnp.array([1.5]))
    params = kernel.init_params(state)
    proposal.fit(
        jax.random.key(0),
        jnp.full((32, 1), 8.0),
        num_steps=10,
        learning_rate=1e-2,
    )

    result = MCMC(kernel).warmup(
        jax.random.key(1),
        neural_imh_warmup(
            proposal,
            num_adaptations=1,
            fit_steps=1,
            learning_rate=0.0,
        ),
        state,
        params,
        num_steps=2,
    )

    for expected, actual in zip(
        jax.tree.leaves(params.proposal_state),
        jax.tree.leaves(result.params.proposal_state),
        strict=True,
    ):
        assert jnp.array_equal(expected, actual)


def test_neural_imh_replay_buffer_keeps_recent_positions():
    data = {"x": jnp.arange(10), "y": jnp.arange(10) + 10}

    limited = _limit_data(data, 4)

    assert jnp.array_equal(limited["x"], jnp.arange(6, 10))
    assert jnp.array_equal(limited["y"], jnp.arange(16, 20))


def test_neural_imh_rao_blackwellizes_accept_reject_outcomes():
    positions, weights = _rao_blackwellized_data(
        jnp.array([0.0]),
        jnp.array([[1.0], [1.0]]),
        jnp.array([[1.0], [2.0]]),
        jnp.array([0.25, 0.75]),
    )

    assert jnp.array_equal(positions[:, 0], jnp.array([0.0, 1.0, 1.0, 2.0]))
    assert jnp.allclose(weights, jnp.array([0.75, 0.25, 0.25, 0.75]))


def test_neural_imh_rejects_sample_only_model():
    class ZeroNet(nnx.Module):
        def __call__(self, _time, value, **_kwargs):
            return jnp.zeros_like(value)

    with pytest.raises(ValueError, match="tractable logpdf"):
        neural_imh(
            _normal_logdensity,
            LinearFlow(ZeroNet()),
            event_spec=(1,),
        )
