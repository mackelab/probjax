import sys

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.inference import neural_imh, neural_imh_warmup, run, warmup
from probjax.inference.mcmc.imh import _limit_data, _rao_blackwellized_data
from probjax.nn import FitConfig, LinearFlow, maf


def _normal_logdensity(position):
    return jax.scipy.stats.norm.logpdf(position, 1.5, 0.7).sum()


def test_neural_imh_runs_in_compiled_mcmc():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(None, jnp.zeros(1))
    params = kernel.init_params(state)

    result = run(
        jax.random.key(1),
        kernel,
        state,
        params,
        8,
        collect=("position", "acceptance_rate"),
    )

    assert result.trace["position"].shape == (8, 1)
    assert result.trace["acceptance_rate"].shape == (8,)
    assert jnp.all(jnp.isfinite(result.trace["position"]))


def test_neural_imh_warmup_fits_chain_positions_without_data():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(None, jnp.array([1.5]))
    params = kernel.init_params(state)
    policy = neural_imh_warmup(
        proposal,
        num_adaptations=2,
        fit_steps=3,
        batch_size=8,
    )

    result = warmup(jax.random.key(1), kernel, state, params, 12, policy)

    assert result.trace.losses.shape == (2, 3)
    assert result.trace.acceptance_rate.shape == (2,)
    assert jnp.all(jnp.isfinite(result.trace.losses))

    samples = run(jax.random.key(2), kernel, result.state, result.params, 8).trace[
        "position"
    ]
    assert samples.shape == (8, 1)


def test_neural_imh_warmup_augments_chain_with_data():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    distribution = proposal.as_dist()
    seed_data = jax.random.normal(jax.random.key(0), (128, 1)) * 0.4 + 2.0
    density_before = distribution.logpdf(seed_data).mean()
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(None, jnp.array([1.5]))
    params = kernel.init_params(state)

    result = warmup(
        jax.random.key(1),
        kernel,
        state,
        params,
        8,
        neural_imh_warmup(
            proposal,
            data=seed_data,
            num_adaptations=1,
            fit_steps=30,
            batch_size=64,
            learning_rate=1e-2,
        ),
    )

    assert distribution.logpdf(seed_data).mean() > density_before
    assert result.trace.losses[-1, -1] < result.trace.losses[0, 0]


def test_neural_imh_warmup_resumes_from_kernel_params():
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    proposal.fit_standardization(jnp.zeros((32, 1)))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(None, jnp.array([1.5]))
    params = kernel.init_params(state)
    proposal.fit(
        jax.random.key(0),
        jnp.full((32, 1), 8.0),
        config=FitConfig(max_steps=10, learning_rate=1e-2),
    )

    result = warmup(
        jax.random.key(1),
        kernel,
        state,
        params,
        2,
        neural_imh_warmup(
            proposal,
            num_adaptations=1,
            fit_steps=1,
            learning_rate=0.0,
        ),
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
            LinearFlow(ZeroNet(), event_spec=1),
            event_spec=(1,),
        )


def test_neural_imh_warmup_draws_fresh_keys_per_block(monkeypatch):
    # ``probjax.inference.mcmc.imh`` names the exported class, not the module.
    imh_module = sys.modules["probjax.inference.mcmc.imh"]
    block_keys = []
    original = imh_module.run_kernel

    def recording_run_kernel(rng, *args, **kwargs):
        block_keys.append(jax.random.key_data(rng))
        return original(rng, *args, **kwargs)

    monkeypatch.setattr(imh_module, "run_kernel", recording_run_kernel)
    proposal = maf(1, 1, rngs=nnx.Rngs(0))
    kernel = neural_imh(_normal_logdensity, proposal)
    state = kernel.init(None, jnp.array([1.5]))
    warmup = neural_imh_warmup(proposal, num_adaptations=3, fit_steps=2, batch_size=8)

    warmup.run(jax.random.key(1), kernel, state, kernel.init_params(state), 9)

    assert len(block_keys) == 3
    assert not jnp.array_equal(block_keys[0], block_keys[1])
    assert not jnp.array_equal(block_keys[1], block_keys[2])
