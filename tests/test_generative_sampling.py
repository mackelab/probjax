import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM
from probjax.nn.generative.flow_matching import LinearFlow, LinearFlowSolverConfig


class TinyFlowNet(nnx.Module):
    def __init__(self, features=3, seed=0):
        self.proj = nnx.Linear(features, features, rngs=nnx.Rngs(seed))

    def __call__(self, _t, x, **_kwargs):
        return self.proj(x)


class TinyDiffusionNet(nnx.Module):
    def __init__(self, features=3, seed=0):
        self.proj = nnx.Linear(features, features, rngs=nnx.Rngs(seed))

    def __call__(self, _t, x):
        return self.proj(x)


def test_flow_build_sampler_is_cached_and_batch_polymorphic():
    model = LinearFlow(TinyFlowNet())
    sampler = model.build_sampler((3,), num_steps=4)

    assert sampler is model.build_sampler((3,), num_steps=4)
    assert "float32[b,3]" in str(sampler.exported.in_avals[-1])
    assert sampler.from_noise(jnp.ones((2, 3))).shape == (2, 3)
    assert sampler.from_noise(jnp.ones((5, 3))).shape == (5, 3)
    assert sampler.from_noise(jnp.ones((3,))).shape == (3,)
    assert sampler(jax.random.key(0), (2, 4)).shape == (2, 4, 3)


def test_built_sampler_reads_current_model_state():
    model = LinearFlow(TinyFlowNet())
    sampler = model.build_sampler((3,), num_steps=4)
    eps = jnp.ones((2, 3))

    before = sampler.from_noise(eps)
    model.net.proj.bias.set_value(model.net.proj.bias.get_value() + 1.0)
    after = sampler.from_noise(eps)

    assert not jnp.allclose(before, after)
    assert sampler is model.build_sampler((3,), num_steps=4)


@pytest.mark.parametrize("mode", ["ode", "sde"])
def test_diffusion_build_sampler_is_batch_polymorphic(mode):
    model = EDM(TinyDiffusionNet(), num_steps=4)
    sampler = model.build_sampler((3,), mode=mode, num_steps=4)

    samples_2 = sampler(jax.random.key(0), (2,))
    samples_5 = sampler(jax.random.key(1), (5,))

    assert samples_2.shape == (2, 3)
    assert samples_5.shape == (5, 3)
    assert jnp.all(jnp.isfinite(samples_2))
    assert jnp.all(jnp.isfinite(samples_5))


@pytest.mark.parametrize("model_type", [EDM, VE, VP, CosineDM])
def test_diffusion_ode_sampler_preserves_specialized_drift(model_type):
    model = model_type(TinyDiffusionNet(features=2), num_steps=3)

    samples = model.build_sampler((2,), num_steps=3)(jax.random.key(0), (2,))

    assert samples.shape == (2, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_flow_sampler_retains_custom_inverse_lowering():
    model = LinearFlow(TinyFlowNet(features=2))
    sampler = model.build_sampler((2,), num_steps=3)

    assert "custom_inverse_forward" in sampler.exported.mlir_module()


def test_as_distribution_uses_built_sampler():
    flow = LinearFlow(TinyFlowNet())
    flow_dist = flow.as_distribution((3,), num_steps=4)
    diffusion = EDM(TinyDiffusionNet(), num_steps=4)
    diffusion_dist = diffusion.as_distribution((3,), mode="ode", num_steps=4)
    diffusion_sde_dist = diffusion.as_distribution((3,), mode="sde", num_steps=4)

    assert flow_dist.rvs(jax.random.key(0), shape=(2,)).shape == (2, 3)
    assert flow_dist.rvs(jax.random.key(1), shape=(5,)).shape == (5, 3)
    assert diffusion_dist.rvs(jax.random.key(2), shape=(2,)).shape == (2, 3)
    assert diffusion_sde_dist.rvs(jax.random.key(3), shape=(2,)).shape == (2, 3)


def test_set_solver_cfg_invalidates_cached_sampler():
    model = LinearFlow(TinyFlowNet())
    before = model.build_sampler((3,), num_steps=4)

    model.set_solver_cfg(LinearFlowSolverConfig(num_steps=8))
    after = model.build_sampler((3,), num_steps=4)

    assert after is not before
