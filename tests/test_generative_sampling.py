import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM
from probjax.nn.generative.flow_matching import LinearFlow, LinearFlowSolverConfig
from probjax.nn.generative.nflows.models import NormalizingFlow
from probjax.nn.generative.mean_flow import LinearMeanFlow
from probjax.stats.continuous import norm
from probjax.stats.indep import indep


class ShiftTransform:
    def __call__(self, x, context=None, *, rng=None):
        del rng
        if context is None:
            return x + 1.0
        return x + context

    def inverse_and_logdet(self, y):
        return y - 1.0, jnp.zeros((), dtype=y.dtype)


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


class TinyPyTreeNet(nnx.Module):
    def __init__(self, seed=0):
        self.proj_a = nnx.Linear(2, 2, rngs=nnx.Rngs(seed))
        self.proj_b = nnx.Linear(3, 3, rngs=nnx.Rngs(seed + 1))

    def __call__(self, _t, x, **_kwargs):
        return {"a": self.proj_a(x["a"]), "b": self.proj_b(x["b"])}


class TinyConditionalPyTreeNet(nnx.Module):
    def __init__(self, seed=0):
        self.proj_a = nnx.Linear(2, 2, rngs=nnx.Rngs(seed))
        self.proj_b = nnx.Linear(3, 3, rngs=nnx.Rngs(seed + 1))

    def __call__(self, _t, x, *, context=None, **_kwargs):
        if context is None:
            raise ValueError("context is required")
        return {
            "a": self.proj_a(x["a"]) + context["a"],
            "b": self.proj_b(x["b"]) + context["b"],
        }


class PyTreeBase:
    def rvs(self, rng, shape=()):
        key_a, key_b = jax.random.split(rng)
        return {
            "a": jax.random.normal(key_a, shape + (2,)),
            "b": jax.random.normal(key_b, shape + (3,)),
        }


class PyTreeShiftTransform:
    def __call__(self, value, context=None, *, rng=None):
        del rng
        return jax.tree.map(lambda x, shift: x + shift, value, context)


def test_flow_distribution_supports_pytree_events():
    distribution = LinearFlow(TinyPyTreeNet()).as_dist(
        {"a": (2,), "b": (3,)}, num_steps=4
    )

    initial = {"a": jnp.ones((5, 2)), "b": jnp.ones((5, 3))}
    from_noise = distribution.sample_from(initial)
    samples = distribution.sample(jax.random.key(0), (2, 4))

    assert from_noise["a"].shape == (5, 2)
    assert from_noise["b"].shape == (5, 3)
    assert samples["a"].shape == (2, 4, 2)
    assert samples["b"].shape == (2, 4, 3)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(LinearFlow(TinyConditionalPyTreeNet()), id="flow"),
        pytest.param(LinearMeanFlow(TinyConditionalPyTreeNet()), id="mean-flow"),
        pytest.param(EDM(TinyConditionalPyTreeNet(), num_steps=3), id="diffusion"),
    ],
)
def test_distribution_supports_pytree_events_and_context(model):
    event_spec = {"a": (2,), "b": (3,)}
    distribution = model.as_dist(event_spec, context_spec=event_spec, num_steps=3)
    context = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}

    samples = distribution.sample(jax.random.key(0), (2, 4), context=context)

    assert samples["a"].shape == (2, 4, 2)
    assert samples["b"].shape == (2, 4, 3)
    assert jnp.all(jnp.isfinite(samples["a"]))
    assert jnp.all(jnp.isfinite(samples["b"]))


def test_diffusion_sde_distribution_supports_pytree_context():
    model = EDM(TinyConditionalPyTreeNet(), num_steps=3)
    event_spec = {"a": (2,), "b": (3,)}
    distribution = model.as_dist(
        event_spec,
        context_spec=event_spec,
        mode="sde",
        num_steps=3,
    )
    context = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}

    samples = distribution.sample(jax.random.key(0), (2,), context=context)

    assert samples["a"].shape == (2, 2)
    assert samples["b"].shape == (2, 3)
    assert jnp.all(jnp.isfinite(samples["a"]))
    assert jnp.all(jnp.isfinite(samples["b"]))


def test_diffusion_sde_trace_is_time_first_with_pytree_context():
    model = EDM(TinyConditionalPyTreeNet(), num_steps=3)
    event_spec = {"a": (2,), "b": (3,)}
    distribution = model.as_dist(
        event_spec,
        context_spec=event_spec,
        mode="sde",
        num_steps=3,
    )
    context = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}

    samples = distribution.sample_path(jax.random.key(0), (2, 4), context=context)

    assert samples["a"].shape == (3, 2, 4, 2)
    assert samples["b"].shape == (3, 2, 4, 3)


def test_flow_distribution_is_batch_polymorphic():
    distribution = LinearFlow(TinyFlowNet()).as_dist((3,), num_steps=4)

    assert distribution.sample_from(jnp.ones((2, 3))).shape == (2, 3)
    assert distribution.sample_from(jnp.ones((5, 3))).shape == (5, 3)
    assert distribution.sample_from(jnp.ones((3,))).shape == (3,)
    assert distribution.sample(jax.random.key(0), (2, 4)).shape == (2, 4, 3)


def test_mean_flow_distribution_is_batch_polymorphic():
    distribution = LinearMeanFlow(TinyFlowNet()).as_dist((3,), num_steps=4)

    assert distribution.sample_from(jnp.ones((2, 3))).shape == (2, 3)
    assert distribution.sample_from(jnp.ones((5, 3))).shape == (5, 3)
    assert distribution.sample(jax.random.key(0), (2, 4)).shape == (2, 4, 3)


def test_normalizing_flow_distribution_is_batch_polymorphic():
    base_dist = indep(norm(jnp.zeros(2), jnp.ones(2)))
    distribution = NormalizingFlow(base_dist, ShiftTransform()).as_dist()

    assert distribution.sample_from(jnp.zeros((2, 2))).shape == (2, 2)
    assert distribution.sample_from(jnp.zeros((5, 2))).shape == (5, 2)
    assert distribution.sample(jax.random.key(0), (3,)).shape == (3, 2)
    assert distribution.logpdf(jnp.ones((4, 2))).shape == (4,)


def test_normalizing_flow_distribution_supports_context():
    base_dist = indep(norm(jnp.zeros(2), jnp.ones(2)))
    distribution = NormalizingFlow(base_dist, ShiftTransform()).as_dist(
        context_spec=(2,)
    )

    sample = distribution.sample_from(jnp.zeros((3, 2)), context=jnp.array([1.0, 2.0]))

    assert sample.shape == (3, 2)
    assert jnp.allclose(sample, jnp.array([[1.0, 2.0]] * 3))


def test_normalizing_flow_distribution_supports_pytree_events_and_context():
    event_spec = {"a": (2,), "b": (3,)}
    distribution = NormalizingFlow(PyTreeBase(), PyTreeShiftTransform()).as_dist(
        event_spec, context_spec=event_spec
    )
    context = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}

    samples = distribution.sample(jax.random.key(0), (2, 4), context=context)

    assert samples["a"].shape == (2, 4, 2)
    assert samples["b"].shape == (2, 4, 3)


def test_distribution_reads_current_model_state():
    model = LinearFlow(TinyFlowNet())
    distribution = model.as_dist((3,), num_steps=4)
    eps = jnp.ones((2, 3))

    before = distribution.sample_from(eps)
    model.net.proj.bias.set_value(model.net.proj.bias.get_value() + 1.0)
    after = distribution.sample_from(eps)

    assert not jnp.allclose(before, after)


@pytest.mark.parametrize("mode", ["ode", "sde"])
def test_diffusion_distribution_is_batch_polymorphic(mode):
    model = EDM(TinyDiffusionNet(), num_steps=4)
    distribution = model.as_dist((3,), mode=mode, num_steps=4)

    samples_2 = distribution.sample(jax.random.key(0), (2,))
    samples_5 = distribution.sample(jax.random.key(1), (5,))

    assert samples_2.shape == (2, 3)
    assert samples_5.shape == (5, 3)
    assert jnp.all(jnp.isfinite(samples_2))
    assert jnp.all(jnp.isfinite(samples_5))


@pytest.mark.parametrize("model_type", [EDM, VE, VP, CosineDM])
def test_diffusion_ode_sampler_preserves_specialized_drift(model_type):
    model = model_type(TinyDiffusionNet(features=2), num_steps=3)

    samples = model.as_dist((2,), num_steps=3).sample(jax.random.key(0), (2,))

    assert samples.shape == (2, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_flow_distribution_exposes_paths_separately():
    distribution = LinearFlow(TinyFlowNet()).as_dist((3,), num_steps=4)

    out = distribution.path_from(jnp.ones((2, 3)))
    assert out.shape == (4, 2, 3)
    assert jnp.all(jnp.isfinite(out))

    out = distribution.sample_path(jax.random.key(0), (2, 4))
    assert out.shape == (4, 2, 4, 3)


def test_diffusion_distribution_exposes_paths_separately():
    distribution = EDM(TinyDiffusionNet(), num_steps=4).as_dist((3,), num_steps=4)

    out = distribution.path_from(jnp.ones((2, 3)))
    assert out.shape == (4, 2, 3)
    assert jnp.all(jnp.isfinite(out))


def test_as_dist_is_the_inference_entry_point():
    flow = LinearFlow(TinyFlowNet())
    flow_dist = flow.as_dist((3,), num_steps=4)
    diffusion = EDM(TinyDiffusionNet(), num_steps=4)
    diffusion_dist = diffusion.as_dist((3,), mode="ode", num_steps=4)
    diffusion_sde_dist = diffusion.as_dist((3,), mode="sde", num_steps=4)

    assert flow_dist.sample(jax.random.key(0), (2,)).shape == (2, 3)
    assert flow_dist.sample(jax.random.key(1), (5,)).shape == (5, 3)
    assert diffusion_dist.sample(jax.random.key(2), (2,)).shape == (2, 3)
    assert diffusion_sde_dist.sample(jax.random.key(3), (2,)).shape == (2, 3)

    assert not hasattr(flow, "build_sampler")
    assert not hasattr(flow, "sample")
    assert not hasattr(diffusion, "sample_ode")
    assert not hasattr(diffusion, "sample_sde")


def test_distribution_binds_pytree_context():
    model = LinearFlow(TinyConditionalPyTreeNet())
    event_spec = {"a": (2,), "b": (3,)}
    context = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}
    distribution = model.as_dist(
        event_spec,
        context_spec=event_spec,
        num_steps=3,
    ).condition(context)

    samples = distribution.sample(jax.random.key(0), (2, 4))

    assert distribution.event_shape == event_spec
    assert samples["a"].shape == (2, 4, 2)
    assert samples["b"].shape == (2, 4, 3)


def test_distribution_exposes_flow_logpdf():
    base_dist = indep(norm(jnp.zeros(2), jnp.ones(2)))
    flow = NormalizingFlow(base_dist, ShiftTransform())
    distribution = flow.as_dist()
    values = jnp.ones((4, 2))

    assert distribution.has_logpdf
    assert distribution.logpdf(values).shape == (4,)
    assert jnp.allclose(distribution.logpdf(values), base_dist.logpdf(values - 1.0))
    assert not hasattr(flow, "logpdf")


def test_distribution_binds_context_for_sample_and_logpdf():
    base_dist = indep(norm(jnp.zeros(2), jnp.ones(2)))
    flow = NormalizingFlow(base_dist, ShiftTransform())
    context = jnp.array([1.0, 2.0])
    distribution = flow.as_dist(context_spec=(2,)).condition(context)
    values = jnp.array([[1.0, 2.0], [2.0, 3.0]])

    samples = distribution.sample(jax.random.key(0), (3,))
    logpdf = distribution.logpdf(values)

    assert samples.shape == (3, 2)
    assert logpdf.shape == (2,)
    assert jnp.allclose(logpdf, base_dist.logpdf(values - context))


def test_distribution_reports_missing_logpdf():
    model = LinearFlow(TinyFlowNet())
    distribution = model.as_dist((3,), num_steps=3)

    assert not distribution.has_logpdf
    with pytest.raises(NotImplementedError, match="tractable logpdf"):
        distribution.logpdf(jnp.ones((2, 3)))


def test_distribution_rebuilds_after_cache_invalidation():
    model = LinearFlow(TinyFlowNet())
    distribution = model.as_dist((3,), num_steps=4)
    distribution.compile("sample")

    model.set_solver_cfg(LinearFlowSolverConfig(num_steps=8))
    samples = distribution.sample(jax.random.key(0), (2,))

    assert samples.shape == (2, 3)
