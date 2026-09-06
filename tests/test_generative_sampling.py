import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM
from probjax.nn.generative.flow_matching import LinearFlow, LinearFlowSolverConfig
from probjax.nn.generative.nflows.models import NormalizingFlow
from probjax.nn.generative.mean_flow import LinearMeanFlow
from probjax.nn.generative.mean_flow.config import SigmoidPairFlowTrainingConfig
from probjax.nn.generative.flow_matching.config import LinearInterpolationSchedule
from probjax.nn.losses.mean_flow import build_mean_flow_matching_loss
from probjax.nn import TimeMLP
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


def test_sigmoid_pair_config_produces_interior_pairs():
    cfg = SigmoidPairFlowTrainingConfig()
    t, r = cfg.sample_times_pair(jax.random.key(0), (20_000,))

    assert bool(jnp.all(t <= r))
    assert bool(jnp.all((t >= cfg.t_min) & (r <= cfg.t_max)))
    # Default percent_rt=0.25 distinct pairs, rest exactly r == t.
    assert float((r == t).mean()) == pytest.approx(0.75, abs=0.01)
    # Distinct pairs must be genuinely interior, not pinned at r == 1
    # (the old additive clip put ~70% of them exactly at 1).
    interior = (r > t) & (r < 1.0)
    assert float(interior.mean()) == pytest.approx(0.25, abs=0.01)
    assert float((r >= 1.0).mean()) < 0.01


class RPassThroughNet(nnx.Module):
    """Stub net returning its r conditioning, to probe the r-JVP path."""

    def __call__(self, _t, x, r=None, **_kwargs):
        assert r is not None
        return jnp.broadcast_to(r, x.shape)


def test_mean_flow_r_clip_has_no_tie_split():
    # At r == t, jnp.maximum's JVP tie rule would blend t's tangent into r
    # (0.5/0.5 split). The stop_gradient bound must make the r-tangent at a
    # tie match the interior limit instead.
    model = LinearMeanFlow(RPassThroughNet())
    x = jnp.ones((2, 2))
    t = jnp.full((2, 1), 0.3)

    def call(r_, t_):
        return model(t_, x, r=r_)

    dr = jnp.ones_like(t)
    zt = jnp.zeros_like(t)
    _, tie_tangent = jax.jvp(call, (t, t), (dr, zt))
    _, interior_tangent = jax.jvp(call, (t + 1e-3, t), (dr, zt))
    assert jnp.all(jnp.isfinite(tie_tangent))
    assert float(jnp.abs(tie_tangent - interior_tangent).max()) < 1e-2


def test_mean_flow_loss_is_finite_with_new_sampler():
    model = LinearMeanFlow(TinyFlowNet())
    data = jax.random.normal(jax.random.key(0), (8, 3))
    loss = model.loss(jax.random.key(1), data)
    assert jnp.all(jnp.isfinite(loss))


def _stub_velocity_fn(t, x, r=None):
    rr = 0.0 if r is None else r
    return x * t + rr


def test_mean_flow_imf_matches_original_on_diagonal():
    # At r == t both objectives reduce to ||v_t - u_t||^2 exactly.
    schedule = LinearInterpolationSchedule()
    imf_fn = build_mean_flow_matching_loss(_stub_velocity_fn, schedule, imf=True)
    orig_fn = build_mean_flow_matching_loss(_stub_velocity_fn, schedule, imf=False)
    x0 = jax.random.normal(jax.random.key(0), (32, 2))
    x1 = jax.random.normal(jax.random.key(1), (32, 2))
    t = jnp.full((32, 1), 0.4)
    assert float(imf_fn(t, t, x0, x1)) == pytest.approx(
        float(orig_fn(t, t, x0, x1)), rel=1e-6
    )


def test_mean_flow_imf_differs_off_diagonal():
    # Off the diagonal the boundary-condition tangent changes the loss.
    schedule = LinearInterpolationSchedule()
    imf_fn = build_mean_flow_matching_loss(_stub_velocity_fn, schedule, imf=True)
    orig_fn = build_mean_flow_matching_loss(_stub_velocity_fn, schedule, imf=False)
    x0 = jax.random.normal(jax.random.key(0), (32, 2))
    x1 = jax.random.normal(jax.random.key(1), (32, 2))
    t = jnp.full((32, 1), 0.4)
    r = jnp.full((32, 1), 0.8)
    imf_loss = float(imf_fn(r, t, x0, x1))
    orig_loss = float(orig_fn(r, t, x0, x1))
    assert jnp.isfinite(imf_loss) and jnp.isfinite(orig_loss)
    assert imf_loss != pytest.approx(orig_loss, rel=1e-3)


def test_mean_flow_original_objective_still_available():
    # imf=False must reproduce the pre-iMF fixed-seed losses exactly.
    net = TimeMLP(2, rngs=nnx.Rngs(0))
    model = LinearMeanFlow(net)
    data = jax.random.normal(jax.random.key(42), (256, 2)) * 2.0
    expected = [4.606965065002441, 5.016578197479248, 4.452657699584961]
    for seed, want in zip([0, 1, 2], expected):
        got = float(model.loss(jax.random.key(seed), data, imf=False))
        assert got == pytest.approx(want, rel=1e-6)
    # And the default (iMF) path is finite but different.
    got_imf = float(model.loss(jax.random.key(0), data))
    assert jnp.isfinite(got_imf)
    assert got_imf != pytest.approx(expected[0], rel=1e-3)
