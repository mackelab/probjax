"""Tests for the normalizing-flow configuration layer.

These cover the mapping from an unconstrained parameter vector onto the natural
parameters the ``probjax.stats.bijective`` functions expect, and the assembly of
those configs into flows.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.core import inverse_and_logabsdet
from probjax.nn.generative.nflows import (
    AffineBijectorConfig,
    AutoregressiveNFlowConfig,
    BernsteinBijectorConfig,
    BijectorConfigProtocol,
    ConditionerConfigProtocol,
    CouplingNFlowConfig,
    DeepSigmoidBijectorConfig,
    ElementwiseNFlowConfig,
    FlipMixingConfig,
    MixingConfigProtocol,
    MixtureCDFBijectorConfig,
    MLPConditionerConfig,
    MonotoneHermiteCubicSplineConfig,
    NFlow,
    NoMixingConfig,
    PermuteMixingConfig,
    PiecewiseAffineSplineConfig,
    RationalLinearSplineConfig,
    RationalQuadraticSplineConfig,
    RotationMixingConfig,
    ShiftBijectorConfig,
    SSMConditionerConfig,
    SumOfSquaresBijectorConfig,
    TransformerConditionerConfig,
    UMNNBijectorConfig,
)

SPLINE_CONFIGS = [
    RationalQuadraticSplineConfig,
    RationalLinearSplineConfig,
    MonotoneHermiteCubicSplineConfig,
    PiecewiseAffineSplineConfig,
]

ALL_BIJECTORS = [
    pytest.param(ShiftBijectorConfig(), id="shift"),
    pytest.param(AffineBijectorConfig(), id="affine"),
    pytest.param(AffineBijectorConfig(scale_transform="exp"), id="affine-exp"),
    pytest.param(RationalQuadraticSplineConfig(num_bins=5), id="rq-spline"),
    pytest.param(
        RationalQuadraticSplineConfig(num_bins=5, x_min=-3.0, x_max=3.0,
                                      y_min=-2.0, y_max=4.0),
        id="rq-spline-asymmetric",
    ),
    pytest.param(RationalLinearSplineConfig(num_bins=5), id="rl-spline"),
    pytest.param(MonotoneHermiteCubicSplineConfig(num_bins=5), id="hermite-spline"),
    pytest.param(PiecewiseAffineSplineConfig(num_bins=5), id="affine-spline"),
    pytest.param(DeepSigmoidBijectorConfig(num_components=3), id="deep-sigmoid"),
    pytest.param(UMNNBijectorConfig(num_hidden=3), id="umnn"),
    pytest.param(SumOfSquaresBijectorConfig(num_polys=2, degree=3), id="sos"),
    pytest.param(BernsteinBijectorConfig(degree=6), id="bernstein"),
    pytest.param(MixtureCDFBijectorConfig(num_components=3), id="mixture-cdf"),
]


def _params(cfg, key, batch=(), scale=0.5):
    return jax.random.normal(key, batch + (cfg.params_dim(),)) * scale


def _can_be_identity(cfg):
    """Whether zero parameters give back exactly the identity map.

    Two families cannot. A spline whose x and y domains differ is a rescaling
    by construction. And a mixture-like head deliberately spreads its
    components apart at initialisation -- a mixture of *identical* components
    is just one component, so breaking that symmetry and being exactly the
    identity are mutually exclusive. Those heads are held to
    ``test_near_identity_at_zero_params`` instead.
    """
    if getattr(cfg, "spread", 0.0):
        return False
    return not (
        hasattr(cfg, "x_min")
        and (cfg.x_min, cfg.x_max) != (cfg.y_min, cfg.y_max)
    )


# ---------------------------------------------------------------------------
# Bijector configs: packing contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_satisfies_protocol(cfg):
    assert isinstance(cfg, BijectorConfigProtocol)


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_unpack_consumes_exactly_params_dim(cfg):
    """``params_dim`` must match what ``unpack`` actually reads."""
    params = _params(cfg, jax.random.PRNGKey(0))
    natural = cfg.unpack(params)
    assert sum(int(jnp.asarray(p).size) for p in natural) > 0

    with pytest.raises(ValueError, match="trailing dimension"):
        cfg(jnp.zeros((cfg.params_dim() + 1,)), jnp.array(0.0))


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_identity_at_zero_params(cfg):
    """Zero-initialised conditioners must emit the identity bijection."""
    if not _can_be_identity(cfg):
        pytest.skip("asymmetric domain cannot map to the identity")
    zeros = jnp.zeros((cfg.params_dim(),))
    for x in (-1.7, -0.2, 0.0, 0.42, 2.3):
        assert jnp.allclose(cfg(zeros, jnp.array(x)), x, atol=1e-5)


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_near_identity_at_zero_params(cfg):
    """Heads that spread their components must still start well-conditioned.

    They give up the exact identity, but the map has to stay close to it: a
    badly-scaled start compounds across stacked transforms.
    """
    if not getattr(cfg, "spread", 0.0):
        pytest.skip("does not spread components; identity is covered above")
    xs = jnp.linspace(-3.0, 3.0, 64)
    params = jnp.zeros((xs.size, cfg.params_dim()))
    ys = cfg(params, xs)

    assert jnp.all(jnp.isfinite(ys))
    assert jnp.all(jnp.diff(ys) > 0), "not monotone at initialisation"
    assert jnp.max(jnp.abs(ys - xs)) < 0.5, "starts too far from the identity"
    slopes = jnp.diff(ys) / jnp.diff(xs)
    assert jnp.all((slopes > 0.5) & (slopes < 2.0)), "badly scaled at initialisation"


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_roundtrip_and_logdet(cfg):
    params = _params(cfg, jax.random.PRNGKey(0), batch=(4,))
    x = jax.random.normal(jax.random.PRNGKey(1), (4,))

    y = cfg(params, x)
    x_rec, logdet = inverse_and_logabsdet(cfg, invertible_arg=1)(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-4)

    # finite differences on the forward, negated (this is the inverse's log-det)
    h = 1e-3
    slope = (cfg(params, x + h) - cfg(params, x - h)) / (2 * h)
    assert jnp.allclose(logdet, -jnp.sum(jnp.log(jnp.abs(slope))), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_is_strictly_increasing(cfg):
    params = _params(cfg, jax.random.PRNGKey(3))
    xs = jnp.linspace(-3.0, 3.0, 64)
    ys = cfg(jnp.broadcast_to(params, (64, cfg.params_dim())), xs)
    assert jnp.all(jnp.diff(ys) > 0.0)


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_batching_matches_elementwise(cfg):
    params = _params(cfg, jax.random.PRNGKey(4), batch=(3, 5))
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 5))

    batched = cfg(params, x)
    assert batched.shape == x.shape
    for i in range(3):
        for j in range(5):
            single = cfg(params[i, j], x[i, j])
            assert jnp.allclose(batched[i, j], single, atol=1e-5)


# ---------------------------------------------------------------------------
# Spline knot packing (regression tests for the old off-by-one packing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SPLINE_CONFIGS)
@pytest.mark.parametrize("num_bins", [1, 4, 10])
def test_spline_knot_positions(cls, num_bins):
    cfg = cls(num_bins=num_bins, x_min=-4.0, x_max=6.0, y_min=-3.0, y_max=8.0)
    params = _params(cfg, jax.random.PRNGKey(0), scale=2.0)
    x_pos, y_pos = cfg.unpack(params)[:2]

    # num_bins bins means num_bins + 1 boundaries
    assert x_pos.shape == (num_bins + 1,)
    assert y_pos.shape == (num_bins + 1,)

    # endpoints land exactly on the configured bounds
    assert x_pos[0] == -4.0 and x_pos[-1] == 6.0
    assert y_pos[0] == -3.0 and y_pos[-1] == 8.0

    # strictly increasing, with every bin at least min_bin_size wide
    assert jnp.all(jnp.diff(x_pos) > 0.0)
    assert jnp.all(jnp.diff(y_pos) > 0.0)
    assert jnp.min(jnp.diff(x_pos)) >= cfg.min_bin_size * (6.0 - -4.0) * 0.99


@pytest.mark.parametrize("cls", [c for c in SPLINE_CONFIGS if c is not PiecewiseAffineSplineConfig])
@pytest.mark.parametrize("num_bins", [1, 4, 10])
def test_spline_knot_slopes(cls, num_bins):
    cfg = cls(num_bins=num_bins)
    params = _params(cfg, jax.random.PRNGKey(0), scale=2.0)
    slopes = cfg.unpack(params)[2]

    assert slopes.shape == (num_bins + 1,)
    assert jnp.all(slopes > cfg.min_knot_slope)

    # zero params give unit slopes, i.e. the identity
    unit = cfg.unpack(jnp.zeros((cfg.params_dim(),)))[2]
    assert jnp.allclose(unit, 1.0, atol=1e-6)


@pytest.mark.parametrize("cls", SPLINE_CONFIGS)
def test_spline_params_dim(cls):
    has_slopes = cls is not PiecewiseAffineSplineConfig
    for num_bins in (1, 3, 8):
        cfg = cls(num_bins=num_bins)
        expected = 3 * num_bins + 1 if has_slopes else 2 * num_bins
        assert cfg.params_dim() == expected


def test_spline_rejects_degenerate_bins():
    with pytest.raises(ValueError, match="min_bin_size"):
        RationalQuadraticSplineConfig(num_bins=10, min_bin_size=0.2)
    with pytest.raises(ValueError, match="num_bins"):
        RationalQuadraticSplineConfig(num_bins=0)
    with pytest.raises(ValueError, match="bounds"):
        RationalQuadraticSplineConfig(x_min=1.0, x_max=-1.0)


# ---------------------------------------------------------------------------
# Other packing invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg", [DeepSigmoidBijectorConfig(num_components=4), MixtureCDFBijectorConfig(num_components=4)]
)
def test_log_weights_are_on_the_log_simplex(cfg):
    params = _params(cfg, jax.random.PRNGKey(0), scale=3.0)
    log_weights = cfg.unpack(params)[0]
    assert log_weights.shape == (4,)
    assert jnp.allclose(jax.nn.logsumexp(log_weights), 0.0, atol=1e-6)


def test_deep_sigmoid_slopes_positive():
    cfg = DeepSigmoidBijectorConfig(num_components=4)
    _, slopes, _ = cfg.unpack(_params(cfg, jax.random.PRNGKey(0), scale=3.0))
    assert jnp.all(slopes > cfg.min_slope)


def test_mixture_cdf_scales_positive():
    cfg = MixtureCDFBijectorConfig(num_components=4)
    _, _, scales = cfg.unpack(_params(cfg, jax.random.PRNGKey(0), scale=3.0))
    assert jnp.all(scales > cfg.min_scale)


def test_bernstein_theta_is_a_normalised_increasing_sequence():
    cfg = BernsteinBijectorConfig(degree=6)
    (theta,) = cfg.unpack(_params(cfg, jax.random.PRNGKey(0), scale=3.0))
    assert theta.shape == (cfg.degree + 1,)
    assert jnp.allclose(theta[0], 0.0)
    assert jnp.allclose(theta[-1], 1.0, atol=1e-6)
    assert jnp.all(jnp.diff(theta) > 0.0)


def test_sos_coefficients_shape_and_unit_slope_at_zero():
    cfg = SumOfSquaresBijectorConfig(num_polys=3, degree=2)
    coeffs, constant = cfg.unpack(jnp.zeros((cfg.params_dim(),)))
    assert coeffs.shape == (3, 3)
    assert constant.shape == ()
    # derivative at zero params is eps + sum_k a_k0^2 == 1
    assert jnp.allclose(jnp.sum(coeffs[..., 0] ** 2), 1.0, atol=1e-5)


def test_mixture_cdf_free_init_spreads_locations():
    """The free parameter table needs distinct locations or components collapse."""
    cfg = MixtureCDFBijectorConfig(num_components=4)
    init = cfg.params_init()
    params = init(jax.random.PRNGKey(0), (5, cfg.params_dim()), jnp.float32)
    log_w, locs, scales = cfg.unpack(params)

    assert jnp.all(jnp.std(locs, axis=-1) > 0.0)
    assert jnp.allclose(log_w, jnp.log(0.25), atol=1e-6)  # weights still uniform
    assert jnp.allclose(scales, 1.0, atol=1e-6)  # scales still unit


@pytest.mark.parametrize("cfg", ALL_BIJECTORS)
def test_default_params_init_is_identity(cfg):
    if isinstance(cfg, MixtureCDFBijectorConfig):
        pytest.skip("deliberately spreads locations; covered separately")
    if not _can_be_identity(cfg):
        pytest.skip("asymmetric domain cannot map to the identity")
    init = cfg.params_init()
    params = init(jax.random.PRNGKey(0), (3, cfg.params_dim()), jnp.float32)
    x = jnp.array([0.5, -1.0, 2.0])
    assert jnp.allclose(cfg(params, x), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Conditioner / mixing configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cond",
    [
        MLPConditionerConfig(hidden_dims=(16,)),
        SSMConditionerConfig(model_dim=8, num_layers=1),
        TransformerConditionerConfig(model_dim=8, num_heads=2, num_layers=1),
    ],
)
def test_conditioner_satisfies_protocol(cond):
    assert isinstance(cond, ConditionerConfigProtocol)


@pytest.mark.parametrize("num_bins", [4, 10])
def test_coupling_conditioner_width_matches_params_dim(num_bins):
    """Regression: the conditioner must emit exactly params_dim per element."""
    bij = RationalQuadraticSplineConfig(num_bins=num_bins)
    input_dim, split_index = 5, 2
    transformed_dims = input_dim - split_index

    net = MLPConditionerConfig(hidden_dims=(16,)).build_coupling(
        split_index,
        transformed_dims * bij.params_dim(),
        bij,
        context_features=None,
        rngs=nnx.Rngs(0),
    )
    out = net.conditioner(jnp.zeros((split_index,)))
    assert out.shape == (transformed_dims * bij.params_dim(),)


@pytest.mark.parametrize("num_bins", [4, 10])
def test_autoregressive_conditioner_width_matches_params_dim(num_bins):
    """Regression: AutoregressiveMLP scales by in_out_features itself.

    Passing an already-scaled width made the output layer input_dim times too
    wide and misaligned every dimension's parameter block.
    """
    bij = RationalQuadraticSplineConfig(num_bins=num_bins)
    input_dim = 5

    net = MLPConditionerConfig(hidden_dims=(16,)).build_autoregressive(
        input_dim,
        bij.params_dim(),
        bij,
        context_features=None,
        output_order="grouped",
        rngs=nnx.Rngs(0),
    )
    params = net.predict_bij_params(jnp.zeros((input_dim,)))
    assert params.shape == (input_dim * bij.params_dim(),)


def test_ssm_conditioner_builds_autoregressive_transform():
    bij = ShiftBijectorConfig()
    net = SSMConditionerConfig(model_dim=8, num_layers=1).build_autoregressive(
        5,
        bij.params_dim(),
        bij,
        context_features=None,
        output_order="grouped",
        rngs=nnx.Rngs(0),
    )
    x = jnp.arange(5, dtype=jnp.float32)

    y = net(x)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)


@pytest.mark.parametrize(
    "mixing",
    [
        FlipMixingConfig(),
        PermuteMixingConfig(mode="random"),
        PermuteMixingConfig(mode="reverse"),
        RotationMixingConfig(),
        RotationMixingConfig(learnable=True),
        NoMixingConfig(),
    ],
)
def test_mixing_configs(mixing):
    assert isinstance(mixing, MixingConfigProtocol)
    layer = mixing.build(4, rngs=nnx.Rngs(0))
    if isinstance(mixing, NoMixingConfig):
        assert layer is None
        return
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = layer(x)
    assert y.shape == x.shape
    # mixing is volume preserving, so it must not change the value multiset
    if not isinstance(mixing, RotationMixingConfig):
        assert jnp.allclose(jnp.sort(y), jnp.sort(x))


def test_permute_reverse_is_a_flip():
    layer = PermuteMixingConfig(mode="reverse").build(4, rngs=nnx.Rngs(0))
    x = jnp.arange(4.0)
    assert jnp.allclose(layer(x), x[::-1])


def test_permute_random_is_seeded_by_rngs():
    a = PermuteMixingConfig().build(6, rngs=nnx.Rngs(0)).permutation[...]
    b = PermuteMixingConfig().build(6, rngs=nnx.Rngs(0)).permutation[...]
    c = PermuteMixingConfig().build(6, rngs=nnx.Rngs(1)).permutation[...]
    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)


# ---------------------------------------------------------------------------
# Flow configs and assembly
# ---------------------------------------------------------------------------


def test_coupling_split_index_defaults_and_validates():
    assert CouplingNFlowConfig(input_dim=7).split_index == 3
    assert CouplingNFlowConfig(input_dim=7, split_index=5).split_index == 5
    with pytest.raises(ValueError, match="split_index"):
        CouplingNFlowConfig(input_dim=4, split_index=4)
    with pytest.raises(ValueError, match="split_index"):
        CouplingNFlowConfig(input_dim=4, split_index=0)


def test_flow_config_validates_shape():
    with pytest.raises(ValueError, match="input_dim"):
        AutoregressiveNFlowConfig(input_dim=0)
    with pytest.raises(ValueError, match="num_transforms"):
        AutoregressiveNFlowConfig(input_dim=3, num_transforms=0)
    with pytest.raises(ValueError, match="context"):
        ElementwiseNFlowConfig(input_dim=3, context_features=2)


def test_nflow_rejects_wrong_config_types():
    with pytest.raises(TypeError, match="NFlowConfig"):
        NFlow(object(), nnx.Rngs(0))
    cfg = AutoregressiveNFlowConfig(input_dim=3)
    cfg.bijector = object()
    with pytest.raises(TypeError, match="BijectorConfigProtocol"):
        NFlow(cfg, nnx.Rngs(0))


@pytest.mark.parametrize(
    "cfg_cls", [CouplingNFlowConfig, AutoregressiveNFlowConfig, ElementwiseNFlowConfig]
)
def test_nflow_builds_and_round_trips(cfg_cls):
    input_dim = 4
    cfg = cfg_cls(input_dim=input_dim, num_transforms=2)
    flow = NFlow(cfg, nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(0), (8, input_dim))
    loss = flow.loss(None, x)
    assert jnp.isfinite(loss)

    samples = flow.as_dist().rvs(jax.random.PRNGKey(1), (16,))
    assert samples.shape == (16, input_dim)
    assert jnp.all(jnp.isfinite(flow.as_dist().logpdf(samples)))


def test_ssm_autoregressive_flow_samples_and_evaluates_logpdf():
    cfg = AutoregressiveNFlowConfig(
        input_dim=4,
        num_transforms=2,
        conditioner=SSMConditionerConfig(model_dim=8, num_layers=1),
    )
    flow = NFlow(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (8, cfg.input_dim))

    assert jnp.isfinite(flow.loss(None, x))

    distribution = flow.as_dist()
    samples = distribution.rvs(jax.random.PRNGKey(1), (16,))
    logpdf = distribution.logpdf(samples)

    assert samples.shape == (16, cfg.input_dim)
    assert logpdf.shape == (16,)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(logpdf))


def test_mixing_is_interleaved_not_appended():
    cfg = AutoregressiveNFlowConfig(input_dim=3, num_transforms=3)
    flow = NFlow(cfg, nnx.Rngs(0))
    kinds = [type(layer).__name__ for layer in flow.transformation.layers]
    assert kinds == [
        "AutoregressiveMLP",
        "Flip",
        "AutoregressiveMLP",
        "Flip",
        "AutoregressiveMLP",
    ]


def test_no_mixing_config_omits_layers():
    cfg = AutoregressiveNFlowConfig(input_dim=3, num_transforms=3, mixing=NoMixingConfig())
    flow = NFlow(cfg, nnx.Rngs(0))
    assert len(flow.transformation.layers) == 3


def test_flow_starts_at_the_identity():
    """Zero-initialised conditioners plus flips leave the density unchanged."""
    input_dim = 4
    x = jax.random.normal(jax.random.PRNGKey(0), (32, input_dim))
    base_nll = 0.5 * jnp.sum(x**2, axis=-1) + 0.5 * input_dim * jnp.log(2 * jnp.pi)

    for cfg_cls in (CouplingNFlowConfig, AutoregressiveNFlowConfig):
        # DeepSigmoidBijectorConfig is excluded on purpose: it spreads its
        # components at initialisation, so it starts near -- not at -- the
        # identity. See test_near_identity_at_zero_params.
        for bijector in (
            AffineBijectorConfig(),
            RationalQuadraticSplineConfig(num_bins=6),
        ):
            flow = NFlow(
                cfg_cls(input_dim=input_dim, num_transforms=3, bijector=bijector),
                nnx.Rngs(0),
            )
            assert jnp.allclose(flow.loss(None, x), jnp.mean(base_nll), atol=1e-4)


def test_flow_config_overrides_reach_the_bijection():
    """A non-default bijector config must actually change the transform."""
    cfg = AutoregressiveNFlowConfig(
        input_dim=3,
        num_transforms=1,
        bijector=RationalQuadraticSplineConfig(num_bins=16),
    )
    flow = NFlow(cfg, nnx.Rngs(0))
    conditioner = flow.transformation.layers[0]
    assert conditioner.bijector_dim == 3 * 16 + 1
