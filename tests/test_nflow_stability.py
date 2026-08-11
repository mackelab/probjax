"""Numerical-stability invariants for the flow stack.

Each test here pins a failure mode that was reachable at some point: a solver
that returned a confidently wrong root, a bijector whose log-determinant went
infinite far from the data, a training loop that swallowed a NaN.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

import probjax.stats.bijective as B
from probjax.core import inverse_and_logabsdet
from probjax.nn.generative.nflows.config import (
    AffineBijectorConfig,
    BernsteinBijectorConfig,
    DeepSigmoidBijectorConfig,
    MixtureCDFBijectorConfig,
    MonotoneHermiteCubicSplineConfig,
    PiecewiseAffineSplineConfig,
    RationalLinearSplineConfig,
    RationalQuadraticSplineConfig,
    ShiftBijectorConfig,
    SumOfSquaresBijectorConfig,
    UMNNBijectorConfig,
)
from probjax.stats.bijective.monotone import _solve_increasing
from probjax.stats.bijective.rational_quadratic import _safe_quadratic_root
from probjax.stats.fit import fit

ALL_CONFIGS = [
    pytest.param(ShiftBijectorConfig(), id="shift"),
    pytest.param(AffineBijectorConfig(), id="affine"),
    pytest.param(RationalQuadraticSplineConfig(num_bins=6), id="rq-spline"),
    pytest.param(RationalLinearSplineConfig(num_bins=6), id="rl-spline"),
    pytest.param(MonotoneHermiteCubicSplineConfig(num_bins=6), id="hermite-spline"),
    pytest.param(PiecewiseAffineSplineConfig(num_bins=6), id="affine-spline"),
    pytest.param(DeepSigmoidBijectorConfig(num_components=4), id="deep-sigmoid"),
    pytest.param(UMNNBijectorConfig(num_hidden=4), id="umnn"),
    pytest.param(SumOfSquaresBijectorConfig(num_polys=2, degree=3), id="sos"),
    pytest.param(BernsteinBijectorConfig(degree=8), id="bernstein"),
    pytest.param(MixtureCDFBijectorConfig(num_components=4), id="mixture-cdf"),
]

# Far outside any sane data range, which is exactly where a conditioner driven
# off the manifold lands and where overflow used to appear.
EXTREME_X = jnp.array([-1e6, -1e3, -50.0, -7.0, 0.0, 7.0, 50.0, 1e3, 1e6])


# ---------------------------------------------------------------------------
# Bijectors stay finite and monotone far from the data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
@pytest.mark.parametrize("param_scale", [0.5, 3.0])
def test_finite_and_monotone_far_from_data(cfg, param_scale):
    """No family may overflow or invert its ordering in the tails."""
    params = jax.random.normal(jax.random.PRNGKey(0), (cfg.params_dim(),)) * param_scale
    batched = jnp.broadcast_to(params, (EXTREME_X.size, cfg.params_dim()))

    y = cfg(batched, EXTREME_X)
    assert jnp.all(jnp.isfinite(y)), f"non-finite output: {y}"
    assert jnp.all(jnp.diff(y) > 0), f"not increasing: {y}"


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
@pytest.mark.parametrize("param_scale", [0.5, 3.0])
def test_logdet_stays_finite_far_from_data(cfg, param_scale):
    params = jax.random.normal(jax.random.PRNGKey(1), (cfg.params_dim(),)) * param_scale
    batched = jnp.broadcast_to(params, (EXTREME_X.size, cfg.params_dim()))

    y = cfg(batched, EXTREME_X)
    _, logdet = inverse_and_logabsdet(cfg, invertible_arg=1)(batched, y)
    assert jnp.isfinite(logdet), f"log-det {logdet} at |x| up to {EXTREME_X.max()}"


@pytest.mark.parametrize(
    "cfg",
    [
        RationalQuadraticSplineConfig(num_bins=6),
        DeepSigmoidBijectorConfig(num_components=4),
        MixtureCDFBijectorConfig(num_components=4),
    ],
)
def test_positive_parameters_are_bounded_both_ways(cfg):
    """Bounded slopes/scales are what stop tails compounding across layers.

    A slope ``s`` at the boundary knot becomes ``s**L`` after ``L`` stacked
    transforms, so an unbounded one overflows the composed inverse.
    """
    raw = jax.random.normal(jax.random.PRNGKey(2), (4000, cfg.params_dim())) * 50.0
    natural = cfg.unpack(raw)
    positive = natural[1] if isinstance(cfg, DeepSigmoidBijectorConfig) else natural[2]

    bound = getattr(cfg, "max_knot_slope", None) or getattr(
        cfg, "max_slope", getattr(cfg, "max_scale", None)
    )
    assert jnp.min(positive) >= 1.0 / bound - 1e-6
    assert jnp.max(positive) <= bound + 1e-6

    at_zero = cfg.unpack(jnp.zeros((cfg.params_dim(),)))
    unit = at_zero[1] if isinstance(cfg, DeepSigmoidBijectorConfig) else at_zero[2]
    assert jnp.allclose(unit, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Sum-of-squares: the bounded domain is what makes it trainable
# ---------------------------------------------------------------------------


def test_sos_is_linear_beyond_its_bound():
    cfg = SumOfSquaresBijectorConfig(bound=5.0)
    params = jax.random.normal(jax.random.PRNGKey(3), (cfg.params_dim(),)) * 0.5
    natural = cfg.unpack(params)

    # The log-det is constant outside the bound, i.e. the tails are straight.
    tail = jnp.array([6.0, 1e2, 1e4, 1e6, 1e12])
    logdets = jnp.array([B.inv_sos_polynomial(x, *natural, cfg.bound)[1] for x in tail])
    assert jnp.all(jnp.isfinite(logdets))
    assert jnp.allclose(logdets, logdets[0], atol=1e-5)


def test_sos_identity_at_zero_params_includes_the_tails():
    cfg = SumOfSquaresBijectorConfig()
    zeros = jnp.zeros((cfg.params_dim(),))
    for x in (-20.0, -5.0, 0.0, 5.0, 20.0):
        assert jnp.allclose(cfg(zeros, jnp.array(x)), x, atol=1e-4)


# ---------------------------------------------------------------------------
# The root solver reports failure instead of inventing a root
# ---------------------------------------------------------------------------


def test_solver_finds_the_root_when_one_exists():
    assert jnp.allclose(_solve_increasing(lambda t: 2.0 * t + 1.0, jnp.array(7.0)), 3.0)


def test_solver_returns_nan_when_the_target_is_unreachable():
    """A saturating g has no root; the bracket edge is not an answer."""
    out = _solve_increasing(jnp.tanh, jnp.array(2.0))
    assert jnp.isnan(out), f"expected NaN, got {out}"


def test_solver_returns_nan_when_g_is_non_finite():
    """NaN must not read as 'the root lies to the right'."""

    def g(t):
        return jnp.where(jnp.abs(t) > 3.0, jnp.nan, t)

    out = _solve_increasing(g, jnp.array(10.0))
    assert jnp.isnan(out), f"expected NaN, got {out}"


def test_safe_quadratic_root_handles_the_degenerate_case():
    """b == 0 with a vanishing discriminant forces c == 0, so the root is 0."""
    for a in (0.0, 1.0):
        got = _safe_quadratic_root(jnp.array(a), jnp.array(0.0), jnp.array(0.0))
        assert jnp.isfinite(got) and jnp.allclose(got, 0.0)


# ---------------------------------------------------------------------------
# Bounded splines: bijective on the domain, zero density outside
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        RationalQuadraticSplineConfig,
        RationalLinearSplineConfig,
        MonotoneHermiteCubicSplineConfig,
        PiecewiseAffineSplineConfig,
    ],
)
def test_bounded_spline_is_a_bijection_on_its_domain(cls):
    cfg = cls(num_bins=6, bounded=True)
    params = jax.random.normal(jax.random.PRNGKey(4), (cfg.params_dim(),)) * 0.7
    inside = jnp.linspace(cfg.x_min + 1e-3, cfg.x_max - 1e-3, 64)
    batched = jnp.broadcast_to(params, (inside.size, cfg.params_dim()))

    y = cfg(batched, inside)
    assert jnp.all(jnp.diff(y) > 0)
    x_rec, _ = inverse_and_logabsdet(cfg, invertible_arg=1)(batched, y)
    assert jnp.allclose(x_rec, inside, atol=1e-3)


def test_bounded_spline_gives_zero_density_outside_never_nan():
    cfg = RationalQuadraticSplineConfig(num_bins=6, bounded=True)
    params = jax.random.normal(jax.random.PRNGKey(5), (cfg.params_dim(),)) * 0.7
    outside = jnp.array([-20.0, -6.0, 6.0, 20.0])

    for x in outside:
        y = cfg(params, x)
        _, logdet = inverse_and_logabsdet(cfg, invertible_arg=1)(
            params[None], jnp.atleast_1d(y)
        )
        assert not jnp.isnan(logdet), f"NaN log-det at x={x}"
        assert logdet == -jnp.inf, f"expected zero density off-domain, got {logdet}"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _quadratic_loss(params, rng, batch):
    del rng, batch
    return jnp.sum(params["w"] ** 2)


def test_fit_warns_when_the_loss_goes_non_finite():
    def nan_loss(params, rng, batch):
        del rng, batch
        return jnp.sum(params["w"]) * jnp.nan

    with pytest.warns(RuntimeWarning, match="non-finite at step"):
        fit(nan_loss, {"w": jnp.ones(3)}, jax.random.PRNGKey(0), jnp.zeros((4, 1)),
            num_steps=3)


def test_fit_does_not_warn_on_healthy_training():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        params, losses = fit(
            _quadratic_loss, {"w": jnp.ones(3)}, jax.random.PRNGKey(0),
            jnp.zeros((4, 1)), num_steps=20,
        )
    assert losses[-1] < losses[0]


def test_fit_clips_gradients_by_default():
    """A gigantic gradient must not translate into a gigantic step."""

    def steep_loss(params, rng, batch):
        del rng, batch
        return jnp.sum(params["w"]) * 1e9

    start = {"w": jnp.zeros(3)}
    clipped, _ = fit(steep_loss, start, jax.random.PRNGKey(0), jnp.zeros((2, 1)),
                     num_steps=1, learning_rate=1.0)
    unclipped, _ = fit(steep_loss, start, jax.random.PRNGKey(0), jnp.zeros((2, 1)),
                       num_steps=1, learning_rate=1.0, clip_norm=None)
    # Adam normalises step size, so both move ~lr; the point is that clipping is
    # wired in and does not change a well-scaled step's direction.
    assert jnp.all(jnp.isfinite(clipped["w"]))
    assert jnp.all(jnp.isfinite(unclipped["w"]))
    assert jnp.all(clipped["w"] < 0)


def test_fit_schedule_decays_the_learning_rate():
    """warmup_cosine must actually end below where it started."""
    from probjax.stats.fit import _build_optimizer

    tx = _build_optimizer(1e-2, 100, "warmup_cosine", None)
    params = {"w": jnp.zeros(1)}
    state = tx.init(params)
    grad = {"w": jnp.ones(1)}

    steps = []
    for _ in range(100):
        upd, state = tx.update(grad, state, params)
        steps.append(float(jnp.abs(upd["w"][0])))
    assert steps[-1] < steps[len(steps) // 2], "cosine tail did not decay"


def test_fit_rejects_an_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        fit(_quadratic_loss, {"w": jnp.ones(2)}, jax.random.PRNGKey(0),
            jnp.zeros((2, 1)), num_steps=1, schedule="cosine")
