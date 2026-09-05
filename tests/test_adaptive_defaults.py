"""Adaptive capacity rules, adaptive fit defaults, and data standardisation."""

import warnings

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import probjax.nn.generative.nflows as N
from probjax.nn.generative.autoregressive import MADE, MixtureAutoregressive
from probjax.nn.generative.autoregressive.config import MLPARConditionerConfig
from probjax.nn.generative.nflows.config import (
    MLPConditionerConfig,
    _DEFAULT_WIDTH,
    _WIDTH_MAX,
    _WIDTH_MIN,
    default_hidden_dims,
    resolve_hidden_dims,
)
from probjax.stats.fit import _resolve_batch_size, _resolve_num_steps


# ---------------------------------------------------------------------------
# The width rule
# ---------------------------------------------------------------------------


def test_without_a_data_hint_the_rule_is_the_measured_default():
    for out in (10, 100, 5000):
        assert default_hidden_dims(2, out) == (_DEFAULT_WIDTH, _DEFAULT_WIDTH)


def test_width_grows_with_data_not_with_dimension():
    """The measured direction: more data buys width, more dimensions do not."""
    out = 2450
    widths = [default_hidden_dims(50, out, n)[0] for n in (5_000, 50_000, 500_000)]
    assert widths == sorted(widths), widths
    assert widths[-1] > widths[0]

    # at fixed data, a bigger output does not inflate the width beyond the cap
    at_fixed_n = [default_hidden_dims(d, d * 49, 50_000)[0] for d in (2, 10, 50, 200)]
    assert max(at_fixed_n) <= max(_WIDTH_MIN, at_fixed_n[0]) or len(set(at_fixed_n)) <= 2


def test_width_is_capped_by_the_output_size():
    """A tiny problem must not get a huge conditioner just because n is large."""
    assert default_hidden_dims(1, 20, 10_000_000)[0] <= max(_WIDTH_MIN, 20)


def test_width_is_clipped_and_well_formed():
    for n in (1, 100, 10**9):
        dims = default_hidden_dims(5, 5000, n)
        assert isinstance(dims, tuple) and len(dims) >= 1
        assert all(isinstance(d, int) and d > 0 for d in dims)
        assert _WIDTH_MIN <= dims[0] <= _WIDTH_MAX


def test_explicit_sequence_wins_over_the_rule():
    assert resolve_hidden_dims((7, 9), 2, 1000, 10_000) == (7, 9)


def test_a_custom_callable_is_honoured():
    assert resolve_hidden_dims(lambda i, o, n: (o // 2,), 2, 40, None) == (20,)


def test_resolve_rejects_degenerate_widths():
    with pytest.raises(ValueError, match="non-empty and positive"):
        resolve_hidden_dims((), 2, 10)
    with pytest.raises(ValueError, match="non-empty and positive"):
        resolve_hidden_dims((0, 8), 2, 10)


def test_the_hint_reaches_the_built_flow():
    """A large-enough problem, so the output-size cap is not what binds."""

    def width(n):
        f = N.nsf(20, 2, nnx.Rngs(0), conditioner=MLPConditionerConfig(num_examples=n))
        return f.transformation.layers[0].masked_mlp.layers[0].kernel.shape[1]

    assert width(2_000_000) > width(5_000)


def test_the_hint_reaches_the_built_ar_model():
    from probjax.nn.generative.autoregressive import SplineAutoregressive

    def width(n):
        m = SplineAutoregressive(
            20, nnx.Rngs(0), conditioner=MLPARConditionerConfig(num_examples=n)
        )
        return m.conditioner.net.layers[0].kernel.shape[1]

    assert width(2_000_000) > width(5_000)


# ---------------------------------------------------------------------------
# Adaptive fit defaults
# ---------------------------------------------------------------------------


def test_auto_batch_size_and_step_count():
    assert _resolve_batch_size("auto", 100) == 100      # smaller than the cap
    assert _resolve_batch_size("auto", 10**6) == 512
    assert _resolve_batch_size(None, 10**6) is None     # full batch, explicitly
    assert _resolve_batch_size(64, 10**6) == 64

    steps = [_resolve_num_steps("auto", n, 512) for n in (1_000, 100_000, 10**7)]
    assert steps == sorted(steps)
    assert 1000 <= min(steps) and max(steps) <= 20_000
    assert _resolve_num_steps(37, 10**6, 512) == 37


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

_SCALE = jnp.array([20.0, 5.0])
_SHIFT = jnp.array([100.0, -50.0])
#: entropy of N(shift, diag(scale)^2)
_ENTROPY = float(jnp.sum(jnp.log(_SCALE)) + 2 * 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def _badly_scaled(key, n=4000):
    return jax.random.normal(key, (n, 2)) * _SCALE + _SHIFT


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda **k: N.nsf(2, 4, nnx.Rngs(0), **k), id="flow"),
        pytest.param(
            lambda **k: MixtureAutoregressive(2, nnx.Rngs(0), num_components=8, **k),
            id="ar",
        ),
    ],
)
def test_standardisation_rescues_badly_scaled_data(build):
    """The spline domain is [-5, 5]; data at mean 100 lands wholly in the tails."""
    train, test = _badly_scaled(jax.random.PRNGKey(0)), _badly_scaled(jax.random.PRNGKey(1))

    on = build(standardize=True)
    on.fit(jax.random.PRNGKey(2), train, num_steps=800, batch_size=512)
    nll_on = float(-jnp.mean(jax.vmap(on._logpdf)(test)))

    off = build(standardize=False)
    off.fit(jax.random.PRNGKey(2), train, num_steps=800, batch_size=512)
    nll_off = float(-jnp.mean(jax.vmap(off._logpdf)(test)))

    assert nll_on < nll_off - 1.0, (nll_on, nll_off)
    assert nll_on < _ENTROPY + 0.5, f"nll {nll_on} vs entropy {_ENTROPY}"


def test_standardised_logpdf_is_a_density_in_the_original_space():
    """The Jacobian term is what makes this true; drop it and the mass is off."""
    train = _badly_scaled(jax.random.PRNGKey(0))
    model = N.nsf(2, 4, nnx.Rngs(0))
    model.fit(jax.random.PRNGKey(1), train, num_steps=600, batch_size=512)

    proposal_scale = 4.0 * _SCALE
    z = jax.random.normal(jax.random.PRNGKey(2), (60_000, 2)) * proposal_scale + _SHIFT
    log_q = jnp.sum(jax.scipy.stats.norm.logpdf(z, _SHIFT, proposal_scale), -1)
    lp = jax.vmap(model._logpdf)(z)
    finite = jnp.isfinite(lp)
    mass = jnp.mean(jnp.where(finite, jnp.exp(lp - log_q), 0.0))
    assert 0.9 < float(mass) < 1.1, float(mass)


def test_samples_come_back_in_the_original_units():
    train = _badly_scaled(jax.random.PRNGKey(0))
    model = N.nsf(2, 4, nnx.Rngs(0))
    model.fit(jax.random.PRNGKey(1), train, num_steps=600, batch_size=512)

    samples = model.as_dist().rvs(jax.random.PRNGKey(2), (4000,))
    assert jnp.allclose(jnp.mean(samples, 0), _SHIFT, atol=0.2 * _SCALE)
    assert jnp.allclose(jnp.std(samples, 0), _SCALE, rtol=0.35)


def test_standardisation_is_fitted_once_and_only_once():
    """A second fit must not move the coordinate system under a trained model."""
    model = N.nsf(2, 2, nnx.Rngs(0))
    assert not model.is_standardized

    model.fit(jax.random.PRNGKey(0), _badly_scaled(jax.random.PRNGKey(0)), num_steps=50)
    shift, scale = model.standardization
    assert model.is_standardized
    assert jnp.allclose(shift, _SHIFT, atol=2.0)

    model.fit(jax.random.PRNGKey(1), jnp.zeros((100, 2)) + 999.0, num_steps=50)
    assert jnp.allclose(model.standardization[0], shift)
    assert jnp.allclose(model.standardization[1], scale)


def test_standardization_can_be_set_explicitly_and_is_validated():
    model = N.nsf(2, 2, nnx.Rngs(0))
    model.set_standardization(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert model.is_standardized
    assert jnp.allclose(model.standardization[1], jnp.array([3.0, 4.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        model.set_standardization(jnp.zeros(2), jnp.zeros(2))


def test_disabled_standardisation_leaves_the_model_untouched():
    model = N.nsf(2, 2, nnx.Rngs(0), standardize=False)
    model.fit(jax.random.PRNGKey(0), _badly_scaled(jax.random.PRNGKey(0)), num_steps=50)
    assert not model.is_standardized
    assert jnp.allclose(model.standardization[0], 0.0)
    assert jnp.allclose(model.standardization[1], 1.0)


def test_constant_columns_do_not_divide_by_zero():
    data = jnp.stack([jax.random.normal(jax.random.PRNGKey(0), (500,)), jnp.full((500,), 7.0)], -1)
    model = N.nsf(2, 2, nnx.Rngs(0))
    model.fit(jax.random.PRNGKey(1), data, num_steps=50)
    assert jnp.all(jnp.isfinite(model.standardization[1]))
    assert jnp.all(model.standardization[1] > 0)
    assert jnp.all(jnp.isfinite(model._logpdf(data[:16])))


def test_discrete_heads_do_not_standardise():
    """Shifting integer labels would destroy them."""
    from probjax.nn.generative.autoregressive import CategoricalAutoregressive

    model = CategoricalAutoregressive(3, nnx.Rngs(0), num_categories=4)
    assert not model.standardize
    tokens = jax.random.randint(jax.random.PRNGKey(0), (64, 3), 0, 4)
    model.fit(jax.random.PRNGKey(1), tokens, num_steps=50)
    assert not model.is_standardized
