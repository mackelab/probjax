"""Tests for the natural-parameter bijections in ``probjax.stats.bijective``.

Everything here talks to the stats layer directly, with already-constrained
natural parameters and ``x`` first. The unconstrained-vector side is covered
separately in ``test_nflow_configs.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.stats.bijective import (
    affine,
    bernstein,
    deep_sigmoid,
    inv_bernstein,
    inv_deep_sigmoid,
    inv_mixture_cdf,
    inv_monotone_hermite_cubic_spline,
    inv_piecewise_affine_spline,
    inv_rational_linear_spline,
    inv_rational_quadratic_spline,
    inv_sos_polynomial,
    inv_unconstrained_monotone,
    mixture_cdf,
    monotone_hermite_cubic_spline,
    monotone_hermite_cubic_spline_and_logdet,
    piecewise_affine_spline,
    piecewise_affine_spline_and_logdet,
    rational_linear_spline,
    rational_linear_spline_and_logdet,
    rational_quadratic_spline,
    rational_quadratic_spline_and_logdet,
    rotate,
    shift,
    sos_polynomial,
    unconstrained_monotone,
)

# (forward, inverse, value_and_logdet, takes_knot_slopes)
SPLINES = [
    pytest.param(
        rational_quadratic_spline,
        inv_rational_quadratic_spline,
        rational_quadratic_spline_and_logdet,
        True,
        id="rational_quadratic",
    ),
    pytest.param(
        rational_linear_spline,
        inv_rational_linear_spline,
        rational_linear_spline_and_logdet,
        True,
        id="rational_linear",
    ),
    pytest.param(
        monotone_hermite_cubic_spline,
        inv_monotone_hermite_cubic_spline,
        monotone_hermite_cubic_spline_and_logdet,
        True,
        id="monotone_hermite_cubic",
    ),
    pytest.param(
        piecewise_affine_spline,
        inv_piecewise_affine_spline,
        piecewise_affine_spline_and_logdet,
        False,
        id="piecewise_affine",
    ),
]


_MIN_BIN = 1e-2


def _knots(rng, num_bins, scale=1.0):
    """Random but valid natural spline parameters.

    Bins are floored at ``_MIN_BIN`` of the total width, mirroring what the
    bijector configs enforce: an unfloored softmax can produce bins narrow
    enough that inverting inside them loses most of the float32 mantissa.
    """
    rng_x, rng_y, rng_s = jax.random.split(rng, 3)

    def positions(key, lo, hi):
        raw = jax.random.normal(key, (num_bins,)) * scale
        widths = _MIN_BIN + (1.0 - _MIN_BIN * num_bins) * jax.nn.softmax(raw)
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)])
        return lo + (hi - lo) * cum

    x_pos = positions(rng_x, -3.0, 3.0)
    y_pos = positions(rng_y, -3.0, 3.0)
    slopes = jax.nn.softplus(jax.random.normal(rng_s, (num_bins + 1,)) * scale) + 1e-3
    return x_pos, y_pos, slopes


# ---------------------------------------------------------------------------
# Splines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("num_bins", [4, 8, 16])
@pytest.mark.parametrize("forward,backward,fwd_logdet,has_slopes", SPLINES)
def test_spline_roundtrip(
    seed, scale, num_bins, forward, backward, fwd_logdet, has_slopes
):
    rng_x, rng_k = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(rng_x)
    x_pos, y_pos, slopes = _knots(rng_k, num_bins, scale)
    knots = (x_pos, y_pos, slopes) if has_slopes else (x_pos, y_pos)

    y = forward(x, *knots)
    x_rec, logdet = backward(y, *knots)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-4)
    assert jnp.isfinite(logdet).all()
    assert jnp.isfinite(y).all()


@pytest.mark.parametrize("num_bins", [4, 16])
@pytest.mark.parametrize("forward,backward,fwd_logdet,has_slopes", SPLINES)
def test_spline_logdet_matches_autodiff(
    num_bins, forward, backward, fwd_logdet, has_slopes
):
    rng_x, rng_k = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x)
    x_pos, y_pos, slopes = _knots(rng_k, num_bins)
    knots = (x_pos, y_pos, slopes) if has_slopes else (x_pos, y_pos)

    y, logdet_fwd = fwd_logdet(x, *knots)
    expected = jnp.log(jnp.abs(jax.grad(lambda t: forward(t, *knots))(x)))

    assert jnp.allclose(y, forward(x, *knots), atol=1e-6)
    assert jnp.allclose(logdet_fwd, expected, atol=1e-4)

    # the inverse's log-det is the negation of the forward's
    _, logdet_inv = backward(y, *knots)
    assert jnp.allclose(logdet_inv, -logdet_fwd, atol=1e-4)


@pytest.mark.parametrize("forward,backward,fwd_logdet,has_slopes", SPLINES)
def test_spline_value_and_logdet_is_registered(
    forward, backward, fwd_logdet, has_slopes
):
    """The forward (value, logdet) pair is reachable off the custom_inverse."""
    x_pos, y_pos, slopes = _knots(jax.random.PRNGKey(1), 6)
    knots = (x_pos, y_pos, slopes) if has_slopes else (x_pos, y_pos)
    x = jnp.array(0.3)

    from_attr = forward.value_and_logdet(x, *knots)
    from_fn = fwd_logdet(x, *knots)
    assert jnp.allclose(jnp.asarray(from_attr), jnp.asarray(from_fn))


@pytest.mark.parametrize("forward,backward,fwd_logdet,has_slopes", SPLINES)
def test_spline_inverse_uses_registered_rule(forward, backward, fwd_logdet, has_slopes):
    """``inverse_and_logabsdet`` picks up the analytic inverse, not a re-derivation."""
    x_pos, y_pos, slopes = _knots(jax.random.PRNGKey(2), 8)
    knots = (x_pos, y_pos, slopes) if has_slopes else (x_pos, y_pos)
    x = jnp.array(-0.7)
    y = forward(x, *knots)

    x_rec, logdet = inverse_and_logabsdet(forward, invertible_arg=0)(y, *knots)
    x_ref, logdet_ref = backward(y, *knots)

    assert jnp.allclose(x_rec, x_ref, atol=1e-6)
    assert jnp.allclose(logdet, logdet_ref, atol=1e-6)


@pytest.mark.parametrize("forward,backward,fwd_logdet,has_slopes", SPLINES)
def test_spline_bounded_tails_clamp(forward, backward, fwd_logdet, has_slopes):
    """With explicit bounds the tails map linearly onto ``[y_min, y_max]``."""
    num_bins = 6
    x_pos = jnp.linspace(-2.0, 2.0, num_bins + 1)
    y_pos = jnp.linspace(-2.0, 2.0, num_bins + 1)
    slopes = jnp.ones((num_bins + 1,))
    knots = (x_pos, y_pos, slopes) if has_slopes else (x_pos, y_pos)
    bounds = dict(x_min=-5.0, x_max=5.0, y_min=-4.0, y_max=4.0)

    assert jnp.allclose(forward(jnp.array(6.0), *knots, **bounds), 4.0, atol=1e-5)
    assert jnp.allclose(forward(jnp.array(-6.0), *knots, **bounds), -4.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Monotone networks
# ---------------------------------------------------------------------------


def _monotone_cases(k=4):
    log_w = jax.nn.log_softmax(jnp.array([0.4, -0.2, 0.1, 0.3]))
    return {
        "deep_sigmoid": (
            deep_sigmoid,
            inv_deep_sigmoid,
            (log_w, jnp.full((k,), 0.9), jnp.linspace(-1.0, 1.0, k)),
        ),
        "mixture_cdf": (
            mixture_cdf,
            inv_mixture_cdf,
            (log_w, jnp.linspace(-1.0, 1.0, k), jnp.full((k,), 0.8)),
        ),
        "unconstrained_monotone": (
            unconstrained_monotone,
            inv_unconstrained_monotone,
            (
                jnp.full((k,), 0.3),
                jnp.zeros((k,)),
                jnp.full((k,), 0.2),
                jnp.array(0.1),
                jnp.array(0.0),
            ),
        ),
        "sos_polynomial": (
            sos_polynomial,
            inv_sos_polynomial,
            (jnp.array([[1.0, 0.1, 0.0, 0.0], [0.3, 0.0, 0.05, 0.0]]), jnp.array(0.0)),
        ),
        "bernstein": (bernstein, inv_bernstein, (jnp.linspace(0.0, 1.0, 9),)),
    }


@pytest.mark.parametrize("name", list(_monotone_cases()))
def test_monotone_roundtrip(name):
    forward, backward, natural = _monotone_cases()[name]
    x = jnp.array(0.37)

    y = forward(x, *natural)
    x_rec, logdet = backward(y, *natural)

    assert jnp.allclose(x, x_rec, atol=1e-5)
    assert jnp.isfinite(logdet).all()


@pytest.mark.parametrize("name", list(_monotone_cases()))
def test_monotone_logdet_is_analytic(name):
    """The registered log-det matches autodiff through the *analytic* direction.

    The forward is a bisection and must not be differentiated; the inverse is
    closed form and is what training actually uses.
    """
    forward, backward, natural = _monotone_cases()[name]
    y = jnp.array(0.61)

    _, logdet = backward(y, *natural)
    expected = jnp.log(jnp.abs(jax.grad(lambda t: backward(t, *natural)[0])(y)))
    assert jnp.allclose(logdet, expected, atol=1e-5)


@pytest.mark.parametrize("name", list(_monotone_cases()))
def test_monotone_is_increasing(name):
    forward, backward, natural = _monotone_cases()[name]
    ys = jnp.linspace(-2.0, 2.0, 32)
    values = jax.vmap(lambda t: backward(t, *natural)[0])(ys)
    assert jnp.all(jnp.diff(values) > 0.0)


@pytest.mark.parametrize("name", list(_monotone_cases()))
def test_monotone_broadcasts_over_batch(name):
    forward, backward, natural = _monotone_cases()[name]
    ys = jnp.linspace(-1.0, 1.0, 5)
    batched, logdets = backward(ys, *natural)
    assert batched.shape == ys.shape
    assert logdets.shape == ys.shape
    for i, y in enumerate(ys):
        single, single_logdet = backward(y, *natural)
        assert jnp.allclose(batched[i], single, atol=1e-6)
        assert jnp.allclose(logdets[i], single_logdet, atol=1e-6)


# ---------------------------------------------------------------------------
# Affine family
# ---------------------------------------------------------------------------


def test_affine():
    rng_l, rng_s, rng_x = jax.random.split(jax.random.PRNGKey(0), 3)
    loc = jax.random.normal(rng_l, (10, 10))
    scale = jax.nn.softplus(jax.random.normal(rng_s, (10, 10))) + 1e-3
    x = jax.random.normal(rng_x, (10, 10))

    y = affine(x, loc, scale)
    assert y.shape == x.shape
    assert jnp.allclose(y, loc + scale * x)

    x_rec = inverse(affine, invertible_arg=0)(y, loc, scale)
    assert jnp.allclose(x, x_rec, atol=1e-4)


def test_shift():
    rng_l, rng_x = jax.random.split(jax.random.PRNGKey(0))
    loc = jax.random.normal(rng_l, (10, 10))
    x = jax.random.normal(rng_x, (10, 10))

    y = shift(x, loc)
    assert y.shape == x.shape

    x_rec = inverse(shift, invertible_arg=0)(y, loc)
    assert jnp.allclose(x, x_rec, atol=1e-4)


def test_rotate():
    theta = 0.7
    R = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    x = jnp.array([1.0, 2.0])

    y = rotate(x, R)
    x_rec, logdet = rotate.inv_and_logdet(y, R)

    assert jnp.allclose(x, x_rec, atol=1e-5)
    # an orthogonal map is volume preserving
    assert jnp.allclose(jnp.asarray(logdet), 0.0)
