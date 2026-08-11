"""Tests for the flexible univariate families.

These exercise each new distribution *as a distribution*, independently of the
autoregressive model that motivated them: a family that does not integrate to
one, or whose sampler disagrees with its own CDF, is broken wherever it is used.
"""

import jax
import jax.numpy as jnp
import pytest

from probjax.stats import (
    histogram,
    logistic_mixture_kernel,
    mixture_kernel,
    spline_normal,
    tailed_histogram,
)

_KEY = jax.random.PRNGKey(0)


def _mixture_params(k=5):
    return dict(
        log_weights=jnp.array([0.5, -1.0, 0.3, 0.8, -0.2])[:k],
        locs=jnp.array([-2.0, -0.5, 0.0, 1.0, 2.5])[:k],
        scales=jnp.array([0.4, 0.8, 0.3, 1.2, 0.5])[:k],
    )


def _histogram_params(num_bins=8):
    return dict(
        logits=jax.random.normal(_KEY, (num_bins,)), low=-3.0, high=4.0
    )


def _tailed_params(num_bins=8):
    return dict(
        **_histogram_params(num_bins),
        tail_logit=-1.0,
        left_rate=0.8,
        right_rate=1.3,
    )


def _spline_params(num_bins=6):
    return dict(
        x_pos=jnp.linspace(-4.0, 4.0, num_bins + 1),
        y_pos=jnp.linspace(-4.0, 4.0, num_bins + 1) * 1.3 + 0.4,
        knot_slopes=jnp.linspace(0.6, 1.8, num_bins + 1),
    )


# (name, dist, params, integration range) -- the range covers essentially all
# the mass for the unbounded families.
FAMILIES = [
    pytest.param(mixture_kernel, _mixture_params(), (-20.0, 20.0), id="mixture"),
    pytest.param(
        logistic_mixture_kernel, _mixture_params(), (-80.0, 80.0), id="logistic-mixture"
    ),
    pytest.param(histogram, _histogram_params(), (-3.0, 4.0), id="histogram"),
    pytest.param(tailed_histogram, _tailed_params(), (-60.0, 60.0), id="tailed-histogram"),
    pytest.param(spline_normal, _spline_params(), (-40.0, 40.0), id="spline-normal"),
]


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_density_integrates_to_one(dist, params, rng_):
    lo, hi = rng_
    xs = jnp.linspace(lo, hi, 200_001)
    total = jnp.trapezoid(jnp.exp(dist.logpdf(xs, **params)), xs)
    assert jnp.allclose(total, 1.0, atol=1e-3), f"integrates to {float(total)}"


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_cdf_spans_the_unit_interval(dist, params, rng_):
    lo, hi = rng_
    span = dist.cdf(jnp.array(hi), **params) - dist.cdf(jnp.array(lo), **params)
    assert jnp.allclose(span, 1.0, atol=1e-3)


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_cdf_is_monotone(dist, params, rng_):
    lo, hi = rng_
    xs = jnp.linspace(lo, hi, 500)
    c = dist.cdf(xs, **params)
    assert jnp.all(jnp.diff(c) >= -1e-6)
    assert jnp.all((c >= -1e-6) & (c <= 1.0 + 1e-6))


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_quantile_roundtrips(dist, params, rng_):
    del rng_
    us = jnp.linspace(0.05, 0.95, 25)
    xs = dist.ppf(us, **params)
    assert jnp.all(jnp.isfinite(xs))
    assert jnp.allclose(dist.cdf(xs, **params), us, atol=1e-4)
    assert jnp.allclose(dist.ppf(dist.cdf(xs, **params), **params), xs, atol=1e-3)


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_sampler_agrees_with_its_own_cdf(dist, params, rng_):
    """KS-style check: the sampler and the CDF must describe one distribution."""
    del rng_
    samples = dist.rvs(jax.random.PRNGKey(1), **params, shape=(20_000,))
    assert samples.shape == (20_000,)
    assert jnp.all(jnp.isfinite(samples))

    grid = jnp.linspace(
        float(jnp.quantile(samples, 0.02)), float(jnp.quantile(samples, 0.98)), 40
    )
    empirical = jnp.mean(samples[:, None] <= grid[None, :], axis=0)
    ks = jnp.max(jnp.abs(empirical - dist.cdf(grid, **params)))
    assert ks < 0.03, f"KS statistic {float(ks)}"


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_batched_parameters_broadcast(dist, params, rng_):
    """A batch of parameters must give a batch of independent densities."""
    del rng_
    batched = jax.tree.map(lambda p: jnp.broadcast_to(p, (4,) + jnp.shape(p)), params)
    x = jnp.linspace(-1.0, 1.0, 4)

    lp = dist.logpdf(x, **batched)
    assert lp.shape == (4,)
    for i in range(4):
        single = dist.logpdf(x[i], **params)
        assert jnp.allclose(lp[i], single, atol=1e-5)

    samples = dist.rvs(jax.random.PRNGKey(2), **batched, shape=(6,))
    assert samples.shape == (6, 4)


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_works_under_jit_and_vmap(dist, params, rng_):
    del rng_
    x = jnp.linspace(-1.0, 1.0, 5)
    ref = dist.logpdf(x, **params)
    assert jnp.allclose(jax.jit(lambda a: dist.logpdf(a, **params))(x), ref, atol=1e-6)

    batched = jax.tree.map(lambda p: jnp.broadcast_to(p, (5,) + jnp.shape(p)), params)
    out = jax.vmap(lambda a, p: dist.logpdf(a, **p))(x, batched)
    assert jnp.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("dist,params,rng_", FAMILIES)
def test_logpdf_is_differentiable(dist, params, rng_):
    del rng_
    g = jax.grad(lambda p: jnp.sum(dist.logpdf(jnp.array([0.1, 0.3]), **p)))(params)
    leaves = jax.tree.leaves(g)
    assert leaves and all(jnp.all(jnp.isfinite(v)) for v in leaves)
    assert any(jnp.any(v != 0) for v in leaves), "no parameter affects the density"


# ---------------------------------------------------------------------------
# Family-specific properties
# ---------------------------------------------------------------------------


def test_histogram_has_zero_density_outside_its_domain():
    params = _histogram_params()
    outside = jnp.array([-10.0, -3.5, 4.5, 10.0])
    assert jnp.all(jnp.isneginf(histogram.logpdf(outside, **params)))
    assert histogram.support(**params).lower == params["low"]
    assert histogram.support(**params).upper == params["high"]


def test_tailed_histogram_keeps_unbounded_support():
    params = _tailed_params()
    outside = jnp.array([-40.0, -10.0, 20.0, 60.0])
    lp = tailed_histogram.logpdf(outside, **params)
    assert jnp.all(jnp.isfinite(lp)), "tails must carry mass"
    # and the tails decay
    assert lp[0] < lp[1] and lp[3] < lp[2]


def test_flat_histogram_is_uniform():
    num_bins = 10
    params = dict(logits=jnp.zeros(num_bins), low=-2.0, high=3.0)
    xs = jnp.linspace(-1.9, 2.9, 17)
    lp = histogram.logpdf(xs, **params)
    assert jnp.allclose(lp, -jnp.log(5.0), atol=1e-5)


def test_spline_with_identity_knots_is_standard_normal():
    num_bins = 6
    knots = jnp.linspace(-5.0, 5.0, num_bins + 1)
    params = dict(x_pos=knots, y_pos=knots, knot_slopes=jnp.ones(num_bins + 1))
    xs = jnp.linspace(-3.0, 3.0, 21)
    assert jnp.allclose(
        spline_normal.logpdf(xs, **params),
        jax.scipy.stats.norm.logpdf(xs),
        atol=1e-5,
    )


def test_mixture_recovers_a_single_component():
    """One component, or K identical ones, must equal the plain kernel."""
    x = jnp.linspace(-2.0, 2.0, 11)
    single = dict(
        log_weights=jnp.zeros(1), locs=jnp.array([0.5]), scales=jnp.array([2.0])
    )
    ref = jax.scipy.stats.norm.logpdf(x, 0.5, 2.0)
    assert jnp.allclose(mixture_kernel.logpdf(x, **single), ref, atol=1e-5)

    repeated = jax.tree.map(lambda p: jnp.repeat(p, 4), single)
    assert jnp.allclose(mixture_kernel.logpdf(x, **repeated), ref, atol=1e-5)


def test_mixture_ppf_reports_the_infinite_boundaries():
    """The solver must not pass off a bracket edge as the 0/1 quantile."""
    params = _mixture_params()
    assert jnp.isneginf(mixture_kernel.ppf(jnp.array(0.0), **params))
    assert jnp.isposinf(mixture_kernel.ppf(jnp.array(1.0), **params))


@pytest.mark.parametrize(
    "dist,hyper,expected",
    [
        (mixture_kernel, {"num_components": 7}, 7),
        (histogram, {"num_bins": 12}, 12),
        (tailed_histogram, {"num_bins": 12}, 12),
        (spline_normal, {"num_bins": 5}, 6),
    ],
)
def test_param_sizes_matches_the_accepted_shapes(dist, hyper, expected):
    sizes = dist.param_sizes(**hyper)
    assert set(sizes) == set(dist.parameters)
    vector = [n for n, s in sizes.items() if s > 1]
    assert vector, "every parameter reported as scalar"
    assert all(sizes[n] == expected for n in vector)
