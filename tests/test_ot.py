# ruff: noqa: B023
"""Log-domain Sinkhorn, MMD, KDE and minibatch-OT pairing (probjax.nn.ot)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from probjax.nn.ot import (
    entropic_map,
    gaussian_kde_logpdf,
    gaussian_kde_logpdf_naive,
    mmd,
    mmd_naive,
    ot_pairing,
    sample_conditional,
    sinkhorn,
    sinkhorn_divergence,
    transport_plan_dense,
)


def _dense_sinkhorn(x, y, eps, iters):
    """Float64 dense log-domain reference (same update order as the solver)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    a = np.full(len(x), 1 / len(x))
    b = np.full(len(y), 1 / len(y))
    c = ((x[:, None] - y[None]) ** 2).sum(-1)
    f = np.zeros(len(x))
    g = np.zeros(len(y))
    for _ in range(iters):
        g = -eps * logsumexp(np.log(a)[:, None] + (f[:, None] - c) / eps, axis=0)
        f = -eps * logsumexp(np.log(b)[None] + (g[None] - c) / eps, axis=1)
    p = np.exp(np.log(a)[:, None] + np.log(b)[None] + (f[:, None] + g[None] - c) / eps)
    return f, g, a @ f + b @ g, p


@pytest.fixture(scope="module")
def clouds():
    k = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k[0], (120, 3))
    y = jax.random.normal(k[1], (90, 3)) * 0.7 + 1.0
    return x, y


@pytest.mark.parametrize("eps", [0.5, 0.1])
def test_sinkhorn_matches_dense_reference(clouds, eps):
    x, y = clouds
    f, g, cost, p = _dense_sinkhorn(x, y, eps, 300)
    out = jax.jit(lambda x, y: sinkhorn(x, y, epsilon=eps, num_iters=300))(x, y)
    assert abs(float(out.reg_ot_cost) - cost) < 1e-4
    assert np.max(np.abs(np.asarray(out.f) - f)) < 1e-4
    assert np.max(np.abs(np.asarray(out.g) - g)) < 1e-4
    assert np.max(np.abs(np.asarray(transport_plan_dense(out)) - p)) < 1e-6


def test_sinkhorn_gradient_implicit_and_unrolled(clouds):
    x, y = clouds
    eps = 0.3
    out = sinkhorn(x, y, epsilon=eps, num_iters=300)
    p = np.asarray(transport_plan_dense(out), np.float64)
    a = np.full(len(x), 1 / len(x))
    # d OT / d x_i = 2 (a_i x_i - sum_j P_ij y_j) (envelope theorem).
    ref = 2 * (a[:, None] * np.asarray(x) - p @ np.asarray(y))
    for mode in ["implicit", "unroll"]:
        gx = jax.grad(
            lambda x: (
                sinkhorn(
                    x, y, epsilon=eps, num_iters=300, differentiate=mode
                ).reg_ot_cost
            )
        )(x)
        rel = np.max(np.abs(np.asarray(gx) - ref)) / np.max(np.abs(ref))
        assert rel < 1e-4, mode


def test_sinkhorn_tolerance_stops_early(clouds):
    x, y = clouds
    out = sinkhorn(x, y, epsilon=1.0, num_iters=1000, tol=1e-4, check_every=5)
    assert int(out.num_iters) < 1000
    assert bool(out.converged)
    full = sinkhorn(x, y, epsilon=1.0, num_iters=400)
    assert abs(float(out.reg_ot_cost) - float(full.reg_ot_cost)) < 1e-4


def test_sinkhorn_weights_and_relative_epsilon(clouds):
    x, y = clouds
    a = jnp.linspace(1.0, 2.0, x.shape[0])
    out = sinkhorn(x, y, a, num_iters=200)
    p = np.asarray(transport_plan_dense(out))
    np.testing.assert_allclose(p.sum(1), np.asarray(a / a.sum()), atol=1e-4)
    np.testing.assert_allclose(p.sum(0), 1 / y.shape[0], atol=1e-6)


def test_entropic_map_and_sampling(clouds):
    x, y = clouds
    out = sinkhorn(x, y, epsilon=0.3, num_iters=300)
    p = np.asarray(transport_plan_dense(out))
    want = (p @ np.asarray(y)) / p.sum(1, keepdims=True)
    np.testing.assert_allclose(np.asarray(entropic_map(out)), want, atol=1e-5)
    # Monte-Carlo check of the conditional sampler (small blocks force the
    # running-argmax path over several key blocks).
    keys = jax.random.split(jax.random.key(1), 400)
    idx = jax.vmap(lambda k: sample_conditional(out, k, block_q=32, block_k=16))(keys)
    emp = np.zeros_like(p)
    np.add.at(emp, (np.tile(np.arange(len(x)), 400), np.asarray(idx).ravel()), 1)
    emp /= 400
    cond = p / p.sum(1, keepdims=True)
    assert np.abs(emp - cond).sum(1).mean() < 0.35  # TV-ish, 400 draws per row
    assert np.max(np.abs(emp.mean(0) - cond.mean(0))) < 0.01


def test_sinkhorn_divergence(clouds):
    x, y = clouds
    assert abs(float(sinkhorn_divergence(x, x, num_iters=50))) < 1e-5
    assert float(sinkhorn_divergence(x, y, num_iters=200)) > 0.1


@pytest.mark.parametrize("kernel", ["rbf", "imq"])
@pytest.mark.parametrize("unbiased", [True, False])
def test_mmd_matches_naive(clouds, kernel, unbiased):
    x, y = clouds
    bw = [0.5, 1.0, 2.0]
    got = mmd(x, y, kernel=kernel, bandwidth=bw, unbiased=unbiased)
    want = mmd_naive(x, y, kernel=kernel, bandwidth=bw, unbiased=unbiased)
    assert abs(float(got) - float(want)) < 1e-5 * max(1.0, abs(float(want)))
    g = jax.grad(lambda x: mmd(x, y, kernel=kernel, bandwidth=bw, unbiased=unbiased))(x)
    g_ref = jax.grad(
        lambda x: mmd_naive(x, y, kernel=kernel, bandwidth=bw, unbiased=unbiased)
    )(x)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), atol=1e-6)


def test_kde_logpdf_matches_naive_and_stays_finite():
    k = jax.random.split(jax.random.key(3), 2)
    data = jax.random.normal(k[0], (200, 32))
    x = jax.random.normal(k[1], (50, 32)) * 3.0
    got = gaussian_kde_logpdf(x, data, bandwidth=0.2)
    want = gaussian_kde_logpdf_naive(x, data, bandwidth=0.2)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5)
    # The density itself underflows in fp32 here; the log-density does not.
    assert np.all(np.exp(np.asarray(got)) == 0.0)
    assert np.all(np.isfinite(np.asarray(got)))


def test_exact_pairing_is_optimal_permutation():
    k = jax.random.split(jax.random.key(4), 2)
    x0 = jax.random.normal(k[0], (64, 2))
    x1 = jax.random.normal(k[1], (64, 2)) + 3.0
    idx = np.asarray(jax.jit(ot_pairing)(x0, x1))
    assert sorted(idx.tolist()) == list(range(64))
    c = ((np.asarray(x1)[:, None] - np.asarray(x0)[None]) ** 2).sum(-1)
    r, cc = linear_sum_assignment(c)
    assert c[np.arange(64), idx].sum() == pytest.approx(c[r, cc].sum(), rel=1e-6)
    # vmap over independent minibatches
    batched = jax.vmap(ot_pairing)(jnp.stack([x0, x1]), jnp.stack([x1, x0]))
    assert batched.shape == (2, 64)


def test_sinkhorn_pairing_reduces_cost():
    k = jax.random.split(jax.random.key(5), 2)
    x0 = jax.random.normal(k[0], (128, 2))
    x1 = jax.random.normal(k[1], (128, 2)) * 0.5 + 2.0
    c = ((np.asarray(x1)[:, None] - np.asarray(x0)[None]) ** 2).sum(-1)
    idx = np.asarray(
        ot_pairing(x0, x1, jax.random.key(0), method="sinkhorn", num_iters=200)
    )
    r, cc = linear_sum_assignment(c)
    paired = c[np.arange(128), idx].mean()
    assert paired < 0.9 * np.trace(c) / 128
    assert paired < 1.5 * c[r, cc].mean()


class _TinyNet(nnx.Module):
    def __init__(self):
        self.proj = nnx.Linear(3, 2, rngs=nnx.Rngs(0))

    def __call__(self, t, x, **kw):
        return self.proj(
            jnp.concatenate([x, jnp.broadcast_to(t, x[..., :1].shape)], -1)
        )


def test_flow_matcher_ot_coupling_option():
    from probjax.nn.generative.flow_matching import LinearFlow

    data = jax.random.normal(jax.random.key(1), (32, 2)) + 4.0
    base = LinearFlow(_TinyNet(), event_spec=2)
    plain = float(base.loss(jax.random.key(0), data))
    # Default is unchanged (no coupling) and reproducible.
    assert (
        float(LinearFlow(_TinyNet(), event_spec=2).loss(jax.random.key(0), data))
        == plain
    )
    for method in ["exact", "sinkhorn"]:
        model = LinearFlow(_TinyNet(), event_spec=2, ot_coupling=method)
        val = jax.jit(lambda d: model.loss(jax.random.key(0), d))(data)
        assert np.isfinite(float(val))
        assert float(val) != pytest.approx(plain)
    with pytest.raises(ValueError):
        LinearFlow(_TinyNet(), event_spec=2, ot_coupling="hungarian")
