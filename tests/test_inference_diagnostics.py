"""BlackJAX 1.7.x diagnostics parity (ported from blackjax.diagnostics)."""

import functools
from typing import NamedTuple

import blackjax.diagnostics as bj_diagnostics
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.vi.svgd import rbf_kernel

from probjax.inference import (
    DivergenceConcentrationReport,
    divergence_concentration,
    divergence_concentration_from_counts,
    effective_sample_size,
    ess_bulk,
    ess_tail,
    format_divergence_warning,
    imq_kernel,
    kernelized_stein_discrepancy,
    pareto_khat,
    potential_scale_reduction,
    rhat,
)


def _chains():
    key = jax.random.key(0)
    return jax.random.normal(key, (4, 100, 2))


def test_diagnostics_match_blackjax():
    chains = _chains()
    for probjax_fn, bj_fn in [
        (effective_sample_size, bj_diagnostics.effective_sample_size),
        (potential_scale_reduction, bj_diagnostics.potential_scale_reduction),
        (rhat, bj_diagnostics.rhat),
        (ess_bulk, bj_diagnostics.ess_bulk),
        (ess_tail, bj_diagnostics.ess_tail),
        (pareto_khat, bj_diagnostics.pareto_khat),
    ]:
        got, want = probjax_fn(chains), bj_fn(chains)
        assert got.shape == want.shape
        assert jnp.allclose(got, want, equal_nan=True)


def test_degenerate_chain_ess_is_zero():
    chains = jnp.ones((4, 50, 1))
    assert jnp.all(effective_sample_size(chains) == 0)


def test_divergence_concentration_report():
    is_divergent = jnp.zeros((4, 100), dtype=bool).at[0, :10].set(True)
    report = divergence_concentration(is_divergent)
    assert isinstance(report, DivergenceConcentrationReport)
    assert isinstance(format_divergence_warning(report), str)

    counts = jnp.array([10, 0, 0, 0])
    from_counts = divergence_concentration_from_counts(counts, 100)
    assert from_counts.num_chains == 4
    assert from_counts.total_divergences == 10


def _rbf_reference(samples, scores, length_scale):
    # Closed-form RBF derivatives, evaluated in NumPy independently of the
    # implementation's autodiff (mirrors upstream's reference).
    result = 0.0
    d = samples.shape[1]
    for x, score_x in zip(samples, scores, strict=True):
        for y, score_y in zip(samples, scores, strict=True):
            delta = x - y
            distance_sq = np.dot(delta, delta)
            value = np.exp(-distance_sq / length_scale)
            result += value * (
                np.dot(score_x, score_y)
                + 2 * np.dot(score_x - score_y, delta) / length_scale
                + 2 * d / length_scale
                - 4 * distance_sq / length_scale**2
            )
    return result / len(samples) ** 2


@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_ksd_matches_closed_form_rbf(length_scale):
    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    mean = np.array([0.5, -0.25])
    precision = np.array([[2.0, 0.25], [0.25, 1.0]])
    score = lambda x: -jnp.asarray(precision) @ (x - mean)
    kernel = functools.partial(rbf_kernel, length_scale=length_scale)
    expected = _rbf_reference(samples, -(samples - mean) @ precision, length_scale)
    actual = jax.jit(
        functools.partial(
            kernelized_stein_discrepancy, grad_logdensity_fn=score, kernel=kernel
        )
    )(samples)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_ksd_diagonal_included_and_squared():
    samples = jnp.zeros((1, 3))
    result = kernelized_stein_discrepancy(
        samples, lambda x: -x, functools.partial(rbf_kernel, length_scale=2.0)
    )
    # At the Gaussian mode, the only contribution is trace(d_x d_y k)=2d/L.
    np.testing.assert_allclose(result, 3.0)


def test_ksd_permutation_and_normalizing_constant():
    samples = jnp.array([[-1.0], [0.5], [2.0]])
    logdensity = lambda x: -jnp.sum(x**2) / 2
    result = kernelized_stein_discrepancy(samples, jax.grad(logdensity), rbf_kernel)
    other = kernelized_stein_discrepancy(
        samples[::-1], jax.grad(lambda x: logdensity(x) + 13.0), rbf_kernel
    )
    np.testing.assert_allclose(result, other, rtol=2e-6)


def test_ksd_pytree_matches_flat():
    class Position(NamedTuple):
        a: jnp.ndarray
        b: jnp.ndarray

    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    score = lambda x: -x
    tree_score = lambda t: jax.tree.map(lambda v: -v, t)
    a, b = jnp.asarray(samples[:, 0]), jnp.asarray(samples[:, 1])

    for statistic in ("v", "u"):
        expected = kernelized_stein_discrepancy(
            samples, score, rbf_kernel, statistic=statistic
        )
        for pytree_samples in ({"a": a, "b": b}, (a, b), Position(a, b)):
            actual = kernelized_stein_discrepancy(
                pytree_samples, tree_score, rbf_kernel, statistic=statistic
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_ksd_u_statistic_excludes_diagonal():
    samples = jnp.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    score = lambda x: -x
    n = samples.shape[0]
    v = kernelized_stein_discrepancy(samples, score, rbf_kernel, statistic="v")
    u = kernelized_stein_discrepancy(samples, score, rbf_kernel, statistic="u")
    diag_sum = sum(
        kernelized_stein_discrepancy(samples[i : i + 1], score, rbf_kernel)
        for i in range(n)
    )
    # n²·V − diag_sum == n·(n−1)·U
    np.testing.assert_allclose(n * (n - 1) * u, n**2 * v - diag_sum, rtol=2e-5)


def test_ksd_rejects_invalid_statistic():
    with pytest.raises(ValueError, match="statistic"):
        kernelized_stein_discrepancy(
            jnp.ones((2, 2)), lambda x: -x, rbf_kernel, statistic="invalid"
        )


def test_imq_kernel_formula():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.0, 0.0])
    np.testing.assert_allclose(imq_kernel(x, y), (1.0 + 5.0) ** -0.5)
    np.testing.assert_allclose(imq_kernel(x, x), 1.0)
