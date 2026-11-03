"""Regression coverage for the October 2026 parallel bug-hunt findings.

Each test pins one confirmed defect from the hunt so it cannot silently return.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sp

from probjax.core import inverse_and_logabsdet
from probjax.inference import (
    LinearGaussianModel,
    kalman_filter,
    rank_reduced_kalman_filter,
    run,
    smooth,
)
from probjax.inference.mcmc import adjusted_mclmc_dynamic, dynamic_hmc
from probjax.inference.mcmc._dynamic_stepping import (
    random_trajectory_length_fns,
)
from probjax.inference.smc.ess import ess
from probjax.inference.smc.tempering import _solve_temperature
from probjax.stats import beta, categorical, dirichlet, norm, t
from probjax.stats.discrete.empirical import empirical
from probjax.utils.graph import find_ancestors_jax
from probjax.utils.jaxutils import nested_checkpoint_scan

# ---------------------------------------------------------------- core / logdet


@pytest.mark.parametrize("scale", [1e-6, 1e-10, 1e-12])
def test_mul_logdet_is_exact_for_small_scales(scale):
    """The log-det of a scaled identity must not be floored by an epsilon."""
    value, logdet = inverse_and_logabsdet(lambda x: x * scale)(jnp.asarray(1.0))
    np.testing.assert_allclose(float(logdet), -np.log(scale), rtol=1e-6)
    np.testing.assert_allclose(float(value), 1.0 / scale, rtol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_mul_logdet_floor_is_dtype_relative(dtype):
    """A singular factor is floored at the dtype's smallest normal.

    The guard is a numerical-stability floor, so it stays finite, but it must
    scale with the dtype instead of using one absolute constant for all of
    them.
    """
    with jax.enable_x64():
        _, logdet = inverse_and_logabsdet(lambda x: x * jnp.asarray(0.0, dtype))(
            jnp.asarray(1.0, dtype)
        )
        expected = float(np.log(np.finfo(np.dtype(dtype)).tiny))
        assert jnp.isfinite(logdet)
        np.testing.assert_allclose(float(logdet), -expected, rtol=1e-6)


# ---------------------------------------------------------------- graph utility


@pytest.mark.parametrize(
    "edges,node,expected",
    [
        ([(0, 1), (1, 2), (2, 3)], 3, [True, True, True, False]),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], 3, [True, True, True, False]),
        ([(0, 1), (1, 2), (3, 4), (4, 5)], 2, [True, True, False, False, False, False]),
    ],
)
def test_find_ancestors_matches_reachability(edges, node, expected):
    """Ancestors are the nodes with a directed path *to* ``node``."""
    size = len(expected)
    mask = jnp.zeros((size, size), dtype=jnp.bool_)
    for parent, child in edges:
        mask = mask.at[parent, child].set(True)
    np.testing.assert_array_equal(
        np.asarray(find_ancestors_jax(mask, node)), np.asarray(expected)
    )


# -------------------------------------------------------------------- filtering


def test_rank_reduced_smoothing_is_finite_below_full_rank():
    """RTS smoothing must not blow up when the predicted covariance is singular."""
    transition = jnp.diag(jnp.array([0.98, 0.92, 0.85]))
    process_noise = jnp.diag(jnp.array([0.08, 0.01, 1e-4]))
    observation = jnp.array([[1.0, 0.2, 0.0]])
    model = LinearGaussianModel(
        lambda theta: (jnp.zeros(3), jnp.eye(3)),
        lambda theta, t0, t1: (transition, process_noise),
        lambda theta, t: (observation, jnp.array([[0.1]])),
    )
    ts = jnp.arange(1.0, 7.0)
    ys = jnp.full((6, 1), 0.05)
    for rank in (1, 2):
        kernel = rank_reduced_kalman_filter(model, rank)
        initial = kernel.init(jax.random.key(0), None)
        result = run(
            jax.random.key(1),
            kernel,
            initial,
            None,
            len(ts),
            args=(ts, ys),
            collect=("state", "info"),
        )
        means, covs = smooth(
            model, None, initial, result.trace["state"], result.trace["info"]
        )
        assert jnp.all(jnp.isfinite(means)), f"rank {rank} means not finite"
        assert jnp.all(jnp.isfinite(covs)), f"rank {rank} covariances not finite"


def test_full_rank_smoothing_still_matches_the_kalman_filter():
    transition = jnp.diag(jnp.array([0.98, 0.92, 0.85]))
    process_noise = jnp.diag(jnp.array([0.08, 0.01, 1e-4]))
    observation = jnp.array([[1.0, 0.2, 0.0]])
    model = LinearGaussianModel(
        lambda theta: (jnp.zeros(3), jnp.eye(3)),
        lambda theta, t0, t1: (transition, process_noise),
        lambda theta, t: (observation, jnp.array([[0.1]])),
    )
    ts = jnp.arange(1.0, 7.0)
    ys = jnp.full((6, 1), 0.05)

    def smoothed(kernel):
        initial = kernel.init(jax.random.key(0), None)
        result = run(
            jax.random.key(1),
            kernel,
            initial,
            None,
            len(ts),
            args=(ts, ys),
            collect=("state", "info"),
        )
        return smooth(model, None, initial, result.trace["state"], result.trace["info"])

    exact_means, exact_covs = smoothed(kalman_filter(model))
    reduced_means, reduced_covs = smoothed(rank_reduced_kalman_filter(model, 3))
    np.testing.assert_allclose(reduced_means, exact_means, atol=1e-5)
    np.testing.assert_allclose(reduced_covs, exact_covs, atol=1e-5)


# -------------------------------------------------------------------------- SMC


def test_ess_of_all_negative_infinity_is_zero():
    assert float(ess(jnp.array([-jnp.inf, -jnp.inf, -jnp.inf]))) == pytest.approx(0.0)


def test_ess_ignores_a_common_offset():
    log_weights = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    expected = 1.0 / float(jnp.sum(jnp.exp(log_weights) ** 2))
    np.testing.assert_allclose(float(ess(log_weights)), expected, rtol=1e-6)
    # A large negative offset used to underflow every weight to zero. The
    # tolerance reflects the float32 resolution of ``log_weights - offset``
    # itself, not the ESS computation.
    np.testing.assert_allclose(float(ess(log_weights - 50.0)), expected, rtol=1e-4)


def test_adaptive_tempering_reduces_vector_loglikelihood_per_particle():
    """A per-observation likelihood must be summed into one weight per particle."""
    loglik = jax.random.normal(jax.random.key(0), (200, 5))
    temperature = jnp.array(0.0)
    per_observation = _solve_temperature(loglik, temperature, 0.5)
    pre_summed = _solve_temperature(jnp.sum(loglik, axis=1), temperature, 0.5)
    np.testing.assert_allclose(per_observation, pre_summed, rtol=1e-6)
    assert float(per_observation) < 1.0


# ------------------------------------------------------------------------- MCMC


@pytest.mark.parametrize("api", [dynamic_hmc, adjusted_mclmc_dynamic])
def test_dynamic_stepping_accepts_every_sequence_kind(api):
    logdensity = lambda x: -0.5 * jnp.sum(x**2)  # noqa: E731
    next_arg, steps = random_trajectory_length_fns(10)
    for sequence in ("halton", "random", (next_arg, steps)):
        kernel = api(logdensity, integration_steps_sequence=sequence)
        state = kernel.init(jax.random.key(0), jnp.zeros(2))
        params = kernel.init_params(state)
        new_state, _ = kernel.step(jax.random.key(1), state, params)
        assert jnp.all(jnp.isfinite(new_state.position))


# ---------------------------------------------------------------------- solver


@pytest.mark.parametrize("nested_lengths", [(8,), (4, 2), (2, 2, 2)])
def test_nested_checkpoint_scan_accepts_pytree_xs(nested_lengths):
    """``xs`` may be a pytree of leading-axis arrays, as SDEs pass."""
    xs = (jnp.arange(8.0), jnp.arange(8.0) * 2.0)

    def body(carry, step_xs):
        return carry + step_xs[0] + step_xs[1], None

    carry, _ = nested_checkpoint_scan(
        body, jnp.asarray(0.0), xs, nested_lengths=nested_lengths
    )
    np.testing.assert_allclose(float(carry), float(jnp.sum(xs[0] + xs[1])))


# ---------------------------------------------------------------------- stats


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("loc,scale", [(0.0, 1.0), (2.0, 1.0), (-1.0, 0.5)])
def test_norm_moment_matches_scipy(n, loc, scale):
    np.testing.assert_allclose(
        float(norm.moment(n, loc, scale)),
        float(sp.norm.moment(n, loc, scale)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("alpha,beta_value", [(2.0, 3.0), (0.5, 0.5), (5.0, 1.0)])
def test_beta_entropy_matches_scipy(alpha, beta_value):
    np.testing.assert_allclose(
        float(beta.entropy(alpha, beta_value)),
        float(sp.beta.entropy(alpha, beta_value)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("df", [1.0, 2.0, 5.0, 30.0])
def test_t_entropy_matches_scipy(df):
    np.testing.assert_allclose(
        float(t.entropy(df)), float(sp.t.entropy(df)), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("alpha", [[2.0, 3.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
def test_dirichlet_entropy_matches_scipy(alpha):
    np.testing.assert_allclose(
        float(dirichlet.entropy(jnp.array(alpha))),
        float(sp.dirichlet.entropy(alpha)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_categorical_entropy_is_zero_at_a_certain_outcome():
    assert float(categorical.entropy(jnp.array([1.0, 0.0]))) == pytest.approx(0.0)
    np.testing.assert_allclose(
        float(categorical.entropy(jnp.array([0.5, 0.5]))), np.log(2.0), rtol=1e-6
    )


def test_empirical_sampler_uses_the_weights_as_probabilities():
    """``random.categorical`` takes logits; passing probabilities skews draws."""
    values = jnp.arange(4.0)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    samples = empirical._rvs_impl(jax.random.key(0), values, weights, shape=(60000,))
    frequencies = np.bincount(np.asarray(samples).astype(int), minlength=4) / 60000
    np.testing.assert_allclose(frequencies, np.asarray(weights), atol=0.015)
