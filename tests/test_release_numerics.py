"""Analytic regressions exposed while writing the release's solver examples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils import odeint


def test_rk4_one_step_matches_classical_stability_polynomial():
    # A single step exposes incorrect stage weights without a fine grid masking it.
    rate = 0.7
    expected = 1 - rate + rate**2 / 2 - rate**3 / 6 + rate**4 / 24
    result = odeint(
        lambda t, y: -rate * y,
        jnp.array([1.0, 2.0]),
        jnp.array([0.0, 1.0]),
        method="rk4",
    )
    np.testing.assert_allclose(result[-1], np.array([1.0, 2.0]) * expected, rtol=1e-6)


@pytest.mark.parametrize("collect_trace", [False, True])
def test_rk4_parameter_gradient_matches_analytic_decay(collect_trace):
    initial = jnp.array([1.0, 2.0])
    times = jnp.linspace(0.0, 1.0, 17)

    def objective(rate):
        result = odeint(
            lambda t, y, r: -r * y,
            initial, times, rate,
            method="rk4", collect_trace=collect_trace,
        )
        return jnp.sum(result[-1] if collect_trace else result)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(0.5))
    expected = 3 * np.exp(-0.5)
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(gradient, -expected, rtol=1e-6)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Release blocker: adaptive custom VJP retains a non-JAX controller residual",
)
def test_adaptive_decay_gradients_match_analytic_solution():
    """Exercise the promised gradient, not an assertion that freezes the error."""
    times = jnp.linspace(0.0, 1.0, 17)

    def objective(initial, rate):
        return odeint(
            lambda t, y, r: -r * y, initial, times, rate, method="dopri5"
        )[-1].sum()

    initial = jnp.array([1.0, 2.0])
    initial_grad, rate_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        initial, jnp.asarray(0.5)
    )
    np.testing.assert_allclose(initial_grad, np.full(2, np.exp(-0.5)), rtol=1e-3)
    np.testing.assert_allclose(rate_grad, -3 * np.exp(-0.5), rtol=1e-3)
