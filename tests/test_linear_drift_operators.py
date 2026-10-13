"""Linear drifts given as structured linear operators in odeint/sdeint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from probjax.utils.functions import const_diffusion, linear_drift
from probjax.utils.linear_operator import Dense, Diagonal, Function
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.sdeutil.solver.linear_exact import (
    LinearExactSDEState,
    _process_noise_cov,
    build_linear_exact_sde_step,
)

A_DENSE = np.array([[-1.0, 2.0], [0.0, -3.0]])
A_DIAG = np.array([-1.0, 0.5])
B = np.array([1.0, -0.5])
X0 = np.array([1.0, -1.0])
TS = np.linspace(0.0, 1.0, 6)


def _affine_reference(a, b, x0, ts):
    """Exact solution of y' = a y + b via expm of the augmented system."""
    aug = np.zeros((3, 3))
    aug[:2, :2], aug[:2, 2] = a, b
    return np.stack([(expm(t * aug) @ np.append(x0, 1.0))[:2] for t in ts])


@pytest.mark.parametrize(
    "A, matrix",
    [
        (Dense(jnp.asarray(A_DENSE)), A_DENSE),
        (Diagonal(jnp.asarray(A_DIAG)), np.diag(A_DIAG)),
        (jnp.asarray(A_DIAG), np.diag(A_DIAG)),  # 1-d arrays are diagonals
        (lambda t: Dense(jnp.asarray(A_DENSE)), A_DENSE),  # time-dependent form
    ],
)
def test_linear_exact_ode_with_operators(A, matrix):
    drift = linear_drift(A=A, b=lambda t: jnp.asarray(B))
    result = odeint(drift, jnp.asarray(X0), jnp.asarray(TS), method="linear_exact")
    np.testing.assert_allclose(
        result, _affine_reference(matrix, B, X0, TS), rtol=1e-5, atol=1e-6
    )


def test_linear_drift_evaluates_operators():
    drift = linear_drift(A=Dense(jnp.asarray(A_DENSE)), b=lambda t: jnp.asarray(B))
    assert drift.is_constant
    np.testing.assert_allclose(drift(0.0, jnp.asarray(X0)), A_DENSE @ X0 + B, rtol=1e-6)
    # Generic solvers only need the drift's value.
    rk4 = odeint(linear_drift(A=Diagonal(jnp.asarray(A_DIAG))), jnp.asarray(X0), TS)
    np.testing.assert_allclose(rk4[-1], np.exp(A_DIAG) * X0, rtol=1e-4)


def test_linear_exact_matrix_free_operator_explains_densify():
    drift = linear_drift(A=Function(lambda x: -x, 2))
    with pytest.raises(NotImplementedError, match="as_dense"):
        odeint(drift, jnp.asarray(X0), jnp.asarray(TS), method="linear_exact")


def test_linear_exact_gradient_through_operator():
    def final_state(diag):
        drift = linear_drift(A=Diagonal(diag))
        return odeint(drift, jnp.asarray(X0), jnp.asarray(TS), method="linear_exact")[
            -1
        ]

    diag = jnp.asarray(A_DIAG)
    grad = jax.jit(jax.jacfwd(final_state))(diag)
    np.testing.assert_allclose(grad, np.diag(X0 * np.exp(A_DIAG)), rtol=1e-5)


def _van_loan(a, ggt, dt):
    n = a.shape[0]
    block = np.block([[a, ggt], [np.zeros((n, n)), -a.T]])
    m = expm(block * dt)
    return m[:n, n:] @ m[:n, :n].T


@pytest.mark.parametrize(
    "G", [np.array([0.3, 0.7]), np.array([[0.3, 0.1], [0.0, 0.7]])]
)
def test_diagonal_process_noise_matches_van_loan(G):
    with jax.enable_x64():
        G_full = np.diag(G) if G.ndim == 1 else G
        reference = _van_loan(np.diag(A_DIAG), G_full @ G_full.T, 0.4)
        for A in (Diagonal(jnp.asarray(A_DIAG)), Dense(jnp.diag(jnp.asarray(A_DIAG)))):
            np.testing.assert_allclose(
                _process_noise_cov(A, jnp.asarray(G), 0.4, 2), reference, rtol=1e-10
            )


def test_linear_exact_sde_diagonal_drift_matches_dense():
    # Vector A used to crash (expm of a vector); it now takes the closed form,
    # which must give the same sample as the Van Loan path for the same noise.
    with jax.enable_x64():
        diffusion = const_diffusion(G=jnp.array([0.3, 0.7]))
        state = LinearExactSDEState(t0=jnp.asarray(0.0), y0=jnp.asarray(X0))
        key = jax.random.key(3)
        samples = [
            build_linear_exact_sde_step(linear_drift(A=A), diffusion)(key, state, 0.4)[
                0
            ].y0
            for A in (jnp.asarray(A_DIAG), jnp.diag(jnp.asarray(A_DIAG)))
        ]
        np.testing.assert_allclose(samples[0], samples[1], rtol=1e-6)


def test_linear_exact_sde_with_operator_drift():
    drift = linear_drift(A=Dense(jnp.asarray(A_DENSE)))
    diffusion = const_diffusion(G=jnp.array([0.0, 0.0]))
    result = sdeint(
        jax.random.key(0),
        drift,
        diffusion,
        jnp.asarray(X0),
        jnp.asarray(TS),
        method="linear_exact_sde",
    )
    trace = result[0] if isinstance(result, tuple) else result
    # Zero noise: the SDE reduces to the exact linear ODE flow, up to the
    # ~1e-4 per-step noise from the solver's 1e-8 Cholesky jitter.
    np.testing.assert_allclose(
        trace, _affine_reference(A_DENSE, np.zeros(2), X0, TS), atol=1e-3
    )
