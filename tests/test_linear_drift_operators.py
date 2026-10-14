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


def test_linear_drift_dense_matrix_state_uses_matmul():
    # Calling the drift acts by matmul, broadcasting over leading axes —
    # the pre-operator behavior generic solvers rely on.
    Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    drift = linear_drift(A=jnp.asarray(A_DENSE))
    np.testing.assert_allclose(drift(0.0, jnp.asarray(Y)), A_DENSE @ Y, rtol=1e-6)
    drift_biased = linear_drift(
        A=jnp.asarray(A_DENSE), b=lambda t: jnp.asarray(B)[:, None]
    )
    np.testing.assert_allclose(
        drift_biased(0.0, jnp.asarray(Y)), A_DENSE @ Y + B[:, None], rtol=1e-6
    )


def test_scalar_drift_acts_elementwise():
    drift = linear_drift(A=jnp.asarray(-0.5))
    np.testing.assert_allclose(drift(0.0, jnp.asarray(X0)), -0.5 * X0, rtol=1e-6)
    result = odeint(drift, jnp.asarray(X0), jnp.asarray(TS), method="linear_exact")
    np.testing.assert_allclose(result, X0 * np.exp(-0.5 * TS)[:, None], rtol=1e-5)


def test_as_operator_rejects_non_square_systems():
    from probjax.utils.linear_operator import as_operator

    # Dense drifts need vector states of matching dimension.
    with pytest.raises(ValueError, match="vector states"):
        as_operator(np.asarray(A_DENSE), (2, 2))
    with pytest.raises(ValueError, match="square"):
        as_operator(np.ones((2, 3)), (2,))
    with pytest.raises(ValueError, match="0-d, 1-d or 2-d"):
        as_operator(np.ones((2, 2, 2)), (2,))
    with pytest.raises(ValueError, match="broadcast"):
        as_operator(np.ones(3), (2,))
    with pytest.raises(ValueError, match="cannot act"):
        as_operator(Dense(jnp.asarray(A_DENSE)), (3,))


def test_linear_exact_sde_noise_and_dwt_use_independent_keys():
    # The sampling noise and the reported dWt must not share draws.
    with jax.enable_x64():
        drift = linear_drift(A=jnp.asarray([0.0, 0.0]))
        diffusion = const_diffusion(G=jnp.array([1.0, 1.0]))
        state = LinearExactSDEState(t0=jnp.asarray(0.0), y0=jnp.asarray(X0))
        dt = 0.4
        step = build_linear_exact_sde_step(drift, diffusion)
        new_state, info = step(jax.random.key(0), state, dt)
        # Diagonal path with A=0: y - y_mean = sqrt(dt) * noise.
        reconstructed = (np.asarray(new_state.y0) - X0) / np.sqrt(dt)
        assert not np.allclose(np.asarray(info.dWt), reconstructed, atol=1e-12)
        # Same key reproduces the same step.
        again_state, again_info = step(jax.random.key(0), state, dt)
        np.testing.assert_allclose(
            np.asarray(again_state.y0), np.asarray(new_state.y0), rtol=0
        )
        np.testing.assert_allclose(
            np.asarray(again_info.dWt), np.asarray(info.dWt), rtol=0
        )


def test_linear_exact_preserves_float64():
    with jax.enable_x64():
        drift = linear_drift(A=jnp.asarray(A_DENSE, dtype=jnp.float64))
        result = odeint(
            drift,
            jnp.asarray(X0, dtype=jnp.float64),
            jnp.asarray(TS, dtype=jnp.float64),
            method="linear_exact",
            dtype=jnp.float64,
        )
        assert result.dtype == jnp.float64
        np.testing.assert_allclose(
            result, _affine_reference(A_DENSE, np.zeros(2), X0, TS), rtol=1e-10
        )


def test_linear_exact_sde_matches_moments_with_full_noise():
    # Empirical mean/covariance over keys match the exact flow and Van Loan Q.
    from probjax.utils.odeutil.solvers.linear_exact import affine_flow

    # Hermetic against executables compiled by earlier tests: on JAX 0.9.x a
    # jitted float32 scalar-times-matrix (e.g. from a time-dependent drift
    # above) can poison the cache for the float64 vmapped solve below
    # (fixed upstream by 0.11; CI runs latest).
    jax.clear_caches()

    with jax.enable_x64():
        dt = 0.3
        G = np.array([[0.5, 0.1], [0.0, 0.4]])
        drift = linear_drift(A=jnp.asarray(A_DENSE))
        diffusion = const_diffusion(G=jnp.asarray(G))
        state = LinearExactSDEState(t0=jnp.asarray(0.0), y0=jnp.asarray(X0))
        step = build_linear_exact_sde_step(drift, diffusion)
        keys = jax.random.split(jax.random.key(1), 2000)
        samples = jax.vmap(lambda k: step(k, state, dt)[0].y0)(keys)
        samples = np.asarray(samples)
        mean = samples.mean(axis=0)
        cov = np.cov(samples, rowvar=False)
        expected_mean = np.asarray(
            affine_flow(Dense(jnp.asarray(A_DENSE)), None, jnp.asarray(X0), dt)
        )
        expected_cov = _process_noise_cov(
            Dense(jnp.asarray(A_DENSE)), jnp.asarray(G), dt, 2
        )
        np.testing.assert_allclose(mean, expected_mean, rtol=0.1, atol=0.05)
        np.testing.assert_allclose(cov, np.asarray(expected_cov), rtol=0.2, atol=0.05)


def test_diagonal_phi_higher_order_matches_scalar_formula():
    # phi_2(z) = (exp(z) - 1 - z) / z^2.
    with jax.enable_x64():
        z = np.array([-0.5, 0.7])
        (_, _, phi2) = Diagonal(jnp.asarray(z)).phi(1.0, k=2)
        expected = (np.exp(z) - 1.0 - z) / z**2
        np.testing.assert_allclose(np.asarray(phi2.diag), expected, rtol=1e-10)
