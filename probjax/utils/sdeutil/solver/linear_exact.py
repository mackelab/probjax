from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from probjax.utils.functions import const_diffusion, linear_drift
from probjax.utils.linalg import is_diagonal_matrix, matrix_fraction_decomposition
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey


class LinearExactSDEState(SDEState):
    t0: Array
    y0: Array


class LinearExactSDEInfo(SDEInfo):
    dWt: Array


def _phi1_scalar(z: Array) -> Array:
    small = jnp.abs(z) < 1e-4
    series = 1.0 + 0.5 * z + (z * z) / 6.0 + (z * z * z) / 24.0
    return jnp.where(small, series, jnp.expm1(z) / z)


def init_state(
    t0: ArrayLike,
    y0: ArrayLike,
    *args,
    drift: Optional[Callable] = None,
    diffusion: Optional[Callable] = None,
    **kwargs,
) -> LinearExactSDEState:
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact_sde requires `drift` to be a `linear_drift`")
    if not isinstance(diffusion, const_diffusion):
        raise TypeError(
            "linear_exact_sde requires `diffusion` to be a `const_diffusion`"
        )

    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)

    return LinearExactSDEState(t0=t0, y0=y0)


def build_linear_exact_sde_step(
    drift: linear_drift,
    diffusion: const_diffusion,
    noise_dim: int | None = None,
    **kwargs,
):
    del kwargs
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact_sde requires `drift` to be a `linear_drift`")
    if not isinstance(diffusion, const_diffusion):
        raise TypeError(
            "linear_exact_sde requires `diffusion` to be a `const_diffusion`"
        )

    G = diffusion.G
    G_shape = G.shape
    if G.ndim == 1:
        state_dim = G.shape[0]
        noise_dim_actual = state_dim
    else:
        state_dim = G.shape[0]
        noise_dim_actual = G.shape[1] if noise_dim is None else noise_dim

    is_const = drift.is_constant
    if is_const:
        A = drift.A if not callable(drift.A) else drift.A(jnp.asarray(0.0))
        # Pytree-flowing drifts arrive with ``A`` as a traced array, so we
        # branch on its static shape info (ndim) rather than on values.
        A_sym = jnp.asarray(A)
        is_scalar = A_sym.ndim == 0 or (
            A_sym.ndim == 1 and A_sym.shape[0] == 1
        )
    else:
        A = None
        is_scalar = False

    def step_fn(
        rng: RngKey, state: LinearExactSDEState, dt: ArrayLike
    ) -> tuple[LinearExactSDEState, LinearExactSDEInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        if is_const:
            A_t = A
        else:
            A_t = drift.A(t0)

        # Compute exp(A * dt) - handle scalar case
        if is_scalar:
            scalar_A = jnp.asarray(A_t).reshape(())
            expm_A = jnp.exp(scalar_A * dt)
        else:
            expm_A = jax.scipy.linalg.expm(A_t * dt)

        if drift.b is None:
            if is_scalar:
                y_mean = expm_A * y0
            else:
                y_mean = expm_A @ y0
        else:
            b = drift.b(t0 + dt)
            phi1 = _phi1_scalar(A_t * dt)
            if is_scalar:
                y_mean = expm_A * y0 + dt * phi1 * b
            else:
                y_mean = expm_A @ y0 + dt * (phi1 @ b)

        d = state_dim

        if G.ndim == 1:
            GGt_scalar = G**2
            Q_diag = jnp.abs(dt) * GGt_scalar
            y_sample = y_mean + jax.random.normal(rng, (d,)) * jnp.sqrt(Q_diag)
        else:
            GGt = G @ G.T
            if is_scalar:
                # For scalar A with 2D G, use simplified covariance formula
                # Q = integral_0^dt exp(2*A*s) ds * GGt
                # For scalar A: Q = (exp(2*A*dt) - 1) / (2*A) * GGt
                scalar_A = jnp.asarray(A_t).reshape(())
                small_A = jnp.abs(scalar_A) < 1e-6
                # When A is near zero: Q ≈ dt * GGt
                Q_small = dt * GGt
                # When A is not small: Q = (exp(2*A*dt) - 1) / (2*A) * GGt
                Q_large = (jnp.exp(2 * scalar_A * dt) - 1.0) / (2 * scalar_A) * GGt
                new_cov = jnp.where(small_A, Q_small, Q_large)

                L = jnp.linalg.cholesky(new_cov + 1e-8 * jnp.eye(d))
                y_sample = y_mean + L @ jax.random.normal(rng, (d,))
            else:
                block = jnp.block([[A_t, GGt], [jnp.zeros((d, d)), -A_t.T]])
                M = jax.scipy.linalg.expm(block * dt)
                Phi = M[:d, :d]
                Q_block = M[:d, d:] @ Phi.T
                new_cov = jnp.abs(dt) * Q_block

                L = jnp.linalg.cholesky(new_cov + 1e-8 * jnp.eye(d))
                y_sample = y_mean + L @ jax.random.normal(rng, (d,))

        return (
            LinearExactSDEState(t0=t0 + dt, y0=y_sample),
            LinearExactSDEInfo(dWt=jax.random.normal(rng, (noise_dim_actual,))),
        )

    return step_fn


class linear_exact_sde(SDESolverAPI):
    init = staticmethod(init_state)
    build_step = staticmethod(build_linear_exact_sde_step)


register_method(
    "linear_exact_sde",
    linear_exact_sde,
    info={
        "order": 1,
        "strong_order": 1.0,
        "weak_order": 1.0,
        "adaptive": False,
        "info": "Exact solver for linear SDEs dy/dt = A y + b, dX = G dW (constant A, G)",
    },
)
