from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from probjax.utils.functions import const_diffusion, linear_drift
from probjax.utils.odeutil.solvers.exponential import compute_phi_functions
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method
from probjax.utils.special.phi1 import phi1_scalar
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey


class LinearExactSDEState(SDEState):
    t0: Array
    y0: Array


class LinearExactSDEInfo(SDEInfo):
    dWt: Array


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
    if G.ndim == 1:
        # Diagonal noise: use diag(G) so the covariance below still includes A.
        G = jnp.diag(G)
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
        is_scalar = A_sym.ndim == 0 or (A_sym.ndim == 1 and A_sym.shape[0] == 1)
    else:
        A = None
        is_scalar = False

    def step_fn(
        rng: RngKey, state: LinearExactSDEState, dt: ArrayLike
    ) -> tuple[LinearExactSDEState, LinearExactSDEInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        A_t = A if is_const else drift.A(t0)

        # Compute exp(A * dt) - handle scalar case
        if is_scalar:
            scalar_A = jnp.asarray(A_t).reshape(())
            expm_A = jnp.exp(scalar_A * dt)
        else:
            expm_A = jax.scipy.linalg.expm(A_t * dt)

        if drift.b is None:
            y_mean = expm_A * y0 if is_scalar else expm_A @ y0
        else:
            b = drift.b(t0 + dt)
            if is_scalar:
                y_mean = expm_A * y0 + dt * phi1_scalar(A_t * dt) * b
            else:
                phi1 = compute_phi_functions(A_t, dt, k=1)[1]
                y_mean = expm_A @ y0 + dt * (phi1 @ b)

        d = state_dim

        GGt = G @ G.T
        if is_scalar:
            # For scalar A, Q = integral_0^dt exp(2*A*s) ds * GGt
            #                 = (exp(2*A*dt) - 1) / (2*A) * GGt
            scalar_A = jnp.asarray(A_t).reshape(())
            small_A = jnp.abs(scalar_A) < 1e-6
            # When A is near zero: Q ≈ dt * GGt
            Q_small = dt * GGt
            Q_large = (jnp.exp(2 * scalar_A * dt) - 1.0) / (2 * scalar_A) * GGt
            new_cov = jnp.where(small_A, Q_small, Q_large)
        else:
            # Van Loan: the top-right block of expm(dt * [[A, GGt], [0, -A^T]])
            # times exp(A^T dt) is Q = integral_0^dt exp(A s) GGt exp(A^T s) ds.
            block = jnp.block([[A_t, GGt], [jnp.zeros((d, d)), -A_t.T]])
            M = jax.scipy.linalg.expm(block * dt)
            Phi = M[:d, :d]
            new_cov = M[:d, d:] @ Phi.T

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
