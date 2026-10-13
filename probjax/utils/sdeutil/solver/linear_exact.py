from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from probjax.utils.functions import const_diffusion, linear_drift
from probjax.utils.linear_operator import Diagonal
from probjax.utils.odeutil.solvers.linear_exact import affine_flow
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

    G = jnp.asarray(diffusion.G)
    diagonal_noise = G.ndim <= 1
    if not diagonal_noise and noise_dim is None:
        noise_dim = G.shape[1]

    def step_fn(
        rng: RngKey, state: LinearExactSDEState, dt: ArrayLike
    ) -> tuple[LinearExactSDEState, LinearExactSDEInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)
        A = drift.operator(t0, y0.shape)
        b = None if drift.b is None else drift.b(t0 + dt)
        y_mean = affine_flow(A, b, y0, dt)

        noise = jax.random.normal(rng, y0.shape)
        if isinstance(A, Diagonal) and diagonal_noise:
            # Independent coordinates: Q_ii = g_i^2 * dt * phi_1(2 a_i dt).
            a = jnp.broadcast_to(A.diag, y0.shape)
            variance = jnp.broadcast_to(G, y0.shape) ** 2 * dt * phi1_scalar(2 * a * dt)
            y_sample = y_mean + jnp.sqrt(jnp.maximum(variance, 0)) * noise
        else:
            cov = _process_noise_cov(A, G, dt, y0.size)
            L = jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(y0.size))
            y_sample = y_mean + (L @ noise.ravel()).reshape(y0.shape)

        dWt = jax.random.normal(rng, (y0.size if diagonal_noise else noise_dim,))
        return (
            LinearExactSDEState(t0=t0 + dt, y0=y_sample),
            LinearExactSDEInfo(dWt=dWt),
        )

    return step_fn


def _process_noise_cov(A, G: Array, dt: Array, size: int) -> Array:
    """``Q = integral_0^dt exp(A s) G G^T exp(A^T s) ds``."""
    if G.ndim <= 1:
        G = jnp.diag(jnp.broadcast_to(G, (size,)))
    GGt = G @ G.T
    if isinstance(A, Diagonal):
        a = jnp.broadcast_to(A.diag, (size,))
        return GGt * dt * phi1_scalar((a[:, None] + a[None, :]) * dt)
    # Van Loan: expm(dt * [[A, GGt], [0, -A^T]]) has F12 in its top-right
    # block and Q = F12 @ exp(A dt)^T.
    A = A.as_array()
    block = jnp.block([[A, GGt], [jnp.zeros_like(A), -A.T]])
    M = jax.scipy.linalg.expm(block * dt)
    return M[:size, size:] @ M[:size, :size].T


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
