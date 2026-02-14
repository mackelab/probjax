from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from probjax.utils.functions import linear_drift
from probjax.utils.linalg import is_diagonal_matrix
from probjax.utils.odeutil.solvers.base import (
    ODEInfo,
    ODESolverAPI,
    ODEState,
    register_method,
)
from probjax.utils.typing import Array, ArrayLike, Callable


class LinearExactState(ODEState):
    t0: Array
    y0: Array


class LinearExactInfo(ODEInfo):
    pass


def _phi1_scalar(z: Array) -> Array:
    """Compute phi_1(z) = (exp(z) - 1) / z with numerical stability."""
    small = jnp.abs(z) < 1e-4
    series = 1.0 + 0.5 * z + (z * z) / 6.0 + (z * z * z) / 24.0
    return jnp.where(small, series, jnp.expm1(z) / z)


def init_linear_exact(
    t0: ArrayLike,
    y0: ArrayLike,
    *args,
    drift: Optional[Callable] = None,
    **kwargs,
) -> LinearExactState:
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact requires `drift` to be a `linear_drift`")

    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)

    return LinearExactState(t0=t0, y0=y0)


def build_linear_exact_step(drift: linear_drift) -> Callable:
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact requires a `linear_drift`")

    is_const = drift.is_constant

    if is_const:
        A = drift.A if not callable(drift.A) else drift.A(jnp.asarray(0.0))
        # Check matrix structure statically using concrete array
        # Convert to numpy to avoid tracing
        A_concrete = np.asarray(A)
        is_scalar = A_concrete.ndim == 0 or (
            A_concrete.ndim == 1 and A_concrete.shape[0] == 1
        )
        # Check diagonal using numpy to avoid JAX tracing
        if not is_scalar and A_concrete.ndim == 2:
            is_diag = np.all(np.diag(np.diagonal(A_concrete)) == A_concrete)
        else:
            is_diag = False

        # Choose computation method based on structure
        if is_scalar:

            def compute_expm(dt):
                scalar_A = jnp.asarray(A).reshape(())
                return jnp.exp(scalar_A * dt)

        elif is_diag:

            def compute_expm(dt):
                diag_A = jnp.diag(A)
                return jnp.exp(diag_A * dt)

        else:

            def compute_expm(dt):
                return jax.scipy.linalg.expm(A * dt)

        def step_fn(
            state: LinearExactState, dt: ArrayLike, *args
        ) -> tuple[LinearExactState, LinearExactInfo]:
            t0, y0 = state.t0, state.y0
            dt = jnp.asarray(dt)

            expm_A = compute_expm(dt)

            if drift.b is None:
                # Handle scalar vs matrix multiplication
                if is_scalar:
                    y1 = expm_A * y0
                elif is_diag:
                    y1 = expm_A * y0
                else:
                    y1 = expm_A @ y0
            else:
                b = drift.b(t0 + dt, *args)
                phi1 = _phi1_scalar(A * dt)
                if is_scalar:
                    y1 = expm_A * y0 + dt * phi1 * b
                elif is_diag:
                    y1 = expm_A * y0 + dt * jnp.diag(phi1) @ b
                else:
                    y1 = expm_A @ y0 + dt * (phi1 @ b)

            return (
                LinearExactState(t0=t0 + dt, y0=y1),
                LinearExactInfo(),
            )

        return step_fn

    # Non-constant A case - need to handle dynamic shapes
    # For simplicity, assume matrix case and let JAX handle it
    def step_fn(
        state: LinearExactState, dt: ArrayLike, *args
    ) -> tuple[LinearExactState, LinearExactInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        A_t = drift.A(t0)
        # For time-varying A, use general matrix exponential
        expm_A = jax.scipy.linalg.expm(A_t * dt)

        if drift.b is None:
            y1 = expm_A @ y0
        else:
            b = drift.b(t0 + dt, *args)
            # Use general matrix case
            phi1_mat = jax.scipy.linalg.expm(A_t * dt) - jnp.eye(A_t.shape[0])
            y1 = expm_A @ y0 + phi1_mat @ b

        return (
            LinearExactState(t0=t0 + dt, y0=y1),
            LinearExactInfo(),
        )

    return step_fn

    # Non-constant A case - need to handle dynamic shapes
    # For simplicity, assume matrix case and let JAX handle it
    def step_fn(
        state: LinearExactState, dt: ArrayLike, *args
    ) -> tuple[LinearExactState, LinearExactInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        A_t = drift.A(t0)
        # For time-varying A, use general matrix exponential
        expm_A = jax.scipy.linalg.expm(A_t * dt)

        if drift.b is None:
            y1 = expm_A @ y0
        else:
            b = drift.b(t0 + dt, *args)
            # Use general matrix case
            phi1_mat = jax.scipy.linalg.expm(A_t * dt) - jnp.eye(A_t.shape[0])
            y1 = expm_A @ y0 + phi1_mat @ b

        return (
            LinearExactState(t0=t0 + dt, y0=y1),
            LinearExactInfo(),
        )

    return step_fn

    # Non-constant A case - need to handle dynamic shapes
    # For simplicity, assume matrix case and let JAX handle it
    def step_fn(
        state: LinearExactState, dt: ArrayLike, *args
    ) -> tuple[LinearExactState, LinearExactInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        A_t = drift.A(t0)
        # For time-varying A, use general matrix exponential
        expm_A = jax.scipy.linalg.expm(A_t * dt)

        if drift.b is None:
            y1 = expm_A @ y0
        else:
            b = drift.b(t0 + dt, *args)
            # Use general matrix case
            phi1_mat = jax.scipy.linalg.expm(A_t * dt) - jnp.eye(A_t.shape[0])
            y1 = expm_A @ y0 + phi1_mat @ b

        return (
            LinearExactState(t0=t0 + dt, y0=y1),
            LinearExactInfo(),
        )

    return step_fn

    # Non-constant A case
    def step_fn(
        state: LinearExactState, dt: ArrayLike, *args
    ) -> tuple[LinearExactState, LinearExactInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)

        A_t = drift.A(t0)
        # For time-varying A, we can't check is_diagonal statically
        # Use a simpler approach that works with tracers
        expm_A = _compute_expm_A(A_t, dt, is_diag=False)

        if drift.b is None:
            # Use element-wise multiplication for scalar/1D, matmul for 2D
            y1 = jax.lax.cond(A_t.ndim <= 1, lambda: expm_A * y0, lambda: expm_A @ y0)
        else:
            b = drift.b(t0 + dt, *args)
            phi1 = _phi1_scalar(A_t * dt)

            def scalar_case():
                return expm_A * y0 + dt * phi1 * b

            def matrix_case():
                return expm_A @ y0 + dt * (phi1 @ b)

            y1 = jax.lax.cond(A_t.ndim <= 1, scalar_case, matrix_case)

        return (
            LinearExactState(t0=t0 + dt, y0=y1, A=A_t, expm_A=expm_A),
            LinearExactInfo(),
        )

    return step_fn


class linear_exact(ODESolverAPI):
    init = staticmethod(init_linear_exact)
    build_step = staticmethod(build_linear_exact_step)


register_method(
    "linear_exact",
    linear_exact,
    info={
        "explicit": False,
        "order": 1,
        "info": "Exact solver for linear ODEs dy/dt = A y + b (constant or time-varying A)",
        "adaptive": False,
    },
)
