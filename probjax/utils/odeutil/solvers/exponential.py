from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import jax.scipy.linalg

from probjax.utils.odeutil.solvers.base import (
    ODEInfo,
    ODESolverAPI,
    ODEState,
    register_method,
)


# Exponential integrator state and info classes
class ExpODEState(ODEState):
    t0: Array
    y0: Array
    f0: Optional[Array]


class ExpODEInfo(ODEInfo):
    phi_products: Optional[Array] = None


def init_exp(t0: ArrayLike, y0: Array, *args, drift=None) -> ODEState:
    """Initialize state for exponential integrators."""
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    f0 = drift(t0, y0, *args) if drift is not None else None
    return ExpODEState(t0=t0, y0=y0, f0=f0)


# Utility functions for exponential integrators
def compute_phi_functions(A, dt, k=1):
    """Compute phi functions for exponential integrators.

    Phi_k(z) = ∫_0^1 e^(z*(1-s)) * s^(k-1)/(k-1)! ds

    Returns phi_0, phi_1, ..., phi_k for a given matrix A and timestep dt.
    """
    n = A.shape[0]

    # Create augmented matrix for computing phi functions
    aug_matrix = jnp.zeros((n * (k + 1), n * (k + 1)))

    # Fill the block diagonal with A
    for i in range(k + 1):
        aug_matrix = aug_matrix.at[i * n : (i + 1) * n, i * n : (i + 1) * n].set(A)

    # Fill the superdiagonal blocks with identity matrices
    for i in range(k):
        aug_matrix = aug_matrix.at[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n].set(
            jnp.eye(n)
        )

    # Compute the matrix exponential
    exp_aug = jax.scipy.linalg.expm(aug_matrix * dt)

    # Extract the phi functions
    phi = []
    for i in range(k + 1):
        phi.append(exp_aug[0:n, i * n : (i + 1) * n])

    return phi


# Exponential Euler method
def build_exp_euler_step(drift: Callable, dtype: jnp.dtype = jnp.float32):
    """Build the exponential Euler step function."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> Tuple[ExpODEState, ExpODEInfo]:
        t0 = state.t0
        y0 = state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        # Compute Jacobian at current point
        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)

        # Compute phi functions
        phi = compute_phi_functions(A, dt, k=1)
        phi0, phi1 = phi

        # Compute next state using exponential Euler formula
        y1 = phi0 @ y0 + dt * phi1 @ (f0 - A @ y0)
        f1 = drift(t0 + dt, y1, *args)

        return ExpODEState(t0=t0 + dt, y0=y1, f0=f1), ExpODEInfo(
            phi_products=jnp.array(phi)
        )

    return step_fn


# Exponential Midpoint method
def build_exp_midpoint_step(drift: Callable, dtype: jnp.dtype = jnp.float32):
    """Build the exponential midpoint step function."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> Tuple[ExpODEState, ExpODEInfo]:
        t0 = state.t0
        y0 = state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        # Compute Jacobian at current point
        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)

        # Compute phi functions
        phi = compute_phi_functions(A, dt / 2, k=1)
        phi0_half, phi1_half = phi

        # Compute midpoint state
        y_mid = phi0_half @ y0 + (dt / 2) * phi1_half @ (f0 - A @ y0)
        f_mid = drift(t0 + dt / 2, y_mid, *args)

        # Compute phi functions for full step
        phi = compute_phi_functions(A, dt, k=1)
        phi0, phi1 = phi

        # Compute final state using midpoint derivative
        y1 = phi0 @ y0 + dt * phi1 @ (f_mid - A @ y0)
        f1 = drift(t0 + dt, y1, *args)

        return ExpODEState(t0=t0 + dt, y0=y1, f0=f1), ExpODEInfo(
            phi_products=jnp.array(phi)
        )

    return step_fn


# Exponential RK4 method
def build_exp_rk4_step(drift: Callable, dtype: jnp.dtype = jnp.float32):
    """Build the exponential RK4 step function."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> Tuple[ExpODEState, ExpODEInfo]:
        t0 = state.t0
        y0 = state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        # Compute Jacobian at current point
        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)

        # Compute phi functions
        phi = compute_phi_functions(A, dt, k=3)
        phi0, phi1, phi2, phi3 = phi

        # Define the nonlinear part of the ODE
        def nonlinear(t, y):
            return drift(t, y, *args) - A @ y

        # RK4 stages for the nonlinear part
        r0 = nonlinear(t0, y0)

        k1 = r0
        y1 = y0 + dt * (phi1 @ k1)
        r1 = nonlinear(t0 + dt / 2, y1)

        k2 = r1 - k1
        y2 = y1 + dt * (phi1 @ k2 - phi2 @ k1)
        r2 = nonlinear(t0 + dt / 2, y2)

        k3 = r2 - 2 * k2 - k1
        y3 = y2 + dt * (2 * phi1 @ k3 - 4 * phi2 @ k2 + phi3 @ k1)
        r3 = nonlinear(t0 + dt, y3)

        k4 = r3 - k3 - k2 - k1

        # Final step
        y_next = phi0 @ y0 + dt * (
            phi1 @ (k1 + 2 * k2 + k3) + dt * phi2 @ (k1 + k2) + dt**2 * phi3 @ k1
        )
        f_next = drift(t0 + dt, y_next, *args)

        return ExpODEState(t0=t0 + dt, y0=y_next, f0=f_next), ExpODEInfo(
            phi_products=jnp.array(phi)
        )

    return step_fn


# Define solver classes
class exp_euler(ODESolverAPI):
    init = init_exp
    build_step = build_exp_euler_step

class exp_midpoint(ODESolverAPI):
    init = init_exp
    build_step = build_exp_midpoint_step

class exp_rk4(ODESolverAPI):
    init = init_exp
    build_step = build_exp_rk4_step

# Register methods
exp_euler_info = {
    "explicit": False,
    "order": 2,
    "info": "Exponential Euler method",
    "adaptive": False,
}

exp_midpoint_info = {
    "explicit": False,
    "order": 3,
    "info": "Exponential Midpoint method",
    "adaptive": False,
}

exp_rk4_info = {
    "explicit": False,
    "order": 4,
    "info": "Exponential RK4 method",
    "adaptive": False,
}

register_method("exp_euler", exp_euler, info=exp_euler_info)
register_method("exp_midpoint", exp_midpoint, info=exp_midpoint_info)
register_method("exp_rk4", exp_rk4, info=exp_rk4_info)
