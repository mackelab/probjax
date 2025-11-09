from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from probjax.utils.odeutil.solvers.base import (
    ODEInfo,
    ODESolverAPI,
    ODEState,
    register_method,
)
from probjax.utils.typing import Array, ArrayLike, Callable


# =============================================================================
# Shared state/info classes
# =============================================================================
class ExpODEState(ODEState):
    t0: Array
    y0: Array
    f0: Optional[Array]  # cached drift or nonlinear part (depending on builder)


class ExpODEInfo(ODEInfo):
    phi_products: Optional[Array] = None


class ExpSplitState(NamedTuple):
    t0: Array
    y0: Array
    f0: Optional[Array]
    nonlin_history: Array
    history_fill: Array


# =============================================================================
# INIT helpers
# =============================================================================
def init_exp(
    t0: ArrayLike, y0: ArrayLike, *args, drift: Optional[Callable] = None
) -> ExpODEState:
    """Initialize state for general exponential integrators where
    `drift(t, y, *args)` is used.
    """
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    f0 = drift(t0, y0, *args) if drift is not None else None
    return ExpODEState(t0=t0, y0=y0, f0=f0)


# --- NEW: split-form initialization for specialized scalar-L solvers
class SplitDrift(NamedTuple):
    """
    dy/dt = L(t) y + N(t,y), with L(t) = c(t) * I (scalar multiple of identity).
    - lin_coeff(t) -> scalar c(t)
    - nonlin(t, y, *args) -> N(t, y)
    """

    lin_coeff: Callable[[Array], Array]
    nonlin: Callable[..., Array]  # (t: Array, y: Array, *args) -> Array

    def __call__(self, x: Array, t: Array, *args, **kwds) -> Array:
        nonl = self.nonlin(t, x, *args, **kwds)
        linear = self.lin_coeff(t) * x
        return linear + nonl


def _as_split(split_or_drift: Callable | SplitDrift) -> SplitDrift:
    """Ensure we always operate on a SplitDrift instance.

    When users pass a plain drift callable (as the legacy solvers expect) we
    treat the entire drift as the nonlinear part and set the linear coefficient
    to zero. Providing an explicit SplitDrift keeps the optimized path.
    """
    if isinstance(split_or_drift, SplitDrift):
        return split_or_drift

    adapter = getattr(split_or_drift, "__split_drift__", None)
    if isinstance(adapter, SplitDrift):
        return adapter

    if callable(split_or_drift):
        zero_lin = lambda t: jnp.zeros_like(jnp.asarray(t))
        return SplitDrift(lin_coeff=zero_lin, nonlin=split_or_drift)

    raise TypeError(
        "split-based exponential methods require either a SplitDrift or a drift callable"
    )


def _history_init(f0: Array, history_size: int) -> Tuple[Array, Array]:
    if history_size <= 0:
        history = jnp.zeros((0,) + f0.shape, dtype=f0.dtype)
    else:
        history = jnp.zeros((history_size,) + f0.shape, dtype=f0.dtype)
    fill = jnp.array(0, dtype=jnp.int32)
    return history, fill


def _history_push(
    history: Array, fill: Array, new_value: Array
) -> Tuple[Array, Array]:
    max_len = history.shape[0]
    if max_len == 0:
        return history, fill
    if max_len == 1:
        history = new_value[None, ...]
    else:
        history = jnp.concatenate([new_value[None, ...], history[:-1]], axis=0)
    fill = jnp.minimum(fill + 1, max_len)
    return history, fill


def _history_get(history: Array, fill: Array, idx: int, fallback: Array) -> Array:
    if history.shape[0] == 0:
        return fallback
    has_entry = fill > idx
    return jnp.where(has_entry, history[idx], fallback)


def init_exp_split(
    t0: ArrayLike,
    y0: Array,
    *args,
    split: Optional[SplitDrift] = None,
    drift: Optional[Callable] = None,
    history_size: int = 0,
) -> ExpSplitState:
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    split = split or _as_split(drift)
    f0 = split.nonlin(t0, y0, *args)  # cache N(t0, y0)
    history, fill = _history_init(f0, history_size)
    return ExpSplitState(
        t0=t0, y0=y0, f0=f0, nonlin_history=history, history_fill=fill
    )


# =============================================================================
# GENERAL exponential integrator utilities (matrix φ via block expm)
# =============================================================================
def compute_phi_functions(A: Array, dt: ArrayLike, k: int = 1):
    """
    Compute phi_0, phi_1, ..., phi_k for matrix A and step dt via block matrix exponential.
    Phi_k(z) = ∫_0^1 e^{z*(1-s)} s^{k-1}/(k-1)! ds
    """
    dt = jnp.asarray(dt)
    n = A.shape[0]
    aug = jnp.zeros((n * (k + 1), n * (k + 1)), dtype=A.dtype)

    # block diagonal A
    for i in range(k + 1):
        aug = aug.at[i * n : (i + 1) * n, i * n : (i + 1) * n].set(A)

    # superdiagonal identities
    I = jnp.eye(n, dtype=A.dtype)
    for i in range(k):
        aug = aug.at[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n].set(I)

    exp_aug = jax.scipy.linalg.expm(aug * dt)

    phi = []
    for i in range(k + 1):
        phi.append(exp_aug[0:n, i * n : (i + 1) * n])
    return phi


# =============================================================================
# ORIGINAL (general) exponential Euler / Midpoint / RK4
#   (light refactors: minor naming, comments, dtype removal that was unused)
# =============================================================================
def build_exp_euler_step(
    drift: Callable,
) -> Callable[[ExpODEState, ArrayLike], tuple[ExpODEState, ExpODEInfo]]:
    """Exponential Euler using matrix φ₀, φ₁ (requires Jacobian & expm)."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> tuple[ExpODEState, ExpODEInfo]:
        t0, y0 = state.t0, state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        # Jacobian and φ
        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)
        phi0, phi1 = compute_phi_functions(A, dt, k=1)

        # y1 = φ0 y0 + dt φ1 (f0 - A y0)
        y1 = phi0 @ y0 + dt * (phi1 @ (f0 - A @ y0))
        f1 = drift(t0 + dt, y1, *args)
        return ExpODEState(t0=t0 + dt, y0=y1, f0=f1), ExpODEInfo(
            phi_products=jnp.array([phi0, phi1])
        )

    return step_fn


def build_exp_midpoint_step(drift: Callable):
    """Exponential midpoint (requires Jacobian & expm)."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> Tuple[ExpODEState, ExpODEInfo]:
        t0, y0 = state.t0, state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)

        phi0_half, phi1_half = compute_phi_functions(A, dt / 2, k=1)
        y_mid = phi0_half @ y0 + (dt / 2) * (phi1_half @ (f0 - A @ y0))
        f_mid = drift(t0 + dt / 2, y_mid, *args)

        phi0, phi1 = compute_phi_functions(A, dt, k=1)
        y1 = phi0 @ y0 + dt * (phi1 @ (f_mid - A @ y0))
        f1 = drift(t0 + dt, y1, *args)

        return ExpODEState(t0=t0 + dt, y0=y1, f0=f1), ExpODEInfo(
            phi_products=jnp.array([phi0, phi1])
        )

    return step_fn


def build_exp_rk4_step(drift: Callable):
    """Exponential RK4 variant (requires Jacobian & expm)."""

    def step_fn(
        state: ExpODEState, dt: ArrayLike, *args
    ) -> Tuple[ExpODEState, ExpODEInfo]:
        t0, y0 = state.t0, state.y0
        f0 = state.f0 if state.f0 is not None else drift(t0, y0, *args)

        jacobian_fn = jax.jacfwd(drift, argnums=1)
        A = jacobian_fn(t0, y0, *args)

        phi0, phi1, phi2, phi3 = compute_phi_functions(A, dt, k=3)

        def nonlinear(t, y):
            return drift(t, y, *args) - A @ y

        r0 = nonlinear(t0, y0)

        k1 = r0
        y1 = y0 + dt * (phi1 @ k1)
        r1 = nonlinear(t0 + dt / 2, y1)

        k2 = r1 - k1
        y2 = y1 + dt * (phi1 @ k2 - phi2 @ k1)
        r2 = nonlinear(t0 + dt / 2, y2)

        k3 = r2 - 2 * k2 - k1
        y3 = y2 + dt * (2 * (phi1 @ k3) - 4 * (phi2 @ k2) + (phi3 @ k1))
        r3 = nonlinear(t0 + dt, y3)

        k4 = r3 - k3 - k2 - k1

        y_next = phi0 @ y0 + dt * (
            (phi1 @ (k1 + 2 * k2 + k3))
            + dt * (phi2 @ (k1 + k2))
            + (dt**2) * (phi3 @ k1)
        )
        f_next = drift(t0 + dt, y_next, *args)

        return ExpODEState(t0=t0 + dt, y0=y_next, f0=f_next), ExpODEInfo(
            phi_products=jnp.array([phi0, phi1, phi2, phi3])
        )

    return step_fn


# =============================================================================
# SPECIALIZED (scalar L): exponential Adams–Bashforth for L(t)=c(t)I
#   - No Jacobian, no matrix expm; φ are scalars.
#   - AB2 ~ DPM-Solver++-2M; AB3 ~ DPM-Solver++-3M.
# =============================================================================
def _phi1_scalar(z: Array) -> Array:
    small = jnp.abs(z) < 1e-4
    series = 1.0 + 0.5 * z + (z * z) / 6.0 + (z * z * z) / 24.0
    return jnp.where(small, series, jnp.expm1(z) / z)


def _phi2_scalar(z: Array) -> Array:
    small = jnp.abs(z) < 1e-3
    series = 0.5 + z / 6.0 + (z * z) / 24.0 + (z * z * z) / 120.0
    return jnp.where(small, series, (jnp.expm1(z) - z) / (z * z))


def _phi3_scalar(z: Array) -> Array:
    small = jnp.abs(z) < 1e-2
    series = (1.0 / 6.0) + z / 24.0 + (z * z) / 120.0 + (z * z * z) / 720.0
    num = jnp.expm1(z) - z - 0.5 * (z * z)
    return jnp.where(small, series, num / (z * z * z))


def build_exp_ab2_scalarL(split: SplitDrift | Callable):
    """
    Exponential AB2 with scalar L. Signature matches your solvers:
      step(state, dt, y_nm1, N_nm1, *user_args) -> (state', info)
    where N_nm1 is the cached nonlinearity at (t_{n-1}, y_{n-1}).
    """

    split = _as_split(split)

    def step_fn(
        state: ExpSplitState, dt: ArrayLike, *args
    ) -> Tuple[ExpSplitState, ExpODEInfo]:
        t_n, y_n = state.t0, state.y0
        # Current N_n (nonlinear) from cache or compute
        N_n = state.f0 if state.f0 is not None else split.nonlin(t_n, y_n, *args)

        history = state.nonlin_history
        fill = state.history_fill
        N_nm1 = _history_get(history, fill, 0, N_n)

        c_np1 = split.lin_coeff(t_n + dt)
        z = c_np1 * dt
        r = jnp.exp(z)
        ph1 = _phi1_scalar(z)

        # y_{n+1} = e^{z} y_n + dt φ1(z) [2 N_n - N_{n-1}]
        y_np1 = r * y_n + dt * ph1 * (2.0 * N_n - N_nm1)

        N_np1 = split.nonlin(t_n + dt, y_np1, *args)
        history, fill = _history_push(history, fill, N_n)
        return (
            ExpSplitState(
                t0=t_n + dt,
                y0=y_np1,
                f0=N_np1,
                nonlin_history=history,
                history_fill=fill,
            ),
            ExpODEInfo(phi_products=None),
        )

    return step_fn


def build_exp_ab3_scalarL(split: SplitDrift | Callable):
    """
    Exponential AB3 with scalar L.
      step(state, dt, y_nm1, N_nm1, y_nm2, N_nm2, *user_args)
    """

    split = _as_split(split)

    def step_fn(
        state: ExpSplitState, dt: ArrayLike, *args
    ) -> Tuple[ExpSplitState, ExpODEInfo]:
        t_n, y_n = state.t0, state.y0
        N_n = state.f0 if state.f0 is not None else split.nonlin(t_n, y_n, *args)

        c_np1 = split.lin_coeff(t_n + dt)
        z = c_np1 * dt
        r = jnp.exp(z)
        ph1 = _phi1_scalar(z)
        ph2 = _phi2_scalar(z)
        ph3 = _phi3_scalar(z)

        history = state.nonlin_history
        fill = state.history_fill
        N_nm1 = _history_get(history, fill, 0, N_n)
        N_nm2 = _history_get(history, fill, 1, N_nm1)

        has_nm1 = fill > 0
        has_nm2 = fill > 1

        dN = N_n - N_nm1
        d2N = N_n - 2.0 * N_nm1 + N_nm2

        y_ab3 = r * y_n + dt * (ph1 * N_n + ph2 * dN + ph3 * d2N)
        y_ab2 = r * y_n + dt * ph1 * (2.0 * N_n - N_nm1)
        y_ab1 = r * y_n + dt * ph1 * N_n

        y_np1 = jnp.where(has_nm2, y_ab3, jnp.where(has_nm1, y_ab2, y_ab1))

        N_np1 = split.nonlin(t_n + dt, y_np1, *args)
        history, fill = _history_push(history, fill, N_n)
        return (
            ExpSplitState(
                t0=t_n + dt,
                y0=y_np1,
                f0=N_np1,
                nonlin_history=history,
                history_fill=fill,
            ),
            ExpODEInfo(phi_products=None),
        )

    return step_fn


# =============================================================================
# Thin solver classes and registry entries
# =============================================================================
class exp_euler(ODESolverAPI):
    init = staticmethod(init_exp)
    build_step = staticmethod(build_exp_euler_step)


class exp_midpoint(ODESolverAPI):
    init = staticmethod(init_exp)
    build_step = staticmethod(build_exp_midpoint_step)


class exp_rk4(ODESolverAPI):
    init = staticmethod(init_exp)
    build_step = staticmethod(build_exp_rk4_step)


class exp_ab2_scalarL(ODESolverAPI):
    @staticmethod
    def init(t0, y0, *args, drift=None, **kwargs):
        return init_exp_split(
            t0,
            y0,
            *args,
            drift=drift,
            history_size=1,
            **kwargs,
        )

    build_step = staticmethod(build_exp_ab2_scalarL)


class exp_ab3_scalarL(ODESolverAPI):
    @staticmethod
    def init(t0, y0, *args, drift=None, **kwargs):
        return init_exp_split(
            t0,
            y0,
            *args,
            drift=drift,
            history_size=2,
            **kwargs,
        )

    build_step = staticmethod(build_exp_ab3_scalarL)


# --- Registration (kept your format) -----------------------------------------
register_method(
    "exp_euler",
    exp_euler,
    info={
        "explicit": False,
        "order": 2,
        "info": "Exponential Euler (general; matrix φ, needs Jacobian/expm)",
        "adaptive": False,
    },
)

register_method(
    "exp_midpoint",
    exp_midpoint,
    info={
        "explicit": False,
        "order": 3,
        "info": "Exponential Midpoint (general; matrix φ, needs Jacobian/expm)",
        "adaptive": False,
    },
)

register_method(
    "exp_rk4",
    exp_rk4,
    info={
        "explicit": False,
        "order": 4,
        "info": "Exponential RK4 (general; matrix φ, needs Jacobian/expm)",
        "adaptive": False,
    },
)

register_method(
    "exp_ab2_scalarL",
    exp_ab2_scalarL,
    info={
        "explicit": False,
        "order": 2,
        "info": "Exponential AB2 with scalar linear operator (DPM++-2M analogue)",
        "adaptive": False,
    },
)

register_method(
    "exp_ab3_scalarL",
    exp_ab3_scalarL,
    info={
        "explicit": False,
        "order": 3,
        "info": "Exponential AB3 with scalar linear operator (DPM++-3M analogue)",
        "adaptive": False,
    },
)
