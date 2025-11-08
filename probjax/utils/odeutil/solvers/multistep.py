# Coefficient functions (not initialized at import)
def get_ab_coeffs(order: int, dtype) -> Array:
    import jax.numpy as jnp
    coeffs = {
        1: [1.0],
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24],
    }
    return jnp.array(coeffs[order], dtype=dtype)

def get_am_coeffs(order: int, dtype) -> Array:
    import jax.numpy as jnp
    coeffs = {
        1: [1.0],
        2: [1/2, 1/2],
        3: [5/12, 2/3, -1/12],
        4: [3/8, 19/24, -5/24, 1/24],
    }
    return jnp.array(coeffs[order], dtype=dtype)

def get_bdf_a(order: int, dtype) -> Array:
    import jax.numpy as jnp
    coeffs = {
        1: [1.0],
        2: [1.0, -4/3, 1/3],
        3: [1.0, -18/11, 9/11, -2/11],
    }
    return jnp.array(coeffs[order], dtype=dtype)

def get_bdf_b(order: int) -> float:
    coeffs = {1: 1.0, 2: 2/3, 3: 6/11}
    return coeffs[order]
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from probjax.utils.odeutil.solvers.base import (
    ODEInfo,
    ODESolverAPI,
    ODEState,
    register_method,
)
from probjax.utils.typing import Array, ArrayLike, Callable

# ============================================================================
# Shared state + info
# ============================================================================
class LMSState(ODEState):
    t0: Array
    y0: Array
    f0: Optional[Array]       # f(t0, y0)
    hist_t: Optional[list[Array]] = None
    hist_f: Optional[list[Array]] = None
    hist_y: Optional[list[Array]] = None

class LMSInfo(ODEInfo):
    newton_iters: int = 0
    converged: bool = True

# ============================================================================
# Adams–Bashforth coefficients (constant step) - templates (will be cast to dtype)
# ============================================================================
_AB_COEFFS_TEMPLATE: dict[int, list[float]] = {
    1: [1.0],                                  # Euler
    2: [3/2, -1/2],
    3: [23/12, -16/12, 5/12],
    4: [55/24, -59/24, 37/24, -9/24],
}

# Adams–Moulton (implicit) predictor-corrector coefficients
_AM_COEFFS_TEMPLATE: dict[int, list[float]] = {
    1: [1.0],                                  # Implicit Euler (trapezoid has k=2)
    2: [1/2, 1/2],                             # Trapezoid: 0.5*f_{n+1} + 0.5*f_n
    3: [5/12, 2/3, -1/12],
    4: [3/8, 19/24, -5/24, 1/24],
}

# Backward Differentiation (BDF) coefficients for y_{n+1} - sum a_j y_{n-j} = h * b * f_{n+1}
_BDF_A_TEMPLATE: dict[int, list[float]] = {
    1: [1.0],                                  # y_{n+1} - y_n = h f_{n+1}
    2: [1.0, -4/3, 1/3],
    3: [1.0, -18/11, 9/11, -2/11],
}
_BDF_B: dict[int, float] = {1: 1.0, 2: 2/3, 3: 6/11}


def _get_coeffs(order: int, coeff_dict: dict[int, list[float]], dtype) -> Array:
    """Get coefficients with proper dtype."""
    return jnp.array(coeff_dict[order], dtype=dtype)

# ============================================================================
# Utilities
# ============================================================================
def _ensure_array(x: ArrayLike) -> Array:
    return jnp.asarray(x)


def init_lms(t0: ArrayLike, y0: ArrayLike, *args, drift: Callable) -> LMSState:
    """Initialize Linear Multistep state."""
    t0_arr = _ensure_array(t0)
    y0_arr = _ensure_array(y0)
    f0 = drift(t0_arr, y0_arr, *args)
    return LMSState(
        t0=t0_arr, y0=y0_arr, f0=f0, hist_t=[t0_arr], hist_f=[f0], hist_y=[y0_arr]
    )


def _append_history(
    state: LMSState, t: Array, y: Array, f: Array, kmax: int
) -> LMSState:
    """Append new values to history and keep only last kmax entries."""
    ht = (state.hist_t or []) + [t]
    hf = (state.hist_f or []) + [f]
    hy = (state.hist_y or []) + [y]
    # keep last kmax entries
    ht, hf, hy = ht[-kmax:], hf[-kmax:], hy[-kmax:]
    return state._replace(hist_t=ht, hist_f=hf, hist_y=hy)

def _need_bootstrap(state: LMSState, order: int) -> bool:
    """Check if bootstrap steps are needed to build up history."""
    return len(state.hist_f or []) < order


def build_bootstrapper(
    one_step: Callable,
) -> Callable[[LMSState, ArrayLike], tuple[LMSState, ODEInfo]]:
    """Build a bootstrap function from a single-step method."""

    def bootstrap(
        state: LMSState, dt: ArrayLike, *args
    ) -> tuple[LMSState, ODEInfo]:
        # one_step must take (state_like, dt, *args) -> (state_like, info)
        return one_step(state, dt, *args)

    return bootstrap


# ============================================================================
# AB(k) explicit multistep
# ============================================================================
def build_ab(
    drift: Callable,
    order: int = 3,
    bootstrap_step: Optional[Callable] = None,
) -> Callable[[LMSState, ArrayLike], tuple[LMSState, LMSInfo]]:
    """Build Adams-Bashforth explicit multistep method."""
    assert order in (1, 2, 3, 4), "AB order must be 1-4"

    def step(state: LMSState, dt: ArrayLike, *args) -> tuple[LMSState, LMSInfo]:
        t_n, y_n = state.t0, state.y0
        if _need_bootstrap(state, order):
            if bootstrap_step is None:
                raise ValueError("AB needs a bootstrap single-step method to fill history.")
            # do one single-step bootstrap
            new_state, _ = bootstrap_step(state, dt, *args)
            return _append_history(new_state, new_state.t0, new_state.y0, new_state.f0, order), LMSInfo()

        # Get coefficients with proper dtype
        beta = get_ab_coeffs(order, y_n.dtype)
        Fs = state.hist_f[::-1]  # [f_n, f_{n-1}, ...]
        incr = sum(b * f for b, f in zip(beta, Fs))
        y_np1 = y_n + dt * incr
        t_np1 = t_n + dt
        f_np1 = drift(t_np1, y_np1, *args)

        new_state = LMSState(t0=t_np1, y0=y_np1, f0=f_np1,
                             hist_t=state.hist_t, hist_f=state.hist_f, hist_y=state.hist_y)
        new_state = _append_history(new_state, t_np1, y_np1, f_np1, order)
        return new_state, LMSInfo()

    return step

# ============================================================================
# AM(k) implicit multistep (single Newton/fixed-point corrector)
# y_{n+1} = y_n + h * sum_{j=0}^{k-1} a_j f_{n-j} + h * a_{-1} f_{n+1}
# Coeff vector for AM(k) stored as [a_{-1}, a_0, a_1, ..., a_{k-1}]
# ============================================================================
def build_am(
    drift: Callable,
    order: int = 2,
    newton_iters: int = 1,
    tol: float = 0.0,
    bootstrap_step: Optional[Callable] = None,
) -> Callable[[LMSState, ArrayLike], tuple[LMSState, LMSInfo]]:
    """Build Adams-Moulton implicit multistep method."""
    assert order in (1, 2, 3, 4), "AM order must be 1-4"

    def step(state: LMSState, dt: ArrayLike, *args) -> tuple[LMSState, LMSInfo]:
        t_n, y_n, f_n = state.t0, state.y0, state.f0
        if _need_bootstrap(state, order):
            if bootstrap_step is None:
                raise ValueError("AM needs a bootstrap single-step method to fill history.")
            new_state, _ = bootstrap_step(state, dt, *args)
            return _append_history(new_state, new_state.t0, new_state.y0, new_state.f0, order), LMSInfo()

        # Get coefficients with proper dtype
        a = get_am_coeffs(order, y_n.dtype)
        a_m1, a_hist = a[0], a[1:]  # a_{-1}, [a_0, ..., a_{k-1}]
        Fs = state.hist_f[::-1]  # [f_n, f_{n-1}, ...]
        # predictor: AB1 (Euler) or use the explicit Adams formula of matching order
        y_pred = y_n + dt * Fs[0]  # cheap predictor

        # corrector: single Newton or fixed-point
        t_np1 = t_n + dt
        y_new = y_pred
        iters = 0
        converged = True
        for it in range(max(1, newton_iters)):
            rhs_hist = sum(a_j * f for a_j, f in zip(a_hist, Fs))
            g = y_n + dt * (rhs_hist + a_m1 * drift(t_np1, y_new, *args)) - y_new
            # simple fixed-point if no tol/iters specified
            if newton_iters <= 1 and tol <= 0:
                y_new = y_new + g  # Picard-like
            else:
                # Jacobian: I - h a_{-1} J_f
                J = jax.jacfwd(lambda yy: drift(t_np1, yy, *args))(y_new)
                mat = jnp.eye(y_new.size, dtype=y_new.dtype).reshape(y_new.shape + y_new.shape) - dt * a_m1 * J
                # solve mat * delta = g
                delta = jnp.linalg.solve(mat.reshape(y_new.size, y_new.size), g.reshape(-1)).reshape(y_new.shape)
                y_new = y_new + delta
                iters += 1
                if jnp.linalg.norm(delta) < tol:
                    break

        f_np1 = drift(t_np1, y_new, *args)
        new_state = LMSState(t0=t_np1, y0=y_new, f0=f_np1,
                             hist_t=state.hist_t, hist_f=state.hist_f, hist_y=state.hist_y)
        new_state = _append_history(new_state, t_np1, y_new, f_np1, order)
        return new_state, LMSInfo(newton_iters=iters, converged=converged)

    return step

# ============================================================================
# BDF(k) implicit (k=1..3 here)
# sum_{j=0}^k a_j y_{n+1-j} = h * b f_{n+1}
# ============================================================================
def build_bdf(
    drift: Callable,
    order: int = 2,
    newton_iters: int = 1,
    tol: float = 0.0,
    bootstrap_step: Optional[Callable] = None,
) -> Callable[[LMSState, ArrayLike], tuple[LMSState, LMSInfo]]:
    """Build Backward Differentiation Formula (BDF) implicit method."""
    assert order in (1, 2, 3), "BDF order must be 1-3"
    b = get_bdf_b(order)

    def step(state: LMSState, dt: ArrayLike, *args) -> tuple[LMSState, LMSInfo]:
        t_n, y_n = state.t0, state.y0
        if _need_bootstrap(state, order):
            if bootstrap_step is None:
                raise ValueError("BDF needs a bootstrap single-step method to fill history.")
            new_state, _ = bootstrap_step(state, dt, *args)
            return _append_history(new_state, new_state.t0, new_state.y0, new_state.f0, order), LMSInfo()

        # Get coefficients with proper dtype
        a = get_bdf_a(order, y_n.dtype)
        # form RHS with history y_n, y_{n-1}, ...
        Ys = state.hist_y[::-1]  # [y_n, y_{n-1}, ...]
        rhs = -sum(a_j * y for a_j, y in zip(a[1:], Ys))  # move known terms to RHS

        # implicit equation: a0 * y_{n+1} = rhs + h b f(t_{n+1}, y_{n+1})
        a0 = a[0]
        t_np1 = t_n + dt

        # predictor: last y
        y_new = y_n
        iters = 0
        for it in range(max(1, newton_iters)):
            g = a0 * y_new - (rhs + dt * b * drift(t_np1, y_new, *args))
            if newton_iters <= 1 and tol <= 0:
                y_new = y_new - g / a0     # fixed-point-ish (good if mildly stiff)
            else:
                J = jax.jacfwd(lambda yy: drift(t_np1, yy, *args))(y_new)
                mat = a0 * jnp.eye(y_new.size, dtype=y_new.dtype).reshape(y_new.shape + y_new.shape) - dt * b * J
                delta = jnp.linalg.solve(mat.reshape(y_new.size, y_new.size), g.reshape(-1)).reshape(y_new.shape)
                y_new = y_new - delta
                iters += 1
                if jnp.linalg.norm(delta) < tol:
                    break

        f_np1 = drift(t_np1, y_new, *args)
        new_state = LMSState(t0=t_np1, y0=y_new, f0=f_np1,
                             hist_t=state.hist_t, hist_f=state.hist_f, hist_y=state.hist_y)
        new_state = _append_history(new_state, t_np1, y_new, f_np1, order)
        return new_state, LMSInfo(newton_iters=iters, converged=True)

    return step

# ============================================================================
# Thin solver classes to match your registry
# ============================================================================
class ab2(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_ab(drift, order=2, **kw))

class ab3(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_ab(drift, order=3, **kw))

class ab4(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_ab(drift, order=4, **kw))

class am2(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_am(drift, order=2, **kw))

class am3(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_am(drift, order=3, **kw))

class am4(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_am(drift, order=4, **kw))

class bdf1(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_bdf(drift, order=1, **kw))

class bdf2(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_bdf(drift, order=2, **kw))

class bdf3(ODESolverAPI):
    init = staticmethod(init_lms)
    build_step = staticmethod(lambda drift, **kw: build_bdf(drift, order=3, **kw))

# Register
register_method("ab2", ab2,  info={"explicit": True,  "order": 2, "info": "Adams–Bashforth 2", "adaptive": False})
register_method("ab3", ab3,  info={"explicit": True,  "order": 3, "info": "Adams–Bashforth 3", "adaptive": False})
register_method("ab4", ab4,  info={"explicit": True,  "order": 4, "info": "Adams–Bashforth 4", "adaptive": False})

register_method("am2", am2,  info={"explicit": False, "order": 2, "info": "Adams–Moulton 2 (trapezoid)", "adaptive": False})
register_method("am3", am3,  info={"explicit": False, "order": 3, "info": "Adams–Moulton 3", "adaptive": False})
register_method("am4", am4,  info={"explicit": False, "order": 4, "info": "Adams–Moulton 4", "adaptive": False})

register_method("bdf1", bdf1, info={"explicit": False, "order": 1, "info": "BDF1 (implicit Euler)", "adaptive": False})
register_method("bdf2", bdf2, info={"explicit": False, "order": 2, "info": "BDF2", "adaptive": False})
register_method("bdf3", bdf3, info={"explicit": False, "order": 3, "info": "BDF3", "adaptive": False})
