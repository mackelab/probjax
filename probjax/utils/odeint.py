import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax import lax
from jax import core
from jax.tree_util import tree_leaves

from functools import partial
from jaxtyping import Array, Float, PyTree, Int
from typing import Callable, Optional


from probjax.core import custom_inverse
from probjax.utils.interpolation import linear_interpolation
from probjax.utils.solver import root
from probjax.utils.linalg import is_triangular_matrix
from probjax.utils.jaxutils import flatten1d


METHOD_STEP_FN = {}
METHOD_INFO = {}


def register_method(name: str, step_fn: Callable, info: Optional[dict] = None):
    """General method to register a step_fn for an ODE solver, thereby creating a new method.

    Args:
        name (str): Name of the method
        step_fn (Callable): Step function. This function should have the following signature:
            func(drift: Callable, t0: Array, y0: Array, f0: Array, dt: Array, **kwargs) -> Tuple[y1: Array, f1: Array, Tuple[y_error: Optional[Array], k: Optional[Array]]]
        info (Optional[dict], optional): Some information about your method. Defaults to None.

    Returns:
        _type_: _description_
    """
    METHOD_STEP_FN[name] = step_fn
    METHOD_INFO[name] = info
    return step_fn


def register_runge_kutta_method(
    name: str,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Optional[Array] = None,
    info: Optional[str] = None,
) -> Callable:
    """Register a Runge-Kutta method. This function will create a step function for the method and register it. A Runge-Kutta method is defined by the following equations:

    y1 = y0 + dt * sum_i b_sol[i] * k[i]
    k[i] = f(t0 + dt * c[i], y0 + dt * sum_j A[i,j] * k[j])

    and thus uniquely defined given a butcher tableau (c, A, b_sol).

    Args:
        name (str): Name of the method
        c (Array): The c vector of the butcher tableau. Time evaluation points.
        A (Array): The A matrix of the butcher tableau. Defines the coefficients of the evaluation points k.
        b_sol (Array): The b_sol vector of the butcher tableau. Defines the coefficients of the solution y1.
        b_error (Optional[Array], optional): How to compute an estimate of the local error. Defaults to None.
        info (Optional[str], optional): Information about the solver. Defaults to None.

    Returns:
        Callable: The step_fn of the method.
    """
    stages = len(c)
    if stages >= 5:
        order = stages - 1
    else:
        order = stages

    assert jnp.all(c >= 0) and jnp.all(c <= 1), "c must be between 0 and 1"
    assert jnp.allclose(jnp.sum(b_sol), 1.0), "b_sol must sum to 1"

    assert A.shape == (
        stages,
        stages,
    ), f"Expected A.shape == ({stages}, {stages}), got {A.shape}"
    assert b_sol.shape == (
        stages,
    ), f"Expected b_sol.shape == ({stages},), got {b_sol.shape}"

    if jnp.allclose(A[-1], b_sol) and c[-1] == 1.0:
        last_equals_next = True
    else:
        last_equals_next = False

    if is_triangular_matrix(A) and jnp.diag(A).sum() == 0:
        explicit = True
    else:
        explicit = False

    if b_error is None:
        adaptive = False
    else:
        adaptive = True

    METHOD_INFO[name] = {
        "explicit": explicit,
        "order": order,
        "c": c,
        "A": A,
        "b_sol": b_sol,
        "b_error": b_error,
        "adaptive": adaptive,
        "info": info,
    }

    if explicit:
        step_fn = partial(
            explicit_runge_kutta_step,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            stages=stages,
            last_equals_next=last_equals_next,
        )
    else:
        step_fn = partial(
            implicit_runge_kutta_step,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            stages=order,
            last_equals_next=last_equals_next,
        )

    METHOD_STEP_FN[name] = step_fn
    return step_fn


# def register_adams_bashforth_method(
#     name: str,
#     a: Array,
#     b: Array,
#     order: int,
#     info: Optional[str] = None,
# ) -> Callable:
#     stages = len(b)

#     assert a[-1] = 1., "a[-1] must be 1."
#     assert jnp.sum(b) == 1., "b must sum to 1."

#     explicit = b[-1] == 0.0

#     # TODO


def get_step_fn(method: str, dtype: Optional[Float] = None):
    """Returns the step function for a given method.

    Returns:
        Callable: Step function with corresponding method name.
    """
    step_fn = METHOD_STEP_FN[method]
    if dtype is not None:
        # Right numerical precision
        if hasattr(step_fn, "keywords"):
            step_fn.keywords["c"] = step_fn.keywords["c"].astype(dtype)
            step_fn.keywords["A"] = step_fn.keywords["A"].astype(dtype)
            step_fn.keywords["b_sol"] = step_fn.keywords["b_sol"].astype(dtype)
            if step_fn.keywords["b_error"] is not None:
                step_fn.keywords["b_error"] = step_fn.keywords["b_error"].astype(dtype)

    return step_fn


def get_method_info(method: str):
    """Returns the information about a given method."""
    return METHOD_INFO[method]


def get_methods():
    """Returns a list of all registered methods."""
    return list(METHOD_STEP_FN.keys())


# def explicit_adam_beth_method(
#     drift: Callable,
#     t0: Array,
#     ys: Array,
#     fs: Array,
#     dt: Array,
#     a: Array,
#     b: Array,
#     order: int,
# ):
#     s = len(a)


@partial(jax.jit, static_argnums=(0, 9, 10))
def explicit_runge_kutta_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Array,
    stages: int,
    last_equals_next: bool,
):
    def body_fun(i, k):
        ti = t0 + dt * c[i]
        yi = y0 + dt * jnp.dot(A[i, :], k)
        ft = drift(ti, yi)
        return k.at[i, :].set(ft)

    k = jnp.zeros((stages, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = lax.fori_loop(1, stages + 1, body_fun, k)

    y1 = dt * jnp.dot(b_sol, k) + y0
    if last_equals_next:
        f1 = k[-1]
    else:
        f1 = drift(t0 + dt, y1)

    if b_error is None:
        y1_error = None
    else:
        y1_error = dt * jnp.dot(b_error, k)

    return y1, f1, (y1_error, k)


def implicit_runge_kutta_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Array,
    stages: int = 2,
):
    ts = t0 + dt * c
    ts = ts.reshape(-1, 1)

    # Solve implicit equation
    def f(k):
        return k - dt * drift(y0 + jnp.dot(A, k), ts)

    # Uses root finding to solve implicit equation
    k0 = jnp.ones((stages, f0.shape[0]), f0.dtype) * f0
    k = root(f, k0)

    # Compute solution
    y1 = f0 * jnp.dot(b_sol, k) + y0
    f1 = lax.cond(c[-1] == 1.0, lambda _: k[-1], lambda _: drift(t0 + dt, y1), None)

    if b_error is None:
        y1_error = None
    else:
        y1_error = dt.astype(f0.dtype) * jnp.dot(b_error, k)

    return y1, f1, (y1_error, k)


def initial_step_size(
    drift: Callable,
    t0: Array,
    y0: Array,
    order: int,
    rtol: float,
    atol: float,
    f0: Array,
):
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    dtype = y0.dtype

    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
    d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
    y1 = y0 + h0.astype(dtype) * f0
    f1 = drift(t0 + h0, y1)
    d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

    h1 = jnp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    return jnp.minimum(100.0 * h0, h1)


def mean_error_ratio(
    error_estimate: Array, rtol: float, atol: float, y0: Array, y1: Array
):
    err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
    return jnp.sqrt(jnp.mean(jnp.abs(err_ratio)))


def step_size_adaption(
    last_step: Array,
    mean_error_ratio: Array,
    dtmin: float = -jnp.inf,
    dtmax: float = jnp.inf,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.1,
    order: int = 5,
):
    """Compute optimal Runge-Kutta stepsize."""
    dfactor = jnp.where(mean_error_ratio < 1, 1.0, dfactor)

    factor = jnp.minimum(
        ifactor, jnp.maximum(mean_error_ratio ** (-1.0 / order) * safety, dfactor)
    )
    dt = jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)
    return jnp.clip(dt, dtmin, dtmax)


# Explicit Runge-Kutta methods
# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods

# 1st order
# Euler's method
c = jnp.array([0])
A = jnp.array([[0]])
b_sol = jnp.array([1])
b_error = None
info = {
    "explicit": True,
    "order": 1,
    "c": c,
    "A": A,
    "b_sol": b_sol,
    "b_error": None,
    "info": "Euler's method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
@partial(jax.jit, static_argnums=(0,))
def _euler_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
):
    y1 = y0 + f0 * dt
    f1 = drift(t0 + dt, y1)
    return y1, f1, None


register_method("euler", _euler_step, info)

# 2nd order
# Heun's method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method("heun", c, A, b_sol, b_error, "Heun's method")

# Adaptive Heun's method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = jnp.array([1.0, 0.0])
register_runge_kutta_method(
    "heun_euler", c, A, b_sol, b_error, "Heun's Euler adaptive method"
)

# Midpoint method
c = jnp.array([0, 1 / 2])
A = jnp.array([[0, 0], [1 / 2, 0]])
b_sol = jnp.array([0, 1])
b_error = None
register_runge_kutta_method("midpoint", c, A, b_sol, b_error, "Midpoint method")

# Ralston's method
c = jnp.array([0, 2 / 3])
A = jnp.array([[0, 0], [2 / 3, 0]])
b_sol = jnp.array([1 / 4, 3 / 4])
b_error = None
register_runge_kutta_method("ralston", c, A, b_sol, b_error, "Ralston's method")


# 3rd order
# Kutta's third-order method
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
b_sol = jnp.array([1 / 6, 2 / 3, 1 / 6])
b_error = None
register_runge_kutta_method("rk3", c, A, b_sol, b_error, "Kutta's third-order method")

# Fehlberg's RK3(2) method (explicit) (adaptive)
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [1 / 256, 255 / 256, 0]])
b_sol = jnp.array([1 / 512, 255 / 256, 1 / 512])
b_error = jnp.array([1 / 256, 255 / 256, 0])
register_runge_kutta_method("rk3(2)", c, A, b_sol, b_error, "Fehlberg's RK3(2) method")

# Bosh3 method
c = jnp.array([0, 1 / 2, 3 / 4])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9])
b_error = None
register_runge_kutta_method("bosh3", c, A, b_sol, b_error, "Bosh3 method")

# Heun's third-order method
c = jnp.array([0, 1 / 3, 2 / 3])
A = jnp.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]])
b_sol = jnp.array([1 / 4, 0, 3 / 4])
b_error = None
register_runge_kutta_method("heun3", c, A, b_sol, b_error, "Heun's third-order method")

# Van der Houwen's/Wray's method
c = jnp.array([0, 8 / 15, 2 / 3])
A = jnp.array([[0, 0, 0], [8 / 15, 0, 0], [1 / 4, 5 / 12, 0]])
b_sol = jnp.array([1 / 4, 0, 3 / 4])
b_error = None
register_runge_kutta_method(
    "vanderhouwen", c, A, b_sol, b_error, "Van der Houwen's/Wray's method"
)

# Ralston's third-order method
c = jnp.array([0, 1 / 3, 2 / 3])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9])
b_error = None
register_runge_kutta_method(
    "ralston3", c, A, b_sol, b_error, "Ralston's third-order method"
)

# Strong stability preserving Runge-Kutta methods of order 3
c = jnp.array([0, 1 / 2, 1])
A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0]])
b_sol = jnp.array([1 / 6, 1 / 6, 2 / 3])
b_error = None
register_runge_kutta_method(
    "ssprk3",
    c,
    A,
    b_sol,
    b_error,
    "Strong stability preserving Runge-Kutta methods of order 3",
)


# 4th order
# Classic Runge-Kutta method
c = jnp.array([0, 1 / 2, 1 / 2, 1])
A = jnp.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
b_sol = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
b_error = None
register_runge_kutta_method("rk4", c, A, b_sol, b_error, "Classic Runge-Kutta method")

# Bogacki-Shampine method, RK4(3) (explicit) (adaptive)
c = jnp.array([0, 1 / 2, 3 / 4, 1])
A = jnp.array(
    [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
)
b_sol = jnp.array([2 / 9, 1 / 3, 4 / 9, 0])
b_error = jnp.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])
register_runge_kutta_method("rk4(5)", c, A, b_sol, b_error, "Bogacki-Shampine method")

# 3/8 rule
c = jnp.array([0, 1 / 3, 2 / 3, 1])
A = jnp.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
b_sol = jnp.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
b_error = None
register_runge_kutta_method("3/8", c, A, b_sol, b_error, "3/8 rule")

# Ralston's method of order 4
c = jnp.array([0.0, 0.4, 0.45573725, 1.0])
A = jnp.array(
    [
        [0, 0, 0, 0],
        [0.4, 0, 0, 0],
        [0.29697761, 0.15875964, 0, 0],
        [0.21810040, -3.05096516, 3.83286476, 0],
    ]
)
b_sol = jnp.array([0.17476028, -0.55148066, 1.20553560, 0.17118478])
b_error = None
register_runge_kutta_method(
    "ralston4", c, A, b_sol, b_error, "Ralston's method of order 4"
)

# 5th order

# Runge-Kutta method of order 5
c = jnp.array([0, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [1 / 8, 1 / 8, 0, 0, 0, 0],
        [0, 0, 1 / 2, 0, 0, 0],
        [3 / 16, -3 / 8, 3 / 8, 9 / 16, 0, 0],
        [-3 / 7, 8 / 7, 6 / 7, -12 / 7, 8 / 7, 0],
    ]
)
b_sol = jnp.array([7 / 90, 0, 32 / 90, 12 / 90, 32 / 90, 7 / 90])
b_error = None
register_runge_kutta_method(
    "rk5", c, A, b_sol, b_error, "Runge-Kutta method of order 5"
)

# Fehlberg's RK5(4) method (explicit) (adaptive)
c = jnp.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 4, 0, 0, 0, 0, 0],
        [3 / 32, 9 / 32, 0, 0, 0, 0],
        [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
    ]
)
b_sol = jnp.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
b_error = jnp.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
register_runge_kutta_method("rk5(4)", c, A, b_sol, b_error, "RK5(4)")

# 6th order
# Runge-Kutta method of order 6
c = jnp.array([0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 6, 0, 0, 0, 0, 0, 0],
        [1 / 12, 1 / 12, 0, 0, 0, 0, 0],
        [1 / 8, 0, 3 / 8, 0, 0, 0, 0],
        [91 / 500, -27 / 100, 78 / 125, 8 / 125, 0, 0, 0],
        [-11 / 20, 27 / 20, 12 / 5, -36 / 5, 5 / 2, 0, 0],
        [1 / 12, 0, 27 / 32, -4 / 3, 125 / 96, 5 / 48, 0],
    ]
)
b_sol = jnp.array([1 / 12, 0, 27 / 32, -4 / 3, 125 / 96, 5 / 48, 0])
b_error = None
register_runge_kutta_method(
    "rk6", c, A, b_sol, b_error, "Runge-Kutta method of order 6"
)


# Dormand-Prince method
c = jnp.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ]
)
b_sol = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
b_error = jnp.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)
register_runge_kutta_method("dopri5", c, A, b_sol, b_error, "Dormand-Prince method")


# Implicit Runge-Kutta methods
# https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods

# 1st order
# Implicit Euler method
c = jnp.array([1.0])
A = jnp.array([[1.0]])
b_sol = jnp.array([1.0])
b_error = None
info = {
    "explicit": False,
    "order": 1,
    "c": c,
    "A": A,
    "b_sol": b_sol,
    "b_error": None,
    "info": "Implicit Euler method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
@partial(jax.jit, static_argnums=(0,))
def _implicit_euler_step(
    drift: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    dt: Array,
):
    def f(x):
        return x - y0 - f0 * dt

    y1 = root(f, y0)
    f1 = drift(t0 + dt, y1)
    return y1, f1, None


register_method("implicit_euler", _implicit_euler_step, info)


# 2nd order
# Implicit trapezoidal rule
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method(
    "implicit_trapezoidal", c, A, b_sol, b_error, "Implicit trapezoidal rule"
)

# Implicit Crank-Nicolson method
c = jnp.array([0, 1])
A = jnp.array([[0, 0], [1 / 2, 0]])
b_sol = jnp.array([1 / 2, 1 / 2])
b_error = None
register_runge_kutta_method(
    "implicit_crank_nicolson", c, A, b_sol, b_error, "Implicit Crank-Nicolson method"
)

# TODO This seems to be wrong...

# Gauss-Legendre method
# c = jnp.array([1 / 2 - jnp.sqrt(3) / 6, 1 / 2 + jnp.sqrt(3) / 6])
# A = jnp.array([[1 / 4, 1 / 4 - jnp.sqrt(3) / 6], [1 / 4 + jnp.sqrt(3) / 6, 1 / 4]])
# b_sol = jnp.array([1 / 2, 1 / 2])
# b_error = None
# register_runge_kutta_method("gauss_legendre", c, A, b_sol, b_error, "Gauss-Legendre")


def _odeint_on_grid(drift: Callable, y0: Array, ts: Array, step_fn: Callable):
    """Solve an ordinary differential equation discretized on a grid i.e. with fixed step size.

    Args:
        drift (Callable): Drift function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        step_fn (Callable): Step function.

    """
    # Time steps
    dts = ts[1:] - ts[:-1]

    def scan_fun(carry, data):
        t0, y0, f0 = carry
        t1, dt = data
        y1, f1, _ = step_fn(drift, t0, y0, f0, dt)
        return (t1, y1, f1), y1

    t0 = ts[0]
    f0 = drift(t0, y0)
    init_carry = (t0, y0, f0)
    _, ys = lax.scan(scan_fun, init_carry, (ts[1:], dts))
    return jnp.concatenate((y0[None], ys))


def _odeint_adaptive(
    drift: Callable,
    y0: Array,
    ts: Array,
    step_fn: Callable,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    order: int = 2,
    dtinit: Optional[float] = None,
    dtmin: float = -jnp.inf,
    dtmax: float = jnp.inf,
    maxerror: float = 1.2,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.1,
):
    def scan_fn(carry, data):
        t0, y0, f0, dt = carry
        t1 = data

        def cond_fun(state):
            i, t0, _, _, _ = state
            return (i < mxstep) & (t0 < t1)

        def body_fn(state):
            i, t0, y0, f0, dt = state

            y1, f1, (error, k) = step_fn(drift, t0, y0, f0, dt)
            error = mean_error_ratio(error, rtol, atol, y0, y1)
            # print(error)
            dt = step_size_adaption(
                dt,
                error,
                order=order,
                safety=safety,
                ifactor=ifactor,
                dfactor=dfactor,
                dtmin=dtmin,
                dtmax=dtmax,
            )
            dt = lax.cond(t0 + dt < t1, lambda _: dt, lambda _: t1 - t0, None)

            # This rejects the step if the error is too large
            # Maybe we should still accept the step, but with a smaller step size?
            # This would accoumulate error, but would be more efficient

            new = [i + 1, t0 + dt, y1, f1, dt]
            old = [i + 1, t0, y0, f0, dt]
            return tuple(map(partial(jnp.where, error <= maxerror), new, old))

        init_state = (0, t0, y0, f0, dt)
        iter, _, y1, f1, dt = lax.while_loop(cond_fun, body_fn, init_state)

        return (t1, y1, f1, dt), y1

    t0 = ts[0]
    f0 = drift(t0, y0)

    if dtinit is None:
        dt = initial_step_size(drift, t0, y0, order, rtol, atol, f0)
    else:
        dt = dtinit

    init_carry = (t0, y0, f0, dt)
    _, ys = lax.scan(scan_fn, init_carry, ts[1:])

    return jnp.concatenate((y0[None], ys))


def _odeint(
    drift,
    y0: PyTree[Array],
    ts: Array,
    *args,
    method="rk4",
    dt: Optional[Float] = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    mxstep: int = jnp.inf,
    dtmin: float = 0.0,
    dtmax: float = jnp.inf,
    maxerror: float = 1.2,
    safety: float = 0.95,
    ifactor: float = 10.0,
    dfactor: float = 0.1,
):
    """Solve an ordinary differential equation.

    This function assumes that y0 is a single initial value, and that ts is a single time grid, with a single set of parameters.
    If you want to solve multiple ODEs, or multiple time grids, use vmap!

    NOTE: You need to use partial(odeint, keywords) to vmap this function with keywords i.e. method="rk4".

    Args:
        drift (Callable): Drift function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        args (Any): Additional arguments to pass to the drift function i.e. if it is parametrized.
        method (str, optional): Methods to use. Defaults to "euler".
        dt (Optional[Float], optional): Fixed step size. If it is an adaptive solver then this will be used as initializer. Defaults to None.
        rtol (float, optional): Relative tolerance (only relevant for adaptive solvers). Defaults to 1e-3.
        atol (float, optional): Absolute tolerance (only relevant for adaptive solvers). Defaults to 1e-3.
        mxstep (int, optional): Maximum number of steps (only relevant for adaptive solvers). Defaults to jnp.inf.
        dtmin (float, optional): Minimum step size (only relevant for adaptive solvers). Defaults to 0.0.
        dtmax (float, optional): Maximum step size (only relevant for adaptive solvers). Defaults to jnp.inf.
        maxerror (float, optional): Maximum error (only relevant for adaptive solvers). Defaults to 1.2.
        safety (float, optional): Safety factor (only relevant for adaptive solvers). Defaults to 0.95.
        ifactor (float, optional): Increase factor (only relevant for adaptive solvers). Defaults to 50.0.
        dfactor (float, optional): Decrease factor (only relevant for adaptive solvers). Defaults to 0.05.

    Returns:
        Array: Solution of the ODE.
    """
    # Flatten the initial value and time grid
    _flatten, _unflatten = flatten1d(y0)

    y0 = jnp.atleast_1d(_flatten(y0))
    ts = jnp.atleast_1d(ts)

    # Consistent dtype, based on the initial value.
    dtype = y0.dtype
    ts = ts.astype(dtype)
    _f = lambda t, y: jnp.atleast_1d(_flatten(drift(t, _unflatten(y), *args))).astype(
        dtype
    )
    step_fn = get_step_fn(method, dtype=dtype)
    method_info = get_method_info(method)

    # Minimum step size, based on the dtype
    adaptive = method_info["adaptive"]
    dtmin = jnp.maximum(dtmin, jnp.finfo(dtype).eps)

    # Solve the ODE
    if not adaptive:
        # Solvers without adaptive step size.
        if dt is None:
            # Use the provided time grid
            time_grid = ts
            ys = _odeint_on_grid(_f, y0, time_grid, step_fn)
        else:
            # Use uniform time grid, with specified step size
            time_grid = jnp.arange(ts[0], ts[-1] + dt, dt)
            ys = _odeint_on_grid(_f, y0, time_grid, step_fn)
            f_sol = jax.vmap(linear_interpolation(time_grid, ys))
            ys = f_sol(ts)
    else:
        # Solvers with adaptive step size.
        order = method_info["order"]
        ys = _odeint_adaptive(
            _f,
            y0,
            ts,
            step_fn,
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            order=order,
            dtinit=dt,
            dtmin=dtmin,
            dtmax=dtmax,
            maxerror=maxerror,
            safety=safety,
            ifactor=ifactor,
            dfactor=dfactor,
        )

    # Unflatten the solution
    ys = jax.vmap(_unflatten)(ys)
    return ys


def _inv_odeint(drift, ys: Array, ts: Array, *args, **kwargs):
    y0 = ys[-1]
    xs = _odeint(drift, y0, ts[::-1], *args, **kwargs)
    return xs[-1]


def _inv_logdet_odeint(drift, ys, ts, *args, **kwargs):
    _jac = jax.jacfwd(drift, argnums=1)
    jac = lambda t, x: jnp.atleast_2d(_jac(t, x))

    def aug_drift(t, state, *args):
        x, logdet = state
        dx = jnp.atleast_1d(drift(t, x, *args))
        dlogdet = jnp.atleast_1d(jnp.trace(jac(t, x)))
        return dx, dlogdet

    y0 = ys[-1]
    logdet0 = jnp.zeros(y0.shape[:-1])
    xs, logdets = _odeint(aug_drift, (y0, logdet0), ts[::-1], *args, **kwargs)

    return xs[-1], logdets[-1]


# ODEs are invertible, so we can define the inverse of the ODE solver
odeint = _odeint
odeint = custom_inverse(_odeint, static_argnums=(0,), inv_argnum=1)
odeint.definv(_inv_odeint)
odeint.definv_and_logdet(_inv_logdet_odeint)
