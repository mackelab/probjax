import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax import lax
from jax import core
from jax.tree_util import tree_leaves

from functools import partial
from jaxtyping import Array, Float, PyTree, Int
from typing import Callable, Optional
from jax.random import PRNGKeyArray

from probjax.utils.linalg import is_matrix, is_triangular_matrix


METHOD_STEP_FN = {}
METHOD_INFO = {}


def register_method(name: str, func: Callable, info: Optional[dict] = None) -> Callable:
    """This function registers a general step_fn for a method.

    Args:
        name (str): Name of the method.
        func (Callable): Step function. The function must have the following signature:
            def step_fn(
                drift: Callable,
                diffusion: Callable,
                t0: Array,
                y0: Array,
                f0: Array,
                g0: Array,
                dt: Array,
                dWt: Array,
                dWtdWs: Array,
            )
        info (Optional[dict], optional): Additional information about the method. Defaults to None.

    Returns:
        Callable: step_fn
    """
    METHOD_STEP_FN[name] = func
    METHOD_INFO[name] = info
    return func


def register_stochastic_runge_kutta_method(
    name: str,
    c0: Array,
    c1: Array,
    A0: Array,
    A1: Array,
    B0: Array,
    B1: Array,
    b_sol: Array,
    gamma0: Array,
    gamma1: Array,
    b_error: Optional[Array] = None,
    order: Optional[int] = None,
    strong_order: Optional[int] = None,
    weak_order: Optional[int] = None,
):
    order = len(c0)

    # TODO: Check if A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error are valid
    # TODO: Check if order, strong_order, weak_order are valid
    if is_triangular_matrix(A0) and jnp.diag(A1).sum() == 0:
        explicit = True
    else:
        explicit = False

    if b_error is None:
        adaptive = False
    else:
        adaptive = True

    METHOD_INFO[name] = {
        "order": order,  # This is the deterministic order
        "strong_order": strong_order,  # Stochastic strong order
        "weak_order": weak_order,  # Stochastic weak order
        "A0": A0,
        "A1": A1,
        "B0": B0,
        "B1": B1,
        "b_sol": b_sol,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "b_error": b_error,
        "adaptive": adaptive,
    }

    if explicit:
        step_fn = partial(
            explicit_stochastic_runge_kutta_step,
            c0=c0,
            c1=c1,
            A0=A0,
            A1=A1,
            B0=B0,
            B1=B1,
            b_sol=b_sol,
            gamma0=gamma0,
            gamma1=gamma1,
            b_error=b_error,
            order=order,
        )
    else:
        raise NotImplementedError

    METHOD_STEP_FN[name] = step_fn

    return step_fn


def get_step_fn(method: str, dtype: Optional[Float] = None) -> Callable:
    """Returns the step function for a given method.

    Returns:
        Callable: Step function with corresponding method name.
    """
    step_fn = METHOD_STEP_FN[method]
    if dtype is not None:
        # Right numerical precision
        if hasattr(step_fn, "keywords"):
            step_fn.keywords["c0"] = step_fn.keywords["c0"].astype(dtype)
            step_fn.keywords["c1"] = step_fn.keywords["c1"].astype(dtype)
            step_fn.keywords["A0"] = step_fn.keywords["A0"].astype(dtype)
            step_fn.keywords["A1"] = step_fn.keywords["A1"].astype(dtype)
            step_fn.keywords["B0"] = step_fn.keywords["B0"].astype(dtype)
            step_fn.keywords["B1"] = step_fn.keywords["B1"].astype(dtype)
            step_fn.keywords["gamma0"] = step_fn.keywords["gamma0"].astype(dtype)
            step_fn.keywords["gamma1"] = step_fn.keywords["gamma1"].astype(dtype)
            step_fn.keywords["b_sol"] = step_fn.keywords["b_sol"].astype(dtype)
            if step_fn.keywords["b_error"] is not None:
                step_fn.keywords["b_error"] = step_fn.keywords["b_error"].astype(dtype)

    return step_fn


def get_method_info(method: str) -> Callable:
    """Returns the info for a given method."""
    return METHOD_INFO[method]


def get_methods():
    """Returns the list of available methods."""
    return list(METHOD_STEP_FN.keys())


@partial(jax.jit, static_argnums=(0, 1, 19, 20))
def explicit_stochastic_runge_kutta_step(
    drift: Callable,
    diffusion: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    g0: Array,
    dt: Array,
    dWt: Array,
    dWtdWs: Array,
    c0: Array,
    c1: Array,
    A0: Array,
    A1: Array,
    B0: Array,
    B1: Array,
    b_sol: Array,
    gamma0: Array,
    gamma1: Array,
    b_error: Array,
    order: int,
    diagonal_diffusion_matrix: bool = False,
):
    dtsqrt = jnp.sqrt(dt)
    dtsqrt_vec = jnp.ones_like(dWt) * dtsqrt
    if diagonal_diffusion_matrix:
        reduction1 = "i, ij -> j"
        reduction2 = "i,i -> i"
    else:
        reduction1 = "i, ijk -> jk"
        reduction2 = "ij, j -> i"

    def body_fun(i, data):
        k1, k2 = data
        ti1 = t0 + dt * c0[i - 1]
        ti2 = t0 + dt * c1[i - 1]

        yi1 = (
            y0
            + dt * jnp.dot(A0[i - 1, :], k1)
            + jnp.einsum(reduction2, jnp.einsum(reduction1, B0[i - 1, :], k2), dWt)
        )
        yi2 = (
            y0
            + dt * jnp.dot(A1[i - 1, :], k1)
            + jnp.einsum(
                reduction2, jnp.einsum(reduction1, B1[i - 1, :], k2), dtsqrt_vec
            )
        )

        ft = drift(ti1, yi1)
        gt = diffusion(ti2, yi2)
        return k1.at[i, :].set(ft), k2.at[i, :].set(gt)

    k1 = (
        jnp.zeros((order,) + f0.shape, f0.dtype).at[0, :].set(f0)
    )  # Drift evaluations at support points
    k2 = (
        jnp.zeros((order,) + g0.shape, g0.dtype).at[0, :].set(g0)
    )  # Diffusion evaluations at support points
    k1, k2 = lax.fori_loop(1, order, body_fun, (k1, k2))

    y1 = (
        y0
        + dt * jnp.dot(b_sol, k1)
        + jnp.einsum(reduction2, jnp.einsum(reduction1, gamma0, k2), dWt)
    )

    if dWtdWs is not None:
        # TODO Implement
        y1 += dWtdWs * jnp.dot(gamma1, k2)

    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)

    if b_error is None:
        y1_error = None
    else:
        raise NotImplementedError

    return y1, f1, g1, (y1_error, k1, k2)


# Strong order 0.5 methods


# Euler-Maruyama method
# We use a custom implementation to avoid the overhead of the general method
@partial(jax.jit, static_argnums=(0, 1, 9))
def _euler_maruyama_step_fn(
    drift: Callable,
    diffusion: Callable,
    t0: Array,
    y0: Array,
    f0: Array,
    g0: Array,
    dt: Array,
    dWt: Array,
    dWtdWs: Array,
    diagonal_diffusion_matrix: bool = False,
):
    if diagonal_diffusion_matrix:
        reduction = "i,i -> i"
    else:
        reduction = "ij, j -> i"

    y1 = y0 + dt * f0 + jnp.einsum(reduction, g0, dWt)
    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)
    return y1, f1, g1, None


info = {
    "order": 1,
    "strong_order": 0.5,
    "weak_order": 1,
    "A0": jnp.array([[0.0]]),
    "A1": jnp.array([[0.0]]),
    "B0": jnp.array([[0.0]]),
    "B1": jnp.array([[0.0]]),
    "b_sol": jnp.array([1.0]),
    "gamma0": jnp.array([1.0]),
    "gamma1": jnp.array([1.0]),
    "b_error": None,
    "adaptive": False,
}
register_method("euler_maruyama", _euler_maruyama_step_fn, info)

# Strong order 1.0 methods

# SRK3
c0 = jnp.array([0.0, 2 / 3, 2 / 3])
c1 = jnp.array([0.0, 1.0, 1.0])
A0 = jnp.array([[0.0, 0.0, 0.0], [2 / 3, 0.0, 0.0], [-1 / 3, 1.0, 0.0]])
A1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
B1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
b_sol = jnp.array([1 / 4, 1 / 2, 1 / 4])
gamma0 = jnp.array([1 / 2, 1 / 4, 1 / 4])
gamma1 = jnp.array([0.0, 1 / 2, -1 / 2])
b_error = None
register_stochastic_runge_kutta_method(
    "srk3(2)", c0, c1, A0, A1, B1, B1, b_sol, gamma0, gamma1, b_error
)


@partial(jax.jit, static_argnums=(1, 2, 5))
def _sdeint_on_grid(
    key: PRNGKeyArray,
    drift: Callable,
    diffusion: Callable,
    y0: Array,
    ts: Array,
    step_fn: Callable,
) -> Array:
    """Solve a stochastic differential equation on a grid.

    Args:
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments.

    Returns:
        Array: Solution of the SDE.
    """

    dts = ts[1:] - ts[:-1]

    def scan_fun(carry, data):
        key, y0, t0, f0, g0 = carry
        t1, dt = data

        # Generate brownian increments
        key, subkey = jrandom.split(key)
        dWt = jrandom.normal(subkey, (noise_dim,)) * jnp.sqrt(dt)
        # TODO Iterated Brownian increments ...

        y1, f1, g1, _ = step_fn(
            drift,
            diffusion,
            t0,
            y0,
            f0,
            g0,
            dt,
            dWt,
            None,
            diagonal_diffusion_matrix=diagonal_diffusion_matrix,
        )

        return (key, y1, t1, f1, g1), y1

    t0 = ts[0]
    f0 = drift(t0, y0)
    g0 = diffusion(t0, y0)

    if g0.ndim < 2:
        noise_dim = 1
        diagonal_diffusion_matrix = True
    elif g0.ndim == 2:
        noise_dim = g0.shape[1]
        diagonal_diffusion_matrix = False
    else:
        raise ValueError("Diffusion function must return a vector or matrix")

    init_carry = (key, y0, t0, f0, g0)
    _, ys = lax.scan(scan_fun, init_carry, (ts[1:], dts))
    return jnp.concatenate((y0[None], ys))


def sdeint(
    key: PRNGKeyArray,
    drift: Callable,
    diffusion: Callable,
    y0: Array,
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    type: str = "ito",
    diagonal_noise: bool = False,
    dt: Optional[Float] = None,
    rtol: Float = 1e-6,
    atol: Float = 1e-6,
    mxstep: Int = jnp.inf,
    dtmin: Float = 0.0,
    dtmax: Float = jnp.inf,
) -> Array:
    """Solve a stochastic differential equation.

    Args:
        key: (PRNGKeyArray): Random generator key.
        drift (Callable): Drift function.
        diffusion (Callable): Diffusion function.
        y0 (Array): Initial value.
        ts (Array): Time points.
        *args: Other arguments, that are passed both to the drift and diffusion functions i.e. parameters!
        method (str, optional): Methods to use. Defaults to "euler_maruyama".
        diagonal_noise (bool, optional): Whether the noise is diagonal. Defaults to False.
        dt (Optional[Float], optional): Fixed step size (optionally infered from ts). Defaults to None.
        rtol (Float, optional): Relative tolerance. Defaults to 1e-6.
        atol (Float, optional): Absolute tolerance. Defaults to 1e-6.
        mxstep (Int, optional): Maximum number of steps used by solver. Defaults to jnp.inf.
        dtmin (Float, optional): Minimal step size used by solver. Defaults to 0..
        dtmax (Float, optional): Maximal step size used by solver. Defaults to jnp.inf.

    Raises:
        TypeError: The arguments passed not jax types.
        TypeError: The arguments passed not jax types.

    Returns:
        ys: Solution path of the SDE.
    """
    y0 = jnp.atleast_1d(y0)
    ts = jnp.atleast_1d(ts)

    # Consistent dtype, based on the initial value.
    dtype = y0.dtype
    ts = ts.astype(dtype)

    # Make sure drift is consistent and is a function _f: R x R^d -> R^d where d >= 1
    # Make sure diffusion is consistent and is a function _g: R x R^d -> R^d (independent noise) where d >= 1 or _g: R x R^d -> R^{d x d} where d >= 1 (correlated noise)
    _f = lambda t, y: jnp.atleast_1d(drift(t, y, *args)).astype(dtype)
    _g = lambda t, y: jnp.atleast_1d(diffusion(t, y, *args)).astype(dtype)

    # Get step_fn
    step_fn = get_step_fn(method, dtype=dtype)
    method_info = get_method_info(method)

    # Minimum step size, based on the dtype
    adaptive = method_info["adaptive"]
    dtmin = jnp.maximum(dtmin, jnp.finfo(dtype).eps)

    return _sdeint_on_grid(key, _f, _g, y0, ts, step_fn)
