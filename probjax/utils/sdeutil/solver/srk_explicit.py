
from functools import partial
import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey
from probjax.utils.brownian import get_iterated_integrals_fn
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method


class SRKInfo(SDEInfo):
    dWt: Array
    dWtdWs: Array
    k1: Array
    k2: Array


class SRKState(SDEState):
    t0: Array
    y0: Array


def init_state(t0: ArrayLike, y0: ArrayLike, **kwargs) -> SRKState:
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    return SRKState(t0, y0)


def build_sri1_coefficients(
    dtype: ArrayLike = jnp.float32,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Build the Butcher tableau coefficients for the SRI1 method.

    Args:
        dtype (jnp.dtype, optional): Data type for the coefficients. Defaults to jnp.float32.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
            (c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error)
    """
    c0 = jnp.zeros((3,), dtype=dtype)
    c1 = jnp.zeros((3,), dtype=dtype)
    A0 = jnp.zeros((3, 3), dtype=dtype)
    A1 = jnp.zeros((3, 3), dtype=dtype)
    B0 = jnp.zeros((3, 3), dtype=dtype)
    B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=dtype)
    b_sol = jnp.array([1, 0, 0], dtype=dtype)
    gamma0 = jnp.array([1, 0, 0], dtype=dtype)
    gamma1 = jnp.array([0, 0.5, -0.5], dtype=dtype)
    b_error = None
    return c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error


def build_sri2_coefficients(
    dtype: ArrayLike = jnp.float32,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Build the Butcher tableau coefficients for the SRI2 method.

    Args:
        dtype (jnp.dtype, optional): Data type for the coefficients. Defaults to jnp.float32.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
            (c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error)
    """
    c0 = jnp.array([0, 1, 0.0], dtype=dtype)
    c1 = jnp.array([0, 1, 1.0], dtype=dtype)
    A0 = jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=dtype)
    A1 = jnp.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=dtype)
    B0 = jnp.zeros((3, 3), dtype=dtype)
    B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=dtype)
    b_sol = jnp.array([0.5, 0.5, 0], dtype=dtype)
    gamma0 = jnp.array([1.0, 0, 0], dtype=dtype)
    gamma1 = jnp.array([0, 0.5, -0.5], dtype=dtype)
    b_error = None
    return c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error


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
    stages: int,
    is_diagonal: bool = False,
    *kwargs,
):
    """Explicit stochastic Runge-Kutta method.

    Paper: https://preprint.math.uni-hamburg.de/public/papers/prst/prst2010-02.pdf

    Args:
        drift (Callable): Drift function f(t, y)
        diffusion (Callable): Diffusion function g(t, y)
        t0 (Array): Initial time
        y0 (Array): Initial state
        f0 (Array): Initial drift evaluation
        g0 (Array): Initial diffusion evaluation
        dt (Array): Time step
        dWt (Array): Brownian increments
        dWtdWs (Array): Iterated integrals
        c0 (Array): Time coefficients for drift stages
        c1 (Array): Time coefficients for diffusion stages
        A0 (Array): Butcher tableau for drift
        A1 (Array): Butcher tableau for diffusion
        B0 (Array): Butcher tableau for drift-diffusion coupling
        B1 (Array): Butcher tableau for diffusion-diffusion coupling
        b_sol (Array): Solution weights for drift
        gamma0 (Array): Solution weights for diffusion
        gamma1 (Array): Solution weights for diffusion
        b_error (Array): Error weights (optional)
        stages (int): Number of stages
        is_diagonal (bool): Whether the noise is diagonal
        *kwargs: Additional arguments

    Returns:
        Tuple[Array, Array, Array, Tuple]: (y1, f1, g1, (y1_error, k1, k2))
    """
    dtsqrt = jnp.sqrt(jnp.abs(dt))
    dtsqrt_vec = jnp.ones_like(dWt) * dtsqrt
    m = dWt.shape[0]
    d = y0.shape[0]

    if is_diagonal:
        reduction_dWt = "s, smi, j -> i"
    else:
        reduction_dWt = (
            "s, smij, j -> i"  # Average drift evaluation over s, then matmul with dWt
        )
    diffusion_vec = jax.vmap(diffusion, in_axes=(None, 0))  # Vectorize diffusion

    def body_fun(i, data):
        k1, k2 = data
        ti1 = t0 + dt * c0[i]
        ti2 = t0 + dt * c1[i]

        yi1 = (
            y0
            + jnp.dot(A0[i, :], k1) * dt
            + 1 / d * jnp.einsum(reduction_dWt, B0[i, :], k2, dWt)
        )

        yi2 = y0 + jnp.dot(A1[i, :], k1) * dt
        yi2 = jnp.broadcast_to(yi2, (m,) + yi2.shape)

        for k in range(m):
            res = jnp.einsum(
                reduction_dWt, B1[i, :], k2, jnp.atleast_1d(dWtdWs[k, ...])
            ) / jnp.sqrt(dt)
            yi2 = yi2.at[k, ...].add(res)

        ft = drift(ti1, yi1)
        gt = diffusion_vec(ti2, yi2)
        return k1.at[i, ...].set(ft), k2.at[i, ...].set(gt)

    # Drift evaluations at support points
    k1 = jnp.zeros((stages,) + f0.shape, f0.dtype).at[0, :].set(f0)
    # Diffusion evaluations at support points
    k2 = jnp.zeros((stages, m) + g0.shape, g0.dtype).at[0, :].set(g0)

    k1, k2 = jax.lax.fori_loop(1, stages + 1, body_fun, (k1, k2))

    y1 = (
        y0
        + jnp.dot(b_sol, k1) * dt
        + 1 / d * jnp.einsum(reduction_dWt, gamma0, k2, dWt)
        + 1 / d * jnp.einsum(reduction_dWt, gamma1, k2, dtsqrt_vec)
    )

    f1 = drift(t0 + dt, y1)
    g1 = diffusion(t0 + dt, y1)

    if b_error is None:
        y1_error = None
    else:
        raise NotImplementedError

    return y1, f1, g1, (y1_error, k1, k2)


def build_srk_step(
    drift: Callable,
    diffusion: Callable,
    noise_type="diagonal",
    sde_type="ito",
    iterated_integrals_fn=get_iterated_integrals_fn,
    stages: int = 3,
    build_coefficients: Callable = None,
):
    """Build a step function for the explicit stochastic Runge-Kutta method.

    Args:
        drift (Callable): Drift function
        diffusion (Callable): Diffusion function
        noise_type (str, optional): Type of noise. Defaults to "diagonal".
        sde_type (str, optional): Type of SDE. Defaults to "ito".
        iterated_integrals_fn (Callable, optional): Function to compute iterated integrals. Defaults to get_iterated_integrals_fn.
        stages (int, optional): Number of stages. Defaults to 3.
        build_coefficients (Callable): Function that builds the Butcher tableau coefficients.

    Returns:
        Callable: Step function for the SRK method
    """
    iterated_integrals_fn = iterated_integrals_fn(noise_type, sde_type)
    is_diagonal = noise_type == "diagonal"

    # Get coefficients from builder function
    c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error = build_coefficients()

    def step_fn(rng: RngKey, state: SRKState, dt: float):
        dt = jnp.asarray(dt)
        t0, y0 = state.t0, state.y0
        rng1, rng2 = jax.random.split(rng, 2)

        f0 = jnp.asarray(drift(t0, y0))
        g0 = jnp.asarray(diffusion(t0, y0))
        dWt = jax.random.normal(rng1, y0.shape) * jnp.sqrt(jnp.abs(dt))
        dWtdWs = iterated_integrals_fn(rng2, dWt, jnp.abs(dt))

        y1, f1, g1, (y1_error, k1, k2) = explicit_stochastic_runge_kutta_step(
            drift,
            diffusion,
            t0,
            y0,
            f0,
            g0,
            dt,
            dWt,
            dWtdWs,
            c0,
            c1,
            A0,
            A1,
            B0,
            B1,
            b_sol,
            gamma0,
            gamma1,
            b_error,
            stages,
            is_diagonal,
        )

        new_state = SRKState(t0 + dt, y1)
        info = SRKInfo(dWt=dWt, dWtdWs=dWtdWs, k1=k1, k2=k2)
        return new_state, info

    return step_fn


class sri1(SDESolverAPI):
    """SRI1 method - Strong order 1.0 stochastic Runge-Kutta method."""

    init = init_state
    build_step = partial(build_srk_step, build_coefficients=build_sri1_coefficients)


class sri2(SDESolverAPI):
    """SRI2 method - Strong order 1.0 stochastic Runge-Kutta method."""

    init = init_state
    build_step = partial(build_srk_step, build_coefficients=build_sri2_coefficients)


# Register the SRI1 method
register_method(
    "sri1",
    sri1,
    info={"order": 1, "strong_order": 1.0, "weak_order": 1.0, "adaptive": False},
)

# Register the SRI2 method
register_method(
    "sri2",
    sri2,
    info={"order": 1, "strong_order": 1.0, "weak_order": 1.0, "adaptive": False},
)
