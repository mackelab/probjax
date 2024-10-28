from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

# TODO Refactor


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
        drift (Callable): _description_
        diffusion (Callable): _description_
        t0 (Array): _description_
        y0 (Array): _description_
        f0 (Array): _description_
        g0 (Array): _description_
        dt (Array): _description_
        dWt (Array): _description_
        dWtdWs (Array): _description_
        c0 (Array): _description_
        c1 (Array): _description_
        A0 (Array): _description_
        A1 (Array): _description_
        B0 (Array): _description_
        B1 (Array): _description_
        b_sol (Array): _description_
        gamma0 (Array): _description_
        gamma1 (Array): _description_
        b_error (Array): _description_
        order (int): _description_
        is_diagonal (bool, optional): _description_. Defaults to False.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    dtsqrt = jnp.sqrt(jnp.abs(dt))
    dtsqrt_vec = jnp.ones_like(dWt) * dtsqrt
    m = dWt.shape[0]
    d = y0.shape[0]

    if is_diagonal:  # noqa: SIM108
        reduction_dWt = "s, smi, j -> i"
    else:
        # General case not working yet ...
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
    k2 = (
        jnp.zeros((stages, m) + g0.shape, g0.dtype).at[0, :].set(g0)
    )  # Diffusion evaluations at support points

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


# Strong order 1.0 methods

# # SRI1
# c0 = jnp.zeros((3,))
# c1 = jnp.zeros((3,))
# A0 = jnp.zeros((3, 3))
# A1 = jnp.zeros((3, 3))
# B0 = jnp.zeros((3, 3))
# B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
# b_sol = jnp.array([1, 0, 0])
# gamma0 = jnp.array([1, 0, 0])
# gamma1 = jnp.array([0, 0.5, -0.5])
# b_error = None
# register_stochastic_runge_kutta_method(
#     "sri1", c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error
# )

# # SRI2
# c0 = jnp.array([0, 1, 0.0])
# c1 = jnp.array([0, 1, 1.0])
# A0 = jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
# A1 = jnp.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
# B0 = jnp.zeros((3, 3))
# B1 = jnp.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
# b_sol = jnp.array([0.5, 0.5, 0])
# gamma0 = jnp.array([1.0, 0, 0])
# gamma1 = jnp.array([0, 0.5, -0.5])
# b_error = None
# register_stochastic_runge_kutta_method(
#     "sri2", c0, c1, A0, A1, B0, B1, b_sol, gamma0, gamma1, b_error
# )
