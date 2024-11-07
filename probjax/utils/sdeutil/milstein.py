from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

from probjax.utils.brownian import get_iterated_integrals_fn
from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method


class MilsteinInfo(SDEInfo):
    dWt: Array
    dWtdWs: Array


class MilsteinState(SDEState):
    t0: Array
    y0: Array


def init_state(t0: Array, y0: Array, **kwargs):
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    return MilsteinState(t0, y0)


def build_milstein_step(
    drift: Callable,
    diffusion: Callable,
    noise_type="diagonal",
    sde_type="ito",
    jac_fn: Callable = jax.jacfwd,
    iterated_integrals_fn=get_iterated_integrals_fn,
):
    g_jac = jac_fn(
        lambda t, x: jnp.sum(jnp.atleast_1d(diffusion(t, x)), axis=0), argnums=1
    )

    iterated_integrals_fn = iterated_integrals_fn(noise_type, sde_type)
    is_diagonal = noise_type == "diagonal"

    def step_fn(rng: Key, state: MilsteinState, dt: float):
        dt = jnp.asarray(dt)
        t0, y0 = state.t0, state.y0
        rng1, rng2 = jax.random.split(rng, 2)
        f0 = jnp.asarray(drift(t0, y0))
        g0 = jnp.asarray(diffusion(t0, y0))
        g0_jac = g_jac(t0, y0)
        dWt = jax.random.normal(rng1, y0.shape) * jnp.sqrt(jnp.abs(dt))
        dWtdWs = iterated_integrals_fn(rng2, dWt, jnp.abs(dt))

        drift_term = dt * f0
        diffusion_term1 = mv_diag_or_dense(g0, dWt)

        if is_diagonal:
            diffusion_term2 = g0_jac * dWtdWs * g0
        else:
            diffusion_term2 = jnp.einsum("nm,mm,mn -> n", g0_jac.T, dWtdWs, g0)

        y1 = y0 + drift_term + diffusion_term1 + diffusion_term2

        new_state = MilsteinState(t0 + dt, y1)
        info = MilsteinInfo(dWt, dWtdWs)
        return new_state, info

    return step_fn


class milstein(SDESolverAPI):
    init = init_state
    build_step = build_milstein_step


register_method(
    "milstein",
    milstein,
    info={"order": 1, "strong_order": 1.0, "weak_order": 1.0, "adaptive": False},
)
