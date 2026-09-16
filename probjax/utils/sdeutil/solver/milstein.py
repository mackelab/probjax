import jax
import jax.numpy as jnp

from probjax.utils._solver_common import make_trivial_init, sample_wiener_increment
from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.sdeutil.base import (
    SDEInfo,
    SDESolverAPI,
    SDEState,
    register_method,
)
from probjax.utils.sdeutil.brownian import get_iterated_integrals_fn
from probjax.utils.typing import Array, Callable, RngKey


class MilsteinInfo(SDEInfo):
    dWt: Array
    dWtdWs: Array


class MilsteinState(SDEState):
    t0: Array
    y0: Array


init_state = make_trivial_init(MilsteinState)


def build_milstein_step(
    drift: Callable,
    diffusion: Callable,
    noise_type: str = "diagonal",
    noise_dim: int | None = None,
    sde_type: str = "ito",
    jac_fn: Callable = jax.jacfwd,
    iterated_integrals_fn=get_iterated_integrals_fn,
) -> Callable[[RngKey, MilsteinState, float], tuple[MilsteinState, MilsteinInfo]]:
    iterated_integrals_fn = iterated_integrals_fn(noise_type, sde_type)
    is_diagonal = noise_type == "diagonal"

    if is_diagonal:
        g_jac = jac_fn(
            lambda t, x: jnp.sum(jnp.atleast_1d(diffusion(t, x)), axis=0), argnums=1
        )
    else:
        g_jac = jac_fn(lambda t, x: jnp.asarray(diffusion(t, x)), argnums=1)

    def step_fn(
        rng: RngKey, state: MilsteinState, dt: float
    ) -> tuple[MilsteinState, MilsteinInfo]:
        dt = jnp.asarray(dt)
        t0, y0 = state.t0, state.y0
        rng1, rng2 = jax.random.split(rng, 2)
        f0 = jnp.asarray(drift(t0, y0))
        g0 = jnp.asarray(diffusion(t0, y0))
        g0_jac = g_jac(t0, y0)

        dWt = sample_wiener_increment(rng1, g0, y0.shape[0], dt, noise_dim)
        dWtdWs = iterated_integrals_fn(rng2, dWt, jnp.abs(dt))

        drift_term = dt * f0
        diffusion_term1 = mv_diag_or_dense(g0, dWt)

        if is_diagonal:
            diffusion_term2 = g0_jac * dWtdWs * g0
        else:
            diffusion_term2 = jnp.einsum("ka,ibk,ab->i", g0, g0_jac, dWtdWs)

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
