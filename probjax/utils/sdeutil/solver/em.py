from functools import partial

import jax.numpy as jnp

from probjax.utils._solver_common import make_trivial_init, sample_wiener_increment
from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.odeutil.solvers.rk_explicit import RKInfo, RKState, heun
from probjax.utils.sdeutil.base import (
    SDEInfo,
    SDESolverAPI,
    SDEState,
    register_method,
)
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey


class EulerMaruyamaInfo(SDEInfo):
    dWt: Array


class EulerMaruyamaState(SDEState):
    t0: Array
    y0: Array


init_state = make_trivial_init(EulerMaruyamaState)


def build_em_step(
    drift: Callable,
    diffusion: Callable,
    noise_dim: int | None = None,
    **kwargs,
) -> Callable[
    [RngKey, EulerMaruyamaState, float], tuple[EulerMaruyamaState, EulerMaruyamaInfo]
]:
    del kwargs

    def step_fn(
        rng: RngKey, state: EulerMaruyamaState, dt: float
    ) -> tuple[EulerMaruyamaState, EulerMaruyamaInfo]:
        t0, y0 = state.t0, state.y0
        f0 = jnp.asarray(drift(t0, y0))
        g0 = jnp.asarray(diffusion(t0, y0))

        dWt = sample_wiener_increment(rng, g0, y0.shape[0], dt, noise_dim)
        y1 = y0 + dt * f0 + mv_diag_or_dense(g0, dWt)
        new_state = EulerMaruyamaState(t0 + dt, y1)
        info = EulerMaruyamaInfo(dWt=dWt)
        return new_state, info

    return step_fn


info = {
    "order": 1,
    "strong_order": 0.5,
    "weak_order": 1,
    "adaptive": False,
}


class euler_maruyama(SDESolverAPI):
    init = init_state
    build_step = build_em_step


register_method("euler_maruyama", euler_maruyama, info=info)


RKMaruyamaState = RKState


class RKMaruyamaInfo(SDEState):
    dWt: Array
    rk_info: RKInfo


def build_rkm_step(
    method,
    drift: Callable,
    diffusion: Callable,
    noise_dim: int | None = None,
    **kwargs,
):
    del kwargs
    solver = method(drift)

    def step_fn(rng: RngKey, state: RKMaruyamaState, dt: ArrayLike):
        t0, y0 = state.t0, state.y0
        g0 = jnp.asarray(diffusion(t0, y0))
        dWt = sample_wiener_increment(rng, g0, y0.shape[0], dt, noise_dim)
        new_state, info = solver(state, dt)
        diffusion_term = mv_diag_or_dense(g0, dWt)
        new_state = new_state._replace(y0=new_state.y0 + diffusion_term)
        info = RKMaruyamaInfo(dWt=dWt, rk_info=info)
        return new_state, info

    return step_fn


class heun_maruyama(SDESolverAPI):
    init = heun.init
    build_step = partial(build_rkm_step, heun)
