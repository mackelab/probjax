from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.odeutil.rk_explicit import RKInfo, RKState, heun
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method


class EulerMaruyamaInfo(SDEInfo):
    dWt: Array


class EulerMaruyamaState(SDEState):
    t0: Array
    y0: Array


def init_state(t0: Array, y0: Array, **kwargs):
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    return EulerMaruyamaState(t0, y0)


def build_em_step(drift: Callable, diffusion: Callable, **kwargs):
    def step_fn(rng: Key, state: EulerMaruyamaState, dt: float):
        t0, y0 = state.t0, state.y0
        dWt = jax.random.normal(rng, y0.shape) * jnp.sqrt(jnp.abs(dt))
        f0 = drift(t0, y0)
        g0 = diffusion(t0, y0)
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


def build_rkm_step(method, drift: Callable, diffusion: Callable):
    solver = method(drift)

    def step_fn(rng: Key, state: RKMaruyamaState, dt: float):
        t0, y0 = state.t0, state.y0
        dWt = jax.random.normal(rng, y0.shape) * jnp.sqrt(jnp.abs(dt))
        new_state, info = solver(state, dt)
        g0 = diffusion(t0, y0)
        diffusion_term = mv_diag_or_dense(g0, dWt)
        new_state = new_state._replace(y0=new_state.y0 + diffusion_term)
        info = RKMaruyamaInfo(dWt=dWt, rk_info=info)
        return new_state, info

    return step_fn


class heun_maruyama(SDESolverAPI):
    init = heun.init
    build_step = partial(build_rkm_step, heun)
