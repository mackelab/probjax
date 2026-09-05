from functools import partial

import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey
from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.odeutil.solvers.rk_explicit import RKInfo, RKState, heun
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method


class EulerMaruyamaInfo(SDEInfo):
    dWt: Array


class EulerMaruyamaState(SDEState):
    t0: Array
    y0: Array


def init_state(t0: ArrayLike, y0: ArrayLike, **kwargs) -> EulerMaruyamaState:
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    return EulerMaruyamaState(t0, y0)


def _infer_noise_dim(g0: Array, default_dim: int, configured_dim: int | None) -> int:
    """Infer Brownian dimension from diffusion coefficients."""
    if configured_dim is not None:
        return int(configured_dim)
    if g0.ndim <= 1:
        return int(default_dim)
    return int(g0.shape[-1])


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

        inferred_noise_dim = _infer_noise_dim(g0, y0.shape[0], noise_dim)
        dWt = jax.random.normal(rng, (inferred_noise_dim,)) * jnp.sqrt(jnp.abs(dt))
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
        inferred_noise_dim = _infer_noise_dim(g0, y0.shape[0], noise_dim)
        dWt = jax.random.normal(rng, (inferred_noise_dim,)) * jnp.sqrt(jnp.abs(dt))
        new_state, info = solver(state, dt)
        diffusion_term = mv_diag_or_dense(g0, dWt)
        new_state = new_state._replace(y0=new_state.y0 + diffusion_term)
        info = RKMaruyamaInfo(dWt=dWt, rk_info=info)
        return new_state, info

    return step_fn


class heun_maruyama(SDESolverAPI):
    init = heun.init
    build_step = partial(build_rkm_step, heun)
