from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from probjax.utils.functions import additive_diffusion, split_drift
from probjax.utils.linalg import mv_diag_or_dense
from probjax.utils.sdeutil.base import SDEInfo, SDESolverAPI, SDEState, register_method
from probjax.utils.typing import Array, ArrayLike, Callable, RngKey


class ExpEulerMaruyamaState(SDEState):
    t0: Array
    y0: Array
    f0: Optional[Array]
    g0: Optional[Array]


class ExpEulerMaruyamaInfo(SDEInfo):
    dWt: Array
    lin_coeff: Array


def _phi1_scalar(z: Array) -> Array:
    small = jnp.abs(z) < 1e-4
    series = 1.0 + 0.5 * z + (z * z) / 6.0 + (z * z * z) / 24.0
    return jnp.where(small, series, jnp.expm1(z) / z)


def _as_scalar_coeff(c: Array) -> Array:
    c = jnp.asarray(c)
    if c.ndim == 0:
        return c
    if c.size == 1:
        return jnp.reshape(c, ())
    raise ValueError("split_drift.lin_coeff must return a scalar")


def _weighted_noise_variance(c: Array, dt: ArrayLike) -> Array:
    """Variance of the exponential-weighted Ito noise increment."""
    dt_abs = jnp.abs(jnp.asarray(dt))
    z = c * dt_abs
    var = dt_abs * _phi1_scalar(2.0 * z)
    return jnp.maximum(var, 0.0)


def _infer_noise_dim(g0: Array, default_dim: int, configured_dim: int | None) -> int:
    if configured_dim is not None:
        return int(configured_dim)
    if g0.ndim <= 1:
        return int(default_dim)
    return int(g0.shape[-1])


def init_state(
    t0: ArrayLike,
    y0: ArrayLike,
    *args,
    drift: Optional[Callable] = None,
    diffusion: Optional[Callable] = None,
    **kwargs,
) -> ExpEulerMaruyamaState:
    del kwargs
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)

    if not isinstance(drift, split_drift):
        raise TypeError("exp_euler_maruyama requires `drift` to be a `split_drift`")

    f0 = jnp.asarray(drift.nonlin(t0, y0, *args))
    if diffusion is None:
        g0 = None
    else:
        g0 = jnp.asarray(diffusion(t0, y0, *args))

    return ExpEulerMaruyamaState(t0=t0, y0=y0, f0=f0, g0=g0)


def build_exp_euler_maruyama_step(
    drift: Callable,
    diffusion: Callable,
    noise_dim: int | None = None,
    **kwargs,
):
    del kwargs
    if not isinstance(drift, split_drift):
        raise TypeError("exp_euler_maruyama requires `drift` to be a `split_drift`")

    split = drift
    is_additive = isinstance(diffusion, additive_diffusion)

    def step_fn(rng: RngKey, state: ExpEulerMaruyamaState, dt: ArrayLike):
        dt = jnp.asarray(dt)
        t0, y0 = state.t0, state.y0

        N_n = state.f0 if state.f0 is not None else jnp.asarray(split.nonlin(t0, y0))
        c_np1 = _as_scalar_coeff(split.lin_coeff(t0 + dt))

        z = c_np1 * dt
        r = jnp.exp(z)
        ph1 = _phi1_scalar(z)

        y_det = r * y0 + dt * ph1 * N_n

        if state.g0 is None:
            g_n = jnp.asarray(diffusion(t0, y0))
        else:
            g_n = state.g0

        inferred_noise_dim = _infer_noise_dim(g_n, y0.shape[0], noise_dim)
        if is_additive:
            noise_var = _weighted_noise_variance(c_np1, dt)
            noise_std = jnp.sqrt(noise_var)
        else:
            noise_std = jnp.sqrt(jnp.abs(dt))

        dWt = jax.random.normal(rng, (inferred_noise_dim,)) * noise_std
        y_np1 = y_det + mv_diag_or_dense(g_n, dWt)

        N_np1 = jnp.asarray(split.nonlin(t0 + dt, y_np1))
        if is_additive:
            g_np1 = jnp.asarray(diffusion(t0 + dt, y0))
        else:
            g_np1 = jnp.asarray(diffusion(t0 + dt, y_np1))

        new_state = ExpEulerMaruyamaState(
            t0=t0 + dt,
            y0=y_np1,
            f0=N_np1,
            g0=g_np1,
        )
        info = ExpEulerMaruyamaInfo(dWt=dWt, lin_coeff=c_np1)
        return new_state, info

    return step_fn


class exp_euler_maruyama(SDESolverAPI):
    init = staticmethod(init_state)
    build_step = staticmethod(build_exp_euler_maruyama_step)


register_method(
    "exp_euler_maruyama",
    exp_euler_maruyama,
    info={
        "order": 1,
        "strong_order": 0.5,
        "weak_order": 1.0,
        "adaptive": False,
        "info": "Exponential Euler-Maruyama (requires split_drift; additive_diffusion optional)",
    },
)
