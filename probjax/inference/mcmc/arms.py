from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

from probjax.inference.mcmc.base import make_kernel_api
from probjax.inference.rejection import univariate as uni


class ARMSParams(NamedTuple):
    pass


class ARMSState(NamedTuple):
    position: Array
    logdensity: Array
    envelope: uni.ARSState


class ARMSInfo(NamedTuple):
    accepted: Array
    proposal: Array


def _ensure_scalar(x: ArrayLike) -> Array:
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x
    if x.size == 1:
        return jnp.reshape(x, ())
    raise ValueError("ARMS/A2RMS only support scalar targets")


def _init_points(position: Array, rng_key: RngKey):
    eps = jax.random.normal(rng_key, (3,)) * 0.1
    return position + eps


def init(
    position: ArrayLike,
    logdensity_fn: Callable,
    rng_key: RngKey | None = None,
    init_points: ArrayLike | None = None,
    lb: ArrayLike = -jnp.inf,
    ub: ArrayLike = jnp.inf,
    max_points: int = 50,
) -> ARMSState:
    position = _ensure_scalar(position)
    if init_points is None:
        if rng_key is None:
            raise ValueError("Provide rng_key or init_points for ARMS initialization")
        init_points = _init_points(position, rng_key)
    init_points = jnp.asarray(init_points)
    env = uni.init_ars_state(logdensity_fn, init_points, lb=lb, ub=ub, max_points=max_points)
    logdensity = logdensity_fn(position)
    return ARMSState(position, logdensity, env)


def init_params(state: PyTree) -> ARMSParams:
    return ARMSParams()


def _arms_step(logdensity_fn: Callable, *, control: bool) -> Callable:
    def step(key: RngKey, state: ARMSState, params: ARMSParams) -> Tuple[ARMSState, ARMSInfo]:
        key, sample_key, accept_key, control_key = jax.random.split(key, 4)
        xt, _ = uni.sample_upper(state.envelope, sample_key)
        log_xt, grad_xt = jax.value_and_grad(logdensity_fn)(xt)
        log_q_xt = uni._log_proposal(state.envelope, xt)
        log_q_x = uni._log_proposal(state.envelope, state.position)

        log_alpha = log_xt - state.logdensity + log_q_x - log_q_xt
        accept = jnp.log(jax.random.uniform(accept_key)) < log_alpha

        x_next = jnp.where(accept, xt, state.position)
        log_next = jnp.where(accept, log_xt, state.logdensity)

        if control:
            gap = log_q_xt - log_xt
            do_update = jnp.logical_or(
                ~accept, jnp.log(jax.random.uniform(control_key)) < gap
            )
        else:
            do_update = ~accept

        env_next = jax.lax.cond(
            do_update,
            lambda s: uni.update_ars_state(s, xt, log_xt, grad_xt),
            lambda s: s,
            state.envelope,
        )
        return ARMSState(x_next, log_next, env_next), ARMSInfo(accept, xt)

    return step


arms = make_kernel_api(
    name="arms",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=lambda logdensity_fn, **_: _arms_step(logdensity_fn, control=False),
)


a2rms = make_kernel_api(
    name="a2rms",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=lambda logdensity_fn, **_: _arms_step(logdensity_fn, control=True),
)
